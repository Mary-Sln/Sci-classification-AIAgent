import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/headless use
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Path configurations
SKILLS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SKILLS_DIR)
RES_DIR = os.path.join(BASE_DIR, "results")

# --- Scatter Correction and Smoothing Methods ---
# NOTE: each function below is written to be FOLD-SAFE. Where a method needs
# a reference computed from data (MSC), the reference must be passed in or
# fit only on the training fold — never on the full dataset before CV splits.

def apply_snv(X):
    """Standard Normal Variate (SNV): row-wise mean/std normalization.
    Row-wise only -- no cross-sample information used, so this is safe to
    apply to train and test independently with no leakage risk."""
    mu = np.mean(X, axis=1, keepdims=True)
    sigma = np.std(X, axis=1, keepdims=True)
    return (X - mu) / (sigma + 1e-10)

def fit_msc_reference(X_train):
    """Fit the MSC reference spectrum using ONLY the training fold."""
    return np.mean(X_train, axis=0)

def apply_msc(X, ref):
    """Multiplicative Scatter Correction (MSC) against a given reference
    spectrum. The reference must be fit on training data only (see
    fit_msc_reference) and then applied to both train and test."""
    X_msc = np.zeros_like(X)
    for i in range(X.shape[0]):
        fit = np.polyfit(ref, X[i], 1)
        X_msc[i] = (X[i] - fit[1]) / (fit[0] + 1e-10)
    return X_msc

def apply_sg(X, window_length, polyorder, deriv):
    """Savitzky-Golay smoothing/derivative. Row-wise only -- safe."""
    return savgol_filter(X, window_length=window_length, polyorder=polyorder,
                          deriv=deriv, axis=1)

# --- Pipeline definitions ---
# Each pipeline is a function (X_train, X_test) -> (X_train_proc, X_test_proc)
# so that anything requiring a fitted reference is fit on X_train only.

def pipeline_baseline_only(X_train, X_test):
    return X_train, X_test

def pipeline_snv(X_train, X_test):
    return apply_snv(X_train), apply_snv(X_test)

def pipeline_msc(X_train, X_test):
    ref = fit_msc_reference(X_train)
    return apply_msc(X_train, ref), apply_msc(X_test, ref)

def pipeline_sg_smooth(X_train, X_test):
    return (apply_sg(X_train, 15, 2, 0), apply_sg(X_test, 15, 2, 0))

def pipeline_snv_sg_smooth(X_train, X_test):
    Xtr_snv, Xte_snv = apply_snv(X_train), apply_snv(X_test)
    return (apply_sg(Xtr_snv, 15, 2, 0), apply_sg(Xte_snv, 15, 2, 0))

def pipeline_sg_deriv1(X_train, X_test):
    return (apply_sg(X_train, 11, 2, 1), apply_sg(X_test, 11, 2, 1))

def pipeline_sg_deriv2(X_train, X_test):
    return (apply_sg(X_train, 11, 3, 2), apply_sg(X_test, 11, 3, 2))

PIPELINES = {
    "Baseline Only": pipeline_baseline_only,
    "Baseline + SNV": pipeline_snv,
    "Baseline + MSC": pipeline_msc,
    "Baseline + SG Smoothing (W=15, P=2)": pipeline_sg_smooth,
    "Baseline + SNV + SG Smoothing": pipeline_snv_sg_smooth,
    "Baseline + SG 1st Deriv (W=11, P=2)": pipeline_sg_deriv1,
    "Baseline + SG 2nd Deriv (W=11, P=3)": pipeline_sg_deriv2,
}


def evaluate_pipeline(X_raw, y_labels, pipeline_fn):
    """Stratified 5-Fold CV evaluation using Logistic Regression as probe
    classifier. The pipeline_fn is applied INSIDE each fold so that any
    fitted reference (e.g. MSC) only ever sees the training fold."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s, sens, specs, precs = [], [], [], [], []

    for train_idx, test_idx in skf.split(X_raw, y_labels):
        X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
        y_train, y_test = y_labels[train_idx], y_labels[test_idx]

        # Apply preprocessing fold-safe: fit only on train, transform both
        X_train, X_test = pipeline_fn(X_train_raw, X_test_raw)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = LogisticRegression(max_iter=1500, random_state=42)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average='macro'))
        sens.append(recall_score(y_test, y_pred, average='macro'))
        precs.append(precision_score(y_test, y_pred, average='macro', zero_division=0))

        cm = confusion_matrix(y_test, y_pred)
        fp = cm.sum(axis=0) - np.diag(cm)
        fn = cm.sum(axis=1) - np.diag(cm)
        tp = np.diag(cm)
        tn = cm.sum() - (fp + fn + tp)
        specs.append(np.mean(tn / (tn + fp + 1e-8)))

    return np.mean(accs), np.mean(f1s), np.mean(sens), np.mean(specs), np.mean(precs)


def build_final_matrix(X_raw, pipeline_name):
    """Apply the WINNING pipeline to the full dataset for downstream steps
    (feature selection, classification). This is NOT used for scoring --
    it's standard practice to refit on all available data once the method
    has already been selected via fold-safe CV above."""
    pipeline_fn = PIPELINES[pipeline_name]
    X_final, _ = pipeline_fn(X_raw, X_raw[:1])  # reuse fn; test half unused here
    if pipeline_name == "Baseline + MSC":
        ref = fit_msc_reference(X_raw)
        return apply_msc(X_raw, ref)
    elif pipeline_name == "Baseline + SNV":
        return apply_snv(X_raw)
    elif pipeline_name == "Baseline + SG Smoothing (W=15, P=2)":
        return apply_sg(X_raw, 15, 2, 0)
    elif pipeline_name == "Baseline + SNV + SG Smoothing":
        return apply_sg(apply_snv(X_raw), 15, 2, 0)
    elif pipeline_name == "Baseline + SG 1st Deriv (W=11, P=2)":
        return apply_sg(X_raw, 11, 2, 1)
    elif pipeline_name == "Baseline + SG 2nd Deriv (W=11, P=3)":
        return apply_sg(X_raw, 11, 3, 2)
    else:
        return X_raw


def generate_plots(X_raw, X_best, y_labels, frequencies, best_name, class_colors):
    """
    Generate and save two figures:
      Figure 1: Overlay of raw spectra grouped by class (mean ± std ribbon)
      Figure 2: Side-by-side raw vs best preprocessed spectra for one sample per class
    """
    classes = np.unique(y_labels)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(
        "Coffee Spectral Profiles: Raw vs Best Preprocessed",
        fontsize=14, fontweight='bold', y=1.01
    )

    for ax, X_data, title in zip(
        axes,
        [X_raw, X_best],
        ["Raw Spectra (Baseline-Corrected)", f"After {best_name}"]
    ):
        for cls in classes:
            mask = y_labels == cls
            spectra_cls = X_data[mask]
            mean_spectrum = np.mean(spectra_cls, axis=0)
            std_spectrum = np.std(spectra_cls, axis=0)
            color = class_colors.get(cls, 'black')
            ax.plot(frequencies, mean_spectrum, label=cls, linewidth=1.5, color=color)
            ax.fill_between(frequencies, mean_spectrum - std_spectrum,
                             mean_spectrum + std_spectrum, alpha=0.15, color=color)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=10)
        ax.set_ylabel("Intensity (a.u.)", fontsize=10)
        ax.legend(loc='upper left', fontsize=8, framealpha=0.7)
        ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    fig.savefig(os.path.join(RES_DIR, "raw_vs_preprocessed.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("[*] Plot saved: results/raw_vs_preprocessed.png")

    n_classes = len(classes)
    fig2, axes2 = plt.subplots(n_classes, 1, figsize=(14, 3 * n_classes), sharex=True)
    fig2.suptitle(
        f"Per-Class Spectral Fingerprints: Raw (grey) vs {best_name} (color)",
        fontsize=13, fontweight='bold'
    )

    for i, cls in enumerate(classes):
        ax = axes2[i]
        mask = y_labels == cls
        color = class_colors.get(cls, '#333333')

        for spec in X_raw[mask]:
            ax.plot(frequencies, spec, color='#cccccc', linewidth=0.5, alpha=0.6)
        for spec in X_best[mask]:
            ax.plot(frequencies, spec, color=color, linewidth=0.7, alpha=0.7)
        ax.plot(frequencies, np.mean(X_best[mask], axis=0),
                color=color, linewidth=2.0, label=f"{cls} (mean preprocessed)")

        ax.set_ylabel(cls, fontsize=9, fontweight='bold', rotation=0, labelpad=60, va='center')
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.legend(loc='upper right', fontsize=7, framealpha=0.6)

    axes2[-1].set_xlabel("Wavenumber (cm⁻¹)", fontsize=10)
    plt.tight_layout()
    fig2.savefig(os.path.join(RES_DIR, "per_class_fingerprints.png"), dpi=200, bbox_inches='tight')
    plt.close(fig2)
    print("[*] Plot saved: results/per_class_fingerprints.png")


def main():
    print("[+] Preprocessing Agent Initiated...")

    X_path = os.path.join(RES_DIR, "X_preprocessed.npy")
    y_path = os.path.join(RES_DIR, "y_labels.npy")
    fr_path = os.path.join(RES_DIR, "frequencies.npy")

    for p in [X_path, y_path]:
        if not os.path.exists(p):
            print(f"[-] Missing: {p}. Please run baseline_agent.py first.")
            sys.exit(1)

    X_base = np.load(X_path)
    y_labels = np.load(y_path)
    frequencies = np.load(fr_path) if os.path.exists(fr_path) else np.arange(X_base.shape[1])

    print(f"[+] Loaded baseline-corrected matrix: {X_base.shape} | Labels: {len(y_labels)}")

    optimization_results = []
    print("\n[->] Screening Preprocessing Combinations (fold-safe evaluation)...")

    for name, pipeline_fn in PIPELINES.items():
        print(f"    Evaluating: {name}")
        acc, f1, sen, spec, prec = evaluate_pipeline(X_base, y_labels, pipeline_fn)
        score = (0.40 * acc) + (0.25 * f1) + (0.15 * sen) + (0.15 * spec) + (0.05 * prec)
        optimization_results.append({
            "Pipeline_Combination": name,
            "Accuracy": acc, "F1_Macro": f1,
            "Sensitivity": sen, "Specificity": spec,
            "Precision": prec, "Optimization_Score": score
        })

    df_results = pd.DataFrame(optimization_results).sort_values(
        by="Optimization_Score", ascending=False
    )
    df_results.to_csv(os.path.join(RES_DIR, "preprocessing_optimization_results.csv"), index=False)

    best_pipeline_name = df_results.iloc[0]["Pipeline_Combination"]
    best_score = df_results.iloc[0]["Optimization_Score"]

    # Refit the winning method on the FULL dataset for downstream steps.
    # This is standard practice after model/method selection -- not used
    # for scoring, so it's not leakage.
    X_best = build_final_matrix(X_base, best_pipeline_name)

    print(f"\n[+] Optimization Complete!")
    print(f"    Supreme Combination : {best_pipeline_name}")
    print(f"    Highest Composite Score: {best_score:.4f}")
    print(df_results.to_string(index=False))

    np.save(os.path.join(RES_DIR, "X_final_chemometrics.npy"), X_best)
    with open(os.path.join(RES_DIR, "best_preprocessing_method.txt"), "w") as f:
        f.write(best_pipeline_name)
    print("[*] Locked matrix saved to results/X_final_chemometrics.npy")

    CLASS_COLORS = {
        "Arabica": "#2196F3",
        "Adulteration": "#F44336",
        "Festtags": "#9C27B0",
        "Guatemala": "#4CAF50",
        "Lavazza": "#FF9800",
        "Sumatra": "#795548",
    }

    print("\n[->] Generating diagnostic plots...")
    generate_plots(X_base, X_best, y_labels, frequencies, best_pipeline_name, CLASS_COLORS)
    print("[+] All plots saved to results/")


if __name__ == "__main__":
    main()