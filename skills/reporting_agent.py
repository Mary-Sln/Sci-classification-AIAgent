"""
Reporting Agent
Compiles results from baseline correction, preprocessing, feature selection,
and classification into a single markdown report with supporting plots.

IMPORTANT: This report documents the CURRENT state of the pipeline. Per
PROJECT.md Step 4, if the best classifier's accuracy is below the ~80%
reference threshold, this report should be treated as an INTERIM checkpoint,
not a final result -- the next step in the agent's loop is a literature
search for alternative methods, followed by re-evaluation.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SKILLS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SKILLS_DIR)
RES_DIR = os.path.join(BASE_DIR, "results")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

PERFORMANCE_THRESHOLD = 0.80  # reference accuracy threshold from PROJECT.md


def load_csv_safe(path, label):
    if not os.path.exists(path):
        print(f"[-] Missing: {path}. Cannot include {label} in report.")
        return None
    return pd.read_csv(path)


def plot_model_comparison(df_class, save_path):
    """Bar chart comparing all classifiers on accuracy and F1_Macro side by side."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df_class))
    width = 0.35

    df_sorted = df_class.sort_values(by="Composite_Score", ascending=False)
    ax.bar(x - width/2, df_sorted["Accuracy"], width, label="Accuracy", color="#2196F3")
    ax.bar(x + width/2, df_sorted["F1_Macro"], width, label="F1 (Macro)", color="#FF9800")

    ax.axhline(y=PERFORMANCE_THRESHOLD, color='red', linestyle='--', linewidth=1,
               label=f"Reference threshold ({PERFORMANCE_THRESHOLD:.0%})")

    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted["Model"], rotation=30, ha='right')
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Classifier Comparison: Accuracy vs F1 (Macro)", fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[*] Plot saved: {save_path}")


def plot_confusion_matrix(cm_path, save_path):
    """Heatmap of the best model's confusion matrix (last CV fold)."""
    if not os.path.exists(cm_path):
        print(f"[-] Missing confusion matrix: {cm_path}. Skipping heatmap.")
        return
    cm_df = pd.read_csv(cm_path, index_col=0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_df.values, cmap='Blues')

    ax.set_xticks(range(len(cm_df.columns)))
    ax.set_yticks(range(len(cm_df.index)))
    ax.set_xticklabels(cm_df.columns, rotation=45, ha='right')
    ax.set_yticklabels(cm_df.index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (best model, last CV fold)", fontsize=12, fontweight='bold')

    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            val = cm_df.values[i, j]
            color = "white" if val > cm_df.values.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[*] Plot saved: {save_path}")


def build_report_text(df_baseline, df_prep, df_feat, df_class, best_model_name,
                       best_accuracy, threshold_met):
    lines = []
    lines.append("# Coffee Authentication — Raman Spectral Classification Report\n")
    lines.append("*Interim checkpoint report — see Status section below.*\n")

    lines.append("## 1. Data Summary")
    lines.append("130 Raman spectral samples across 6 groups: 5 pure-origin coffee groups "
                 "(Arabica, Festtags, Guatemala, Lavazza, Sumatra — 20 samples each) and 1 "
                 "Adulteration group (30 samples). Each spectrum spans 785 wavenumber channels.\n")

    lines.append("## 2. Baseline Correction")
    if df_baseline is not None:
        best_baseline = df_baseline.sort_values(by="Composite_Score", ascending=False).iloc[0]
        lines.append(f"Six baseline-correction methods were compared via Stratified 5-Fold CV "
                     f"using Logistic Regression as a probe classifier. **{best_baseline['Method']}** "
                     f"performed best (accuracy {best_baseline['Accuracy']:.1%}, composite score "
                     f"{best_baseline['Composite_Score']:.3f}).\n")
        lines.append(df_baseline.sort_values(by="Composite_Score", ascending=False)
                     .to_markdown(index=False))
        lines.append("")
        lines.append("Note: airPLS scored below \"No correction,\" likely because its default "
                     "smoothing parameter (lambda=1e5) over-suppressed real spectral peaks for "
                     "this dataset rather than just the baseline drift.\n")
    else:
        lines.append("*(baseline_comparison.csv not found)*\n")

    lines.append("## 3. Advanced Preprocessing")
    if df_prep is not None:
        best_prep = df_prep.sort_values(by="Optimization_Score", ascending=False).iloc[0]
        lines.append(f"Seven preprocessing combinations (scatter correction, smoothing, "
                     f"derivatives) were compared on top of the winning baseline correction. "
                     f"**{best_prep['Pipeline_Combination']}** performed best (accuracy "
                     f"{best_prep['Accuracy']:.1%}). All methods were evaluated fold-safe — "
                     f"any fitted reference (e.g. MSC's reference spectrum) was computed from "
                     f"training folds only, never the full dataset, to avoid leakage.\n")
        lines.append(df_prep.sort_values(by="Optimization_Score", ascending=False)
                     .to_markdown(index=False))
        lines.append("")
        lines.append("Note: the top four methods are within ~1.5 percentage points of each "
                     "other given the small sample size (~22 samples/class on average), so this "
                     "ranking should be read as a reasonable best guess, not a strong, certain "
                     "finding.\n")
    else:
        lines.append("*(preprocessing_optimization_results.csv not found)*\n")

    lines.append("## 4. Feature Selection")
    if df_feat is not None:
        best_feat = df_feat.sort_values(by="Optimization_Score", ascending=False).iloc[0]
        lines.append(f"SelectKBest (ANOVA F-test) was evaluated across K = "
                     f"{sorted(df_feat['K'].tolist())} channels, fit strictly inside each CV "
                     f"fold to prevent leakage. **K={int(best_feat['K'])}** performed best "
                     f"(accuracy {best_feat['Accuracy']:.1%}).\n")
        lines.append(df_feat.sort_values(by="Optimization_Score", ascending=False)
                     .to_markdown(index=False))
        lines.append("")
        lines.append("Note: smaller K values (30-200) underperformed using ALL 785 channels "
                     "with no selection at all -- this suggests the spectral information that "
                     "discriminates coffee origin is spread across many channels rather than "
                     "concentrated in a small number of peaks, consistent with the visual "
                     "similarity observed between the 5 pure-origin mean spectra.\n")
    else:
        lines.append("*(feature_selection_results.csv not found)*\n")

    lines.append("## 5. Classification")
    if df_class is not None:
        lines.append("Six classifiers were evaluated on the selected feature set via fold-safe "
                     "Stratified CV:\n")
        lines.append(df_class.sort_values(by="Composite_Score", ascending=False)
                     .to_markdown(index=False))
        lines.append("")
        lines.append(f"**Best model: {best_model_name}** (accuracy {best_accuracy:.1%}).\n")
        lines.append("Caveats on these results:\n")
        lines.append("- The feature subset (K) was chosen using Logistic Regression as the "
                     "probe classifier during feature selection, which gives Logistic "
                     "Regression a built-in advantage in this comparison. SIMCA and PLS-DA "
                     "performing competitively *despite* this is notable.")
        lines.append("- LDA's relatively low accuracy may reflect its equal-covariance "
                     "assumption being violated: the Adulteration group shows substantially "
                     "higher within-class spectral variance than the pure-origin groups.")
        lines.append("- SVM (RBF) was run with default hyperparameters (C, gamma) and likely "
                     "underperforms its real potential on this small, high-dimensional dataset; "
                     "it has not yet been tuned.")
        lines.append("- SIMCA and PLS-DA are custom implementations (no native scikit-learn "
                     "equivalent) — see code documentation for methodology and limitations.")
        lines.append("- The saved confusion matrix reflects only the LAST cross-validation "
                     "fold, not an aggregate across all folds — useful for a quick look, not a "
                     "rigorous summary.\n")
    else:
        lines.append("*(classification_results.csv not found)*\n")

    lines.append("## 6. Status & Next Step")
    if threshold_met:
        lines.append(f"Best accuracy ({best_accuracy:.1%}) meets the reference threshold "
                     f"({PERFORMANCE_THRESHOLD:.0%}). Pipeline loop complete.\n")
    else:
        lines.append(f"**Best accuracy ({best_accuracy:.1%}) is below the reference threshold "
                     f"({PERFORMANCE_THRESHOLD:.0%}).** Per the agent's design "
                     f"(analyze -> is performance acceptable? -> search literature -> evaluate "
                     f"new methods -> repeat), this is NOT a final result. The next step is a "
                     f"literature search for alternative preprocessing or classification "
                     f"techniques used in comparable Raman/IR coffee-authentication studies, "
                     f"followed by re-evaluation.\n")

    lines.append("## 7. Figures")
    lines.append("![Classifier Comparison](classifier_comparison.png)\n")
    lines.append("![Confusion Matrix](confusion_matrix.png)\n")

    return "\n".join(lines)


def main():
    print("[+] Reporting Agent Initiated...")

    df_baseline = load_csv_safe(os.path.join(RES_DIR, "baseline_comparison.csv"), "baseline comparison")
    df_prep = load_csv_safe(os.path.join(RES_DIR, "preprocessing_optimization_results.csv"), "preprocessing")
    df_feat = load_csv_safe(os.path.join(RES_DIR, "feature_selection_results.csv"), "feature selection")
    df_class = load_csv_safe(os.path.join(RES_DIR, "classification_results.csv"), "classification")

    if df_class is None:
        print("[-] Cannot generate report without classification_results.csv. Exiting.")
        sys.exit(1)

    df_class_sorted = df_class.sort_values(by="Composite_Score", ascending=False)
    best_model_name = df_class_sorted.iloc[0]["Model"]
    best_accuracy = df_class_sorted.iloc[0]["Accuracy"]
    threshold_met = best_accuracy >= PERFORMANCE_THRESHOLD

    print(f"[+] Best model: {best_model_name} | Accuracy: {best_accuracy:.1%}")
    print(f"[+] Threshold ({PERFORMANCE_THRESHOLD:.0%}) met: {threshold_met}")

    # --- Generate plots ---
    plot_model_comparison(df_class, os.path.join(REPORTS_DIR, "classifier_comparison.png"))

    best_model_safe = best_model_name.replace(" ", "_")
    cm_path = os.path.join(RES_DIR, f"confusion_matrix_{best_model_safe}.csv")
    plot_confusion_matrix(cm_path, os.path.join(REPORTS_DIR, "confusion_matrix.png"))

    # --- Write report ---
    report_text = build_report_text(df_baseline, df_prep, df_feat, df_class,
                                     best_model_name, best_accuracy, threshold_met)
    report_path = os.path.join(REPORTS_DIR, "REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n[+] Report written to {report_path}")
    if not threshold_met:
        print(f"[!] NOTE: best accuracy ({best_accuracy:.1%}) is below the "
              f"{PERFORMANCE_THRESHOLD:.0%} reference threshold. This report is an INTERIM "
              f"checkpoint. Next step per PROJECT.md: literature search for alternative "
              f"methods, then re-evaluate.")


if __name__ == "__main__":
    main()