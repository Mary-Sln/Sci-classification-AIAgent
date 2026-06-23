"""
Sci-Classification Pipeline Orchestrator
Phase 1: Baseline Correction + Advanced Preprocessing (stops for user review)
Phase 2: Feature Selection + Classification + Reporting (run after approval)
"""
import os
import sys
import subprocess
import argparse

SKILLS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(SKILLS_DIR)

def run_agent(script_name):
    script_path = os.path.join(SKILLS_DIR, script_name)
    print(f"\n{'='*60}")
    print(f"  Running: {script_name}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=BASE_DIR
    )
    if result.returncode != 0:
        print(f"\n[-] FATAL: {script_name} exited with code {result.returncode}. Stopping pipeline.")
        sys.exit(result.returncode)
    print(f"[+] {script_name} completed successfully.")

def phase1():
    print("\n" + "#"*60)
    print("  PHASE 1: Baseline Correction + Advanced Preprocessing")
    print("#"*60)
    run_agent("baseline_agent.py")
    run_agent("preprocessing_agent.py")
    print("\n" + "="*60)
    print("  PHASE 1 COMPLETE — Stopping for user review.")
    print("  Review the following in results/:")
    print("    • baseline_comparison.csv          (ranked baseline methods)")
    print("    • preprocessing_optimization_results.csv (ranked preprocessing)")
    print("    • raw_vs_preprocessed.png          (class mean spectra overlay)")
    print("    • per_class_fingerprints.png       (per-class raw vs processed)")
    print("  Approve results before running Phase 2.")
    print("="*60 + "\n")

def phase2():
    print("\n" + "#"*60)
    print("  PHASE 2: Feature Selection + Classification + Reporting")
    print("#"*60)
    run_agent("feature_selection_agent.py")
    run_agent("classification_agent.py")
    run_agent("reporting_agent.py")
    print("\n" + "="*60)
    print("  PHASE 2 COMPLETE — Full pipeline finished.")
    print("  Review the final report in reports/REPORT.md")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sci-Classification Pipeline Orchestrator")
    parser.add_argument(
        "--phase", choices=["1", "2", "all"], default="1",
        help="Which phase to run: '1' = Baseline+Preprocessing, "
             "'2' = FeatureSelection+Classification+Reporting, 'all' = both"
    )
    args = parser.parse_args()

    if args.phase in ("1", "all"):
        phase1()
    if args.phase == "2":
        phase2()
    if args.phase == "all":
        phase2()
