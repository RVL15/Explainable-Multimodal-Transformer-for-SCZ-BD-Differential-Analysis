"""
Explainable Multimodal Transformer - Unseen EEG Inference Script

Executes genuine, trained model inference on unseen EEG recordings without retraining.
Supports: .mat, .edf, .bdf, .fif, .npy, .csv
Results are saved to: backend/data/unseen/results/
"""

import sys
import os
import argparse
import json
from pathlib import Path

# Add backend directory to Python sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.config import UNSEEN_INCOMING_DIR, UNSEEN_RESULTS_DIR, CHECKPOINTS_DIR
from app.models.pipeline import run_unseen_eeg_inference


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run trained model inference on unseen EEG recordings without retraining."
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Path to unseen EEG file, or filename inside backend/data/unseen/incoming/. "
             "If not specified, all valid recordings in backend/data/unseen/incoming/ will be processed."
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        default="SZ_VS_HC",
        choices=["SZ_VS_HC", "SCZ_VS_BD"],
        help="Target classification task: 'SZ_VS_HC' (ASEEG trained) or 'SCZ_VS_BD' (B-SNIP 1)."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Uncertainty rejection threshold (0.50 - 1.00). Default: 0.65"
    )
    parser.add_argument(
        "--save-processed",
        action="store_true",
        help="Save preprocessed EEG window arrays to backend/data/unseen/processed/."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print("  EXPLAINABLE MULTIMODAL TRANSFORMER: UNSEEN EEG INFERENCE PIPELINE")
    print("  Scientific Standard: Zero Retraining | Zero Data Leakage | Genuine Checkpoints")
    print("=" * 80)
    print(f"Task:                  {args.task}")
    print(f"Uncertainty Threshold: {args.threshold}")
    print(f"Incoming Directory:    {UNSEEN_INCOMING_DIR}")
    print(f"Results Directory:     {UNSEEN_RESULTS_DIR}")
    print("-" * 80)

    # 1. Collect target files
    files_to_process = []
    if args.file:
        p = Path(args.file)
        if not p.is_absolute() and not p.exists() and (UNSEEN_INCOMING_DIR / p).exists():
            p = UNSEEN_INCOMING_DIR / p
        if not p.exists():
            print(f"[ERROR] File not found: {args.file}")
            sys.exit(1)
        files_to_process.append(p)
    else:
        supported_exts = {".edf", ".bdf", ".fif", ".mat", ".npy", ".csv"}
        for item in sorted(UNSEEN_INCOMING_DIR.iterdir()):
            if item.is_file() and item.name != ".gitkeep" and item.suffix.lower() in supported_exts:
                files_to_process.append(item)

        if not files_to_process:
            print("[INFO] No unseen EEG recordings found in backend/data/unseen/incoming/.")
            print("Please place .mat, .edf, .bdf, .fif, .npy, or .csv files in that folder.")
            sys.exit(0)

    print(f"Total files to process: {len(files_to_process)}\n")

    # 2. Execute inference for each file
    success_count = 0
    failure_count = 0

    for idx, fpath in enumerate(files_to_process, 1):
        print(f"[{idx}/{len(files_to_process)}] Processing: {fpath.name}")
        try:
            result = run_unseen_eeg_inference(
                file_path=fpath,
                target_task=args.task,
                uncertainty_threshold=args.threshold,
                save_result=True,
                save_processed=args.save_processed
            )

            print("   -> Prediction Result:")
            print(f"      Predicted Class:    {result['predicted_class']}")
            print(f"      Class Scores:       {json.dumps(result['class_scores'])}")
            print(f"      Confidence:         {result['confidence']:.4f}")
            print(f"      Uncertainty Status: {result['uncertainty_status']}")
            print(f"      Model Checkpoint:   {result['model_checkpoint']}")
            print(f"      Retrained:          {result['retrained_during_inference']} (Strictly False)")
            print(f"      Usable Windows:     {result['usable_windows']} (2.0s clean segments)")
            if result.get("top_contributing_channels"):
                print(f"      Top Channels (IG):  {', '.join(result['top_contributing_channels'])}")
            print(f"      Result Saved To:    {result.get('saved_result_path')}")
            print()
            success_count += 1

        except Exception as e:
            print(f"   -> [FAILED] {str(e)}\n")
            failure_count += 1

    print("=" * 80)
    print(f"Inference Complete: {success_count} succeeded, {failure_count} failed.")
    print("=" * 80)
    if failure_count > 0 and success_count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
