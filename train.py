"""
AstroClassify - Unified Machine Learning Training CLI
Supports: CatBoost, PyTorch Deep MLP, or Combined Pipeline
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="AstroClassify: Train Space Object Classification Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        choices=["catboost", "pytorch", "all"],
        default="catboost",
        help="Algorithm to train (catboost, pytorch, or all)"
    )
    parser.add_argument("--epochs", type=int, default=80, help="Epochs for PyTorch")
    parser.add_argument("--iterations", type=int, default=1200, help="Iterations for CatBoost")
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for PyTorch")

    args = parser.parse_args()

    print("=" * 65)
    print("  🌌 AstroClassify: Space Object Classifier (NTO AI Olympiad) 🚀")
    print("=" * 65)

    if args.model in ("catboost", "all"):
        print("\n>>> Launching CatBoost Pipeline...")
        from catboost_solution import run_catboost_pipeline
        cb_score = run_catboost_pipeline(iterations=args.iterations, lr=args.lr)
        print(f"CatBoost Finished with Out-of-Fold Macro-F1: {cb_score:.4f}")

    if args.model in ("pytorch", "all"):
        print("\n>>> Launching PyTorch Deep MLP Pipeline...")
        from pytorch_solution import run_pipeline
        pt_score = run_pipeline(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
        print(f"PyTorch Finished with Mean CV Macro-F1: {pt_score:.4f}")


if __name__ == "__main__":
    main()
