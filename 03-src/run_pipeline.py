import argparse
import subprocess
import sys
from pathlib import Path

from model import model_output_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "03-src"
DEFAULT_MODELS = ["mobilenetv3", "efficientnet_b0", "densenet121"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the DS107 real-vs-AI pipeline")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing splits")
    parser.add_argument("--reset-cleaned", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Models to train/evaluate")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "04-reports" / "runs"))
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=PROJECT_ROOT)


def main():
    args = parse_args()

    if args.validate_only:
        run([sys.executable, str(SRC_DIR / "prepare_data.py"), "--validate-only"])
        return

    if not args.skip_prepare:
        cmd = [sys.executable, str(SRC_DIR / "prepare_data.py")]
        if args.reset_cleaned:
            cmd.append("--reset-cleaned")
        run(cmd)
    else:
        run([sys.executable, str(SRC_DIR / "prepare_data.py"), "--validate-only"])

    if not args.skip_train:
        for model_name in args.models:
            output_dir = Path(args.output_root) / model_output_name(model_name)
            cmd = [
                sys.executable,
                str(SRC_DIR / "train.py"),
                "--model-name",
                model_name,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--early-stopping-patience",
                str(args.early_stopping_patience),
                "--early-stopping-min-delta",
                str(args.early_stopping_min_delta),
                "--output-dir",
                str(output_dir),
            ]
            if args.device:
                cmd.extend(["--device", args.device])
            run(cmd)

    if not args.skip_eval:
        for model_name in args.models:
            output_name = model_output_name(model_name)
            checkpoint = Path(args.output_root) / output_name / "best.pt"
            cmd = [
                sys.executable,
                str(SRC_DIR / "predict.py"),
                "--model-name",
                model_name,
                "--model-path",
                str(checkpoint),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--out-csv",
                str(PROJECT_ROOT / "04-reports" / f"predictions_{output_name}.csv"),
                "--metrics-csv",
                str(PROJECT_ROOT / "04-reports" / f"metrics_{output_name}.csv"),
                "--save-misclassified-dir",
                str(PROJECT_ROOT / "04-reports" / f"misclassified_{output_name}"),
            ]
            if args.device:
                cmd.extend(["--device", args.device])
            run(cmd)


if __name__ == "__main__":
    main()
