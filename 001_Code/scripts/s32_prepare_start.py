import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config
from train.sft import resolve_sft_start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_path, _, _ = resolve_sft_start(cfg)
    print("prepared start:", model_path)


if __name__ == "__main__":
    main()
