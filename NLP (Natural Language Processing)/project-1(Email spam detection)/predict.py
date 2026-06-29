"""
Entry point: classify one or more email messages from the command line.

Usage
-----
    python predict.py "Congratulations! You won a free iPhone click here now"
    python predict.py --file emails.txt          # one message per line
    python predict.py --interactive              # REPL mode
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.predictor import SpamPredictor
from src.utils import get_logger

logger = get_logger("predict_entry")


def parse_args():
    p = argparse.ArgumentParser(description="Predict spam / ham")
    p.add_argument("message", nargs="?", help="Single message to classify")
    p.add_argument("--file",        "-f", help="Text file with one message per line")
    p.add_argument("--interactive", "-i", action="store_true",
                   help="Enter interactive REPL mode")
    p.add_argument("--threshold",   "-t", type=float, default=0.5,
                   help="Classification threshold (default 0.5)")
    return p.parse_args()


def print_result(text: str, result: dict) -> None:
    label = result["label"].upper()
    conf  = result["confidence"]
    prob  = result["raw_prob"]
    bar   = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
    print(f"\n  Text       : {text[:80]}")
    print(f"  Prediction : {'🔴 SPAM' if result['is_spam'] else '🟢 HAM'}")
    print(f"  Confidence : {conf:.2%}  |  P(spam)={prob:.4f}  [{bar}]")
    print()


def main():
    args    = parse_args()
    pred    = SpamPredictor(threshold=args.threshold)

    if args.interactive:
        print("\n── BiLSTM Spam Detector (type 'quit' to exit) ──\n")
        while True:
            try:
                msg = input("Enter message: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if msg.lower() in {"quit", "exit", "q"}:
                break
            if msg:
                print_result(msg, pred.predict(msg))

    elif args.file:
        with open(args.file) as f:
            lines = [l.strip() for l in f if l.strip()]
        results = pred.predict_batch(lines)
        for text, result in zip(lines, results):
            print_result(text, result)

    elif args.message:
        print_result(args.message, pred.predict(args.message))

    else:
        print("Please supply a message, --file, or --interactive flag.")
        print("Run  python predict.py --help  for usage.")
        sys.exit(1)


if __name__ == "__main__":
    main()
