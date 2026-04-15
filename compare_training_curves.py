import argparse
import csv
import os


def load_metrics(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--improved", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline_rows = load_metrics(args.baseline)
    improved_rows = load_metrics(args.improved)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5))
    plt.plot(
        [int(r["epoch"]) for r in baseline_rows],
        [float(r["train_loss"]) for r in baseline_rows],
        label="baseline train",
        color="#1f77b4",
    )
    plt.plot(
        [int(r["epoch"]) for r in baseline_rows],
        [float(r["valid_loss"]) for r in baseline_rows],
        label="baseline valid",
        color="#1f77b4",
        linestyle="--",
    )
    plt.plot(
        [int(r["epoch"]) for r in improved_rows],
        [float(r["train_loss"]) for r in improved_rows],
        label="improved train",
        color="#d62728",
    )
    plt.plot(
        [int(r["epoch"]) for r in improved_rows],
        [float(r["valid_loss"]) for r in improved_rows],
        label="improved valid",
        color="#d62728",
        linestyle="--",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Baseline vs Improved Training Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
