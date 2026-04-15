import argparse
import csv
import math


METHODS = ["baseline", "baseline_gnn", "improved_gnn"]


def parse_float(value):
    if value in ("", None):
        return None
    return float(value)


def mean(values):
    values = [v for v in values if v is not None and not math.isnan(v)]
    if not values:
        return None
    return sum(values) / len(values)


def count_better(rows, lhs_key, rhs_key):
    wins = 0
    total = 0
    for row in rows:
        lhs = parse_float(row.get(lhs_key))
        rhs = parse_float(row.get(rhs_key))
        if lhs is None or rhs is None:
            continue
        total += 1
        if lhs < rhs:
            wins += 1
    return wins, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    with open(args.summary, newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"instances={len(rows)}")
    print()

    for method in METHODS:
        runtime = mean([parse_float(r.get(f"{method}_runtime")) for r in rows])
        gap = mean([parse_float(r.get(f"{method}_mip_gap")) for r in rows])
        obj = mean([parse_float(r.get(f"{method}_obj_val")) for r in rows])
        print(f"{method}:")
        print(f"  mean_runtime={runtime}")
        print(f"  mean_mip_gap={gap}")
        print(f"  mean_obj_val={obj}")
        print()

    comparisons = [
        ("baseline_gnn", "baseline"),
        ("improved_gnn", "baseline"),
        ("improved_gnn", "baseline_gnn"),
    ]
    for lhs, rhs in comparisons:
        runtime_wins, runtime_total = count_better(
            rows, f"{lhs}_runtime", f"{rhs}_runtime"
        )
        gap_wins, gap_total = count_better(
            rows, f"{lhs}_mip_gap", f"{rhs}_mip_gap"
        )
        print(f"{lhs} vs {rhs}:")
        print(f"  runtime_win_rate={runtime_wins}/{runtime_total}")
        print(f"  gap_win_rate={gap_wins}/{gap_total}")
        print()


if __name__ == "__main__":
    main()
