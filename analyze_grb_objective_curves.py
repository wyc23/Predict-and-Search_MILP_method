import argparse
import csv
import math
import os
from bisect import bisect_right

METHOD_CONFIG = {
    "baseline": {
        "label": "Gurobi",
        "log_dir_suffix": "GRB_Baseline",
    },
    "baseline_gnn": {
        "label": "Baseline GNN + P&S",
        "log_dir_suffix": "GRB_BaselineGNN",
    },
    "improved_gnn": {
        "label": "Improved GNN + P&S",
        "log_dir_suffix": "GRB_ImprovedGNN",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="CA")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--time-limit", type=int, default=1000)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="")
    return parser.parse_args()


def parse_float(value):
    if value in ("", None):
        return None
    return float(value)


def mean_ignore_nan(values):
    valid = [v for v in values if v is not None and not math.isnan(v)]
    if not valid:
        return None
    return sum(valid) / len(valid)


def parse_time_token(token):
    token = token.strip()
    if token.endswith("s"):
        return float(token[:-1])
    if token.endswith("m"):
        return float(token[:-1]) * 60.0
    if token.endswith("h"):
        return float(token[:-1]) * 3600.0
    return float(token)


def parse_gurobi_log(log_path):
    sense = None
    trajectory = []

    with open(log_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if "(Max)" in line or "Maximizing a MIP problem" in line:
                sense = "max"
                continue
            if "(Min)" in line or "Minimizing a MIP problem" in line:
                sense = "min"
                continue

            tokens = line.split()
            if not tokens:
                continue

            # Gurobi node log rows end with:
            #   Incumbent  BestBd  Gap  It/Node  Time
            # The left side of the row is variable-width, so parse from the end.
            if len(tokens) < 5:
                continue
            if not tokens[-1].endswith(("s", "m", "h")):
                continue
            if not tokens[-3].endswith("%"):
                continue

            try:
                time_value = parse_time_token(tokens[-1])
                best_objective = float(tokens[-5])
            except ValueError:
                continue

            trajectory.append((time_value, best_objective))

    deduped = []
    for time_value, obj_value in trajectory:
        if deduped and time_value == deduped[-1][0]:
            deduped[-1] = (time_value, obj_value)
        else:
            deduped.append((time_value, obj_value))

    return sense, deduped


def build_step_curve(trajectory, horizon, step):
    grid = list(range(0, horizon + 1, step))
    if not trajectory:
        return grid, [math.nan for _ in grid]

    event_times = [item[0] for item in trajectory]
    event_values = [item[1] for item in trajectory]
    curve = []
    for t in grid:
        idx = bisect_right(event_times, t) - 1
        if idx < 0:
            curve.append(math.nan)
        else:
            curve.append(event_values[idx])
    return grid, curve


def get_available_methods(rows):
    methods = []
    for method in METHOD_CONFIG:
        key = f"{method}_obj_val"
        if any(key in row for row in rows):
            methods.append(method)
    return methods


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def maybe_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ModuleNotFoundError:
        return None


def main():
    args = parse_args()
    task_name = args.task.upper()
    output_dir = args.output_dir or os.path.join("./logs", task_name, "analysis")
    ensure_dir(output_dir)

    with open(args.summary, newline="") as f:
        rows = list(csv.DictReader(f))

    methods = get_available_methods(rows)
    if not methods:
        raise ValueError(f"No recognized methods found in summary: {args.summary}")

    objective_sense = rows[0].get("objective_sense", "")
    if objective_sense not in {"min", "max"}:
        objective_sense = None

    method_curves = {method: [] for method in methods}
    method_final_objs = {method: [] for method in methods}
    for row in rows:
        instance_name = row["instance"]
        for method in methods:
            log_dir = os.path.join(
                "./logs", task_name, f"{task_name}_{METHOD_CONFIG[method]['log_dir_suffix']}"
            )
            log_path = os.path.join(log_dir, f"{instance_name}.log")
            if not os.path.exists(log_path):
                continue

            log_sense, trajectory = parse_gurobi_log(log_path)
            if objective_sense is None and log_sense in {"min", "max"}:
                objective_sense = log_sense

            final_obj = parse_float(row.get(f"{method}_obj_val"))
            final_runtime = parse_float(row.get(f"{method}_runtime"))
            if final_obj is not None and final_runtime is not None:
                if not trajectory or final_runtime > trajectory[-1][0]:
                    trajectory.append((final_runtime, final_obj))

            grid, curve = build_step_curve(trajectory, args.time_limit, args.step)
            method_curves[method].append(curve)
            method_final_objs[method].append(final_obj)

    if objective_sense is None:
        objective_sense = "max"

    mean_curve_rows = []
    for idx, time_value in enumerate(grid):
        row = {"time": time_value}
        for method in methods:
            values = [curve[idx] for curve in method_curves[method]]
            coverage = sum(0 if math.isnan(v) else 1 for v in values)
            row[f"{method}_mean_best_objective"] = mean_ignore_nan(values)
            row[f"{method}_coverage"] = coverage
        mean_curve_rows.append(row)

    curve_csv_path = os.path.join(output_dir, f"{task_name.lower()}_mean_objective_curve.csv")
    with open(curve_csv_path, "w", newline="") as f:
        fieldnames = list(mean_curve_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mean_curve_rows)

    plot_path = os.path.join(output_dir, f"{task_name.lower()}_mean_objective_curve.png")
    plt = maybe_import_matplotlib()
    if plt is not None:
        plt.figure(figsize=(10, 6))
        for method in methods:
            y_values = [row[f"{method}_mean_best_objective"] for row in mean_curve_rows]
            plt.plot(grid, y_values, label=METHOD_CONFIG[method]["label"], linewidth=2)
        plt.xlabel("Time (s)")
        plt.ylabel("Average Best Objective")
        plt.title(f"{task_name} Test Set Mean Best Objective Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
    else:
        plot_path = ""

    final_rows = []
    for method in methods:
        final_mean = mean_ignore_nan(method_final_objs[method])
        final_rows.append(
            {
                "method": method,
                "label": METHOD_CONFIG[method]["label"],
                "objective_sense": objective_sense,
                "instances_with_incumbent": sum(
                    1 for value in method_final_objs[method] if value is not None
                ),
                "mean_final_objective": final_mean,
            }
        )

    final_csv_path = os.path.join(
        output_dir, f"{task_name.lower()}_final_objective_summary.csv"
    )
    with open(final_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(final_rows[0].keys()))
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"task={task_name}")
    print(f"objective_sense={objective_sense}")
    print(f"summary={args.summary}")
    print(f"curve_csv={curve_csv_path}")
    print(f"curve_png={plot_path or 'skipped (matplotlib not installed)'}")
    print(f"final_csv={final_csv_path}")
    print()

    baseline_mean = None
    for row in final_rows:
        print(
            f"{row['method']}: mean_final_objective={row['mean_final_objective']} "
            f"instances_with_incumbent={row['instances_with_incumbent']}"
        )
        if row["method"] == "baseline":
            baseline_mean = row["mean_final_objective"]

    if baseline_mean is not None:
        print()
        for row in final_rows:
            if row["method"] == "baseline" or row["mean_final_objective"] is None:
                continue
            delta = row["mean_final_objective"] - baseline_mean
            if objective_sense == "min":
                better = delta < 0
            else:
                better = delta > 0
            print(
                f"{row['method']} vs baseline: delta_mean_final_objective={delta} "
                f"better_than_baseline={better}"
            )


if __name__ == "__main__":
    main()
