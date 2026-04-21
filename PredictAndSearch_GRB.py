import argparse
import csv
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import gurobipy
import torch
from gurobipy import GRB

from helper import get_a_new2

INFER_DEVICE = torch.device("cpu")
random.seed(0)
torch.manual_seed(0)

METHODS = ["baseline", "baseline_gnn", "improved_gnn"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="CA")
    parser.add_argument("--test-num", type=int, default=100)
    parser.add_argument(
        "--mode",
        choices=["baseline", "baseline_gnn", "improved_gnn", "learned", "all"],
        default="all",
    )
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--solver-threads", type=int, default=1)
    parser.add_argument("--time-limit", type=int, default=1000)
    parser.add_argument("--summary-name", type=str, default="summary.csv")
    parser.add_argument("--baseline-model-path", type=str, default="")
    parser.add_argument("--improved-model-path", type=str, default="")
    return parser.parse_args()


def resolve_instance_dir(task):
    candidates = [
        f"./instance/{task}/test",
        f"./instance/{task.lower()}/test",
        f"./instance/{task}",
        f"./instance/{task.lower()}",
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"No instance directory found for task {task}.")


def test_hyperparam(task):
    if task == "IP":
        return 400, 5, 1
    if task == "IS":
        return 300, 300, 15
    if task == "WA":
        return 0, 600, 5
    if task == "CA":
        return 400, 0, 10
    raise ValueError(f"Unsupported task: {task}")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def extract_gurobi_stats(model):
    status = int(model.Status)
    has_incumbent = model.SolCount > 0
    return {
        "status": status,
        "model_sense": "min" if model.ModelSense == 1 else "max",
        "runtime": float(model.Runtime),
        "obj_val": float(model.ObjVal) if has_incumbent else "",
        "obj_bound": float(model.ObjBound),
        "mip_gap": float(model.MIPGap) if has_incumbent else "",
        "sol_count": int(model.SolCount),
        "node_count": float(model.NodeCount),
    }


def get_model_paths(task_name, baseline_model_path, improved_model_path):
    baseline_path = baseline_model_path or f"./models/{task_name}.pth"
    improved_path = improved_model_path or f"./pretrain/{task_name}_improved_train/model_best.pth"
    return baseline_path, improved_path


def load_policy(task_name, method_name, baseline_model_path, improved_model_path):
    if task_name == "IP":
        raise NotImplementedError("IP position-augmented improved evaluation is not wired here.")

    if method_name == "baseline_gnn":
        from GCN import GNNPolicy

        model_cls = GNNPolicy
        model_path = get_model_paths(task_name, baseline_model_path, improved_model_path)[0]
    elif method_name == "improved_gnn":
        from GCN import ImprovedGNNPolicy

        model_cls = ImprovedGNNPolicy
        model_path = get_model_paths(task_name, baseline_model_path, improved_model_path)[1]
    else:
        raise ValueError(f"Unsupported learned method: {method_name}")

    policy = model_cls().to(INFER_DEVICE)
    state = torch.load(model_path, map_location=INFER_DEVICE)
    policy.load_state_dict(state)
    policy.eval()
    return policy


def build_scores(policy, task_name, ins_name_to_read):
    A, v_map, v_nodes, c_nodes, b_vars = get_a_new2(ins_name_to_read)
    constraint_features = c_nodes.cpu()
    constraint_features[torch.isnan(constraint_features)] = 1
    variable_features = v_nodes.cpu()
    edge_indices = A._indices().cpu()
    edge_features = torch.ones(A._values().unsqueeze(1).shape)

    with torch.no_grad():
        bd = policy(
            constraint_features.to(INFER_DEVICE),
            edge_indices.to(INFER_DEVICE),
            edge_features.to(INFER_DEVICE),
            variable_features.to(INFER_DEVICE),
        ).sigmoid().cpu().squeeze()

    all_varname = list(v_map.keys())
    binary_name = [all_varname[i] for i in b_vars]
    scores = []
    for i in range(len(v_map)):
        var_type = "BINARY" if all_varname[i] in binary_name else "C"
        scores.append([i, all_varname[i], bd[i].item(), -1, var_type])

    scores.sort(key=lambda x: x[2], reverse=True)
    return [x for x in scores if x[4] == "BINARY"]


def apply_predict_search_constraints(model, scores, task_name):
    k_0, k_1, delta = test_hyperparam(task_name)

    count1 = 0
    for i in range(len(scores)):
        if count1 < k_1:
            scores[i][3] = 1
            count1 += 1
    scores.sort(key=lambda x: x[2], reverse=False)
    count0 = 0
    for i in range(len(scores)):
        if count0 < k_0:
            scores[i][3] = 0
            count0 += 1

    instance_variables = model.getVars()
    instance_variables.sort(key=lambda v: v.VarName)
    variables_map = {v.VarName: v for v in instance_variables}

    alphas = []
    for i in range(len(scores)):
        tar_var = variables_map[scores[i][1]]
        x_star = scores[i][3]
        if x_star < 0:
            continue
        tmp_var = model.addVar(name=f"alp_{tar_var}", vtype=GRB.CONTINUOUS)
        alphas.append(tmp_var)
        model.addConstr(tmp_var >= tar_var - x_star, name=f"alpha_up_{i}")
        model.addConstr(tmp_var >= x_star - tar_var, name=f"alpha_down_{i}")

    model.addConstr(gurobipy.quicksum(alphas) <= delta, name="sum_alpha")
    return {"fixed_to_zero": k_0, "fixed_to_one": k_1, "delta": delta}


def configure_solver(model, log_path, time_limit, solver_threads):
    model.Params.TimeLimit = time_limit
    model.Params.Threads = solver_threads
    model.Params.MIPFocus = 1
    model.Params.LogToConsole = 0
    model.Params.LogFile = log_path


def run_baseline(ins_name_to_read, log_path, time_limit, solver_threads):
    model = gurobipy.read(ins_name_to_read)
    configure_solver(model, log_path, time_limit, solver_threads)
    model.optimize()
    return extract_gurobi_stats(model)


def run_learned_method(
    method_name,
    task_name,
    ins_name_to_read,
    log_path,
    time_limit,
    solver_threads,
    baseline_model_path,
    improved_model_path,
):
    policy = load_policy(task_name, method_name, baseline_model_path, improved_model_path)
    scores = build_scores(policy, task_name, ins_name_to_read)

    model = gurobipy.read(ins_name_to_read)
    configure_solver(model, log_path, time_limit, solver_threads)
    meta = apply_predict_search_constraints(model, scores, task_name)
    model.optimize()

    result = extract_gurobi_stats(model)
    result.update(meta)
    return result


def resolve_methods(mode):
    if mode == "all":
        return METHODS
    if mode == "learned":
        return ["baseline_gnn", "improved_gnn"]
    return [mode]


def evaluate_instance(
    task_name,
    instance_dir,
    instance_name,
    methods,
    log_dirs,
    time_limit,
    solver_threads,
    baseline_model_path,
    improved_model_path,
):
    ins_name_to_read = os.path.join(instance_dir, instance_name)
    row = {"instance": instance_name, "instance_path": ins_name_to_read}

    for method in methods:
        log_path = os.path.join(log_dirs[method], f"{instance_name}.log")
        if method == "baseline":
            result = run_baseline(ins_name_to_read, log_path, time_limit, solver_threads)
        else:
            result = run_learned_method(
                method,
                task_name,
                ins_name_to_read,
                log_path,
                time_limit,
                solver_threads,
                baseline_model_path,
                improved_model_path,
            )
        row["objective_sense"] = result["model_sense"]
        for key, value in result.items():
            row[f"{method}_{key}"] = value

    return row


def main():
    args = parse_args()
    task_name = args.task.upper()
    instance_dir = resolve_instance_dir(task_name)
    sample_names = sorted(os.listdir(instance_dir))[: args.test_num]
    methods = resolve_methods(args.mode)

    ensure_dir("./logs")
    ensure_dir(f"./logs/{task_name}")

    log_dirs = {
        "baseline": f"./logs/{task_name}/{task_name}_GRB_Baseline",
        "baseline_gnn": f"./logs/{task_name}/{task_name}_GRB_BaselineGNN",
        "improved_gnn": f"./logs/{task_name}/{task_name}_GRB_ImprovedGNN",
    }
    for path in log_dirs.values():
        ensure_dir(path)

    print(
        f"task={task_name} methods={methods} instances={len(sample_names)} "
        f"n_workers={args.n_workers} solver_threads={args.solver_threads} "
        f"infer_device={INFER_DEVICE}"
    )

    rows = []
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {
            executor.submit(
                evaluate_instance,
                task_name,
                instance_dir,
                instance_name,
                methods,
                log_dirs,
                args.time_limit,
                args.solver_threads,
                args.baseline_model_path,
                args.improved_model_path,
            ): instance_name
            for instance_name in sample_names
        }
        for idx, future in enumerate(as_completed(futures), start=1):
            instance_name = futures[future]
            row = future.result()
            rows.append(row)
            print(f"completed {idx}/{len(sample_names)}: {instance_name}")

    rows.sort(key=lambda x: x["instance"])
    summary_path = os.path.join("./logs", task_name, args.summary_name)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"summary saved to {summary_path}")


if __name__ == "__main__":
    main()
