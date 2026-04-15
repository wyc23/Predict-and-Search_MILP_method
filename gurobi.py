import argparse
import os
import pickle
import traceback
from multiprocessing import Process, Queue

import gurobipy as gp
import numpy as np

from helper import get_a_new2


DEFAULT_TASKS = ["IP", "WA", "IS", "CA", "NNV"]
DEFAULT_SPLITS = ["train", "valid", "test"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate MILP dataset artifacts with Gurobi."
    )
    parser.add_argument("--dataDir", type=str, default="./")
    parser.add_argument("--tasks", nargs="+", default=["CA"])
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--nWorkers", type=int, default=16)
    parser.add_argument("--maxTime", type=int, default=3600)
    parser.add_argument("--maxStoredSol", type=int, default=20)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate files even if the target .bg already exists.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the current split early when a worker reports an error.",
    )
    return parser.parse_args()


def solve_grb(filepath, log_dir, settings):
    gp.setParam("LogToConsole", 0)
    m = gp.read(filepath)

    m.Params.PoolSolutions = settings["maxsol"]
    m.Params.PoolSearchMode = settings["mode"]
    m.Params.TimeLimit = settings["maxtime"]
    m.Params.Threads = settings["threads"]

    log_path = os.path.join(log_dir, os.path.basename(filepath) + ".log")
    with open(log_path, "w"):
        pass
    m.Params.LogFile = log_path

    m.optimize()

    sols = []
    objs = []
    solc = m.getAttr("SolCount")
    mvars = m.getVars()
    ori_var_names = [var.varName for var in mvars]

    for sn in range(solc):
        m.Params.SolutionNumber = sn
        sols.append(np.array(m.Xn))
        objs.append(m.PoolObjVal)

    return {
        "var_names": ori_var_names,
        "sols": np.array(sols, dtype=np.float32),
        "objs": np.array(objs, dtype=np.float32),
    }


def collect(ins_dir, task, split, q, result_q, output_dirs, settings):
    while True:
        filename = q.get()
        if filename is None:
            break

        filepath = os.path.join(ins_dir, filename)
        try:
            sol_data = solve_grb(filepath, output_dirs["logs"], settings)
            A2, v_map2, v_nodes2, c_nodes2, b_vars2 = get_a_new2(filepath)
            bg_data = [A2, v_map2, v_nodes2, c_nodes2, b_vars2]

            with open(os.path.join(output_dirs["solution"], filename + ".sol"), "wb") as f:
                pickle.dump(sol_data, f)
            with open(os.path.join(output_dirs["bg"], filename + ".bg"), "wb") as f:
                pickle.dump(bg_data, f)

            result_q.put(
                {
                    "status": "ok",
                    "task": task,
                    "split": split,
                    "filename": filename,
                }
            )
        except Exception as exc:
            result_q.put(
                {
                    "status": "error",
                    "task": task,
                    "split": split,
                    "filename": filename,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )


def resolve_task_instance_dir(data_dir, task_name):
    candidates = [
        os.path.join(data_dir, "instance", task_name),
        os.path.join(data_dir, "instance", task_name.lower()),
        os.path.join(data_dir, "instance", task_name.upper()),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return None


def ensure_split_dirs(base_dir, split_name):
    split_dir = os.path.join(base_dir, split_name)
    dirs = {
        "split": split_dir,
        "solution": os.path.join(split_dir, "solution"),
        "logs": os.path.join(split_dir, "logs"),
        "bg": os.path.join(split_dir, "BG"),
        "nbp": os.path.join(split_dir, "NBP"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def list_instance_files(ins_dir):
    return sorted(
        filename
        for filename in os.listdir(ins_dir)
        if os.path.isfile(os.path.join(ins_dir, filename))
    )


def select_pending_files(filenames, output_dirs, overwrite):
    if overwrite:
        return filenames
    return [
        filename
        for filename in filenames
        if not os.path.exists(os.path.join(output_dirs["bg"], filename + ".bg"))
    ]


def normalize_requested_tasks(requested_tasks):
    if requested_tasks == ["ALL"]:
        return DEFAULT_TASKS
    return [task.upper() for task in requested_tasks]


def run_split(task, split, ins_dir, output_dirs, settings, n_workers, overwrite, fail_fast):
    filenames = list_instance_files(ins_dir)
    pending_files = select_pending_files(filenames, output_dirs, overwrite)

    print(
        f"[{task}/{split}] instances={len(filenames)} "
        f"pending={len(pending_files)} overwrite={overwrite}"
    )

    if not pending_files:
        print(f"[{task}/{split}] nothing to do")
        return []

    worker_count = max(1, min(n_workers, len(pending_files)))
    q = Queue()
    result_q = Queue()

    for filename in pending_files:
        q.put(filename)
    for _ in range(worker_count):
        q.put(None)

    processes = []
    for _ in range(worker_count):
        p = Process(
            target=collect,
            args=(ins_dir, task, split, q, result_q, output_dirs, settings),
        )
        p.start()
        processes.append(p)

    errors = []
    completed = 0
    for _ in range(len(pending_files)):
        result = result_q.get()
        completed += 1
        if result["status"] == "error":
            errors.append(result)
            print(
                f"[{task}/{split}] ERROR {result['filename']}: {result['error']}"
            )
            if fail_fast:
                break
        elif completed % 10 == 0 or completed == len(pending_files):
            print(f"[{task}/{split}] completed {completed}/{len(pending_files)}")

    for p in processes:
        p.join()

    print(
        f"[{task}/{split}] finished with {len(errors)} error(s), "
        f"{len(pending_files) - len(errors)} success(es)"
    )
    return errors


def main():
    args = parse_args()
    tasks = normalize_requested_tasks(args.tasks)
    splits = [split.lower() for split in args.splits]

    settings = {
        "maxtime": args.maxTime,
        "mode": 2,
        "maxsol": args.maxStoredSol,
        "threads": args.threads,
    }

    all_errors = []
    for task in tasks:
        task_instance_dir = resolve_task_instance_dir(args.dataDir, task)
        if task_instance_dir is None:
            print(f"[{task}] skipped: instance directory not found")
            continue

        dataset_task_dir = os.path.join("./dataset", task.lower())
        os.makedirs(dataset_task_dir, exist_ok=True)

        for split in splits:
            ins_dir = os.path.join(task_instance_dir, split)
            if not os.path.isdir(ins_dir):
                print(f"[{task}/{split}] skipped: split directory not found")
                continue

            output_dirs = ensure_split_dirs(dataset_task_dir, split)
            errors = run_split(
                task=task,
                split=split,
                ins_dir=ins_dir,
                output_dirs=output_dirs,
                settings=settings,
                n_workers=args.nWorkers,
                overwrite=args.overwrite,
                fail_fast=args.fail_fast,
            )
            all_errors.extend(errors)

    if all_errors:
        print("\nSummary of failed instances:")
        for error in all_errors:
            print(
                f"- {error['task']}/{error['split']}/{error['filename']}: {error['error']}"
            )
    else:
        print("\nAll requested dataset generation jobs finished successfully.")


if __name__ == "__main__":
    main()
