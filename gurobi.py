import os
import pickle
from multiprocessing import Process, Queue

import argparse
import gurobipy as gp
import numpy as np

from helper import get_a_new2


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

    sols = np.array(sols, dtype=np.float32)
    objs = np.array(objs, dtype=np.float32)

    sol_data = {
        "var_names": ori_var_names,
        "sols": sols,
        "objs": objs,
    }

    return sol_data


def collect(ins_dir, q, sol_dir, log_dir, bg_dir, settings):
    while True:
        filename = q.get()
        if filename is None:
            break

        filepath = os.path.join(ins_dir, filename)
        sol_data = solve_grb(filepath, log_dir, settings)

        # Generate the bipartite graph features for the same instance.
        A2, v_map2, v_nodes2, c_nodes2, b_vars2 = get_a_new2(filepath)
        bg_data = [A2, v_map2, v_nodes2, c_nodes2, b_vars2]

        pickle.dump(sol_data, open(os.path.join(sol_dir, filename + ".sol"), "wb"))
        pickle.dump(bg_data, open(os.path.join(bg_dir, filename + ".bg"), "wb"))


def resolve_task_instance_dir(data_dir, task_name):
    candidates = [
        os.path.join(data_dir, "instance", task_name),
        os.path.join(data_dir, "instance", task_name.lower()),
        os.path.join(data_dir, "instance", task_name.upper()),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"Instance directory not found for task {task_name}.")


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


if __name__ == "__main__":
    tasks = ["IP", "WA", "IS", "CA", "NNV"]
    splits = ["train", "valid", "test"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataDir", type=str, default="./")
    parser.add_argument("--nWorkers", type=int, default=100)
    parser.add_argument("--maxTime", type=int, default=3600)
    parser.add_argument("--maxStoredSol", type=int, default=500)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    settings = {
        "maxtime": args.maxTime,
        "mode": 2,
        "maxsol": args.maxStoredSol,
        "threads": args.threads,
    }

    for task in tasks:
        task_instance_dir = resolve_task_instance_dir(args.dataDir, task)
        dataset_task_dir = os.path.join("./dataset", task.lower())
        os.makedirs(dataset_task_dir, exist_ok=True)

        for split in splits:
            ins_dir = os.path.join(task_instance_dir, split)
            if not os.path.isdir(ins_dir):
                continue

            output_dirs = ensure_split_dirs(dataset_task_dir, split)
            filenames = sorted(os.listdir(ins_dir))
            filenames = [
                filename
                for filename in filenames
                if os.path.isfile(os.path.join(ins_dir, filename))
            ]

            q = Queue()
            for filename in filenames:
                if not os.path.exists(
                    os.path.join(output_dirs["bg"], filename + ".bg")
                ):
                    q.put(filename)

            for _ in range(args.nWorkers):
                q.put(None)

            ps = []
            for _ in range(args.nWorkers):
                p = Process(
                    target=collect,
                    args=(
                        ins_dir,
                        q,
                        output_dirs["solution"],
                        output_dirs["logs"],
                        output_dirs["bg"],
                        settings,
                    ),
                )
                p.start()
                ps.append(p)

            for p in ps:
                p.join()

            print(f"done: {task} {split}")
