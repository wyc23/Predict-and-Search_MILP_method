import argparse
import csv
import os
import random
import time

import torch
import torch_geometric

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="WA")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    return parser.parse_args()


def resolve_dataset_split_dirs(task_name):
    task_key = task_name.lower()
    base_dir = os.path.join("./dataset", task_key)
    train_dirs = (
        os.path.join(base_dir, "train", "BG"),
        os.path.join(base_dir, "train", "solution"),
    )
    valid_dirs = (
        os.path.join(base_dir, "valid", "BG"),
        os.path.join(base_dir, "valid", "solution"),
    )

    if not os.path.isdir(train_dirs[0]) or not os.path.isdir(train_dirs[1]):
        raise FileNotFoundError(
            f"Training split not found for task {task_name}: {train_dirs}"
        )
    if not os.path.isdir(valid_dirs[0]) or not os.path.isdir(valid_dirs[1]):
        raise FileNotFoundError(
            f"Validation split not found for task {task_name}: {valid_dirs}"
        )

    return train_dirs, valid_dirs


def build_sample_files(bg_dir, sol_dir):
    sample_names = sorted(name for name in os.listdir(bg_dir) if name.endswith(".bg"))
    return [
        (os.path.join(bg_dir, name), os.path.join(sol_dir, name.replace(".bg", ".sol")))
        for name in sample_names
        if os.path.exists(os.path.join(sol_dir, name.replace(".bg", ".sol")))
    ]


def energy_weight_norm(task):
    if task == "IP":
        return 1
    if task == "WA":
        return 100
    if task == "IS":
        return -100
    if task == "CA":
        return -1000
    raise ValueError(f"Unsupported task: {task}")


def _save_svg_loss_plot(history, plot_path):
    width = 800
    height = 500
    left = 60
    right = 20
    top = 30
    bottom = 50
    inner_width = width - left - right
    inner_height = height - top - bottom

    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    valid_losses = [row["valid_loss"] for row in history]
    all_losses = train_losses + valid_losses

    min_epoch = min(epochs)
    max_epoch = max(epochs) if max(epochs) > min_epoch else min_epoch + 1
    min_loss = min(all_losses)
    max_loss = max(all_losses)
    if max_loss == min_loss:
        max_loss = min_loss + 1.0

    def sx(epoch):
        return left + (epoch - min_epoch) / (max_epoch - min_epoch) * inner_width

    def sy(loss):
        return top + (max_loss - loss) / (max_loss - min_loss) * inner_height

    def polyline(values):
        return " ".join(f"{sx(ep):.2f},{sy(val):.2f}" for ep, val in zip(epochs, values))

    x_axis_y = top + inner_height
    y_axis_x = left
    train_points = polyline(train_losses)
    valid_points = polyline(valid_losses)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <line x1="{left}" y1="{x_axis_y}" x2="{width-right}" y2="{x_axis_y}" stroke="#222" stroke-width="2"/>
  <line x1="{y_axis_x}" y1="{top}" x2="{y_axis_x}" y2="{x_axis_y}" stroke="#222" stroke-width="2"/>
  <text x="{width/2:.0f}" y="20" text-anchor="middle" font-family="sans-serif" font-size="18">Training and Validation Loss</text>
  <text x="{width/2:.0f}" y="{height-10}" text-anchor="middle" font-family="sans-serif" font-size="14">Epoch</text>
  <text x="18" y="{height/2:.0f}" text-anchor="middle" font-family="sans-serif" font-size="14" transform="rotate(-90 18 {height/2:.0f})">Loss</text>
  <polyline fill="none" stroke="#1f77b4" stroke-width="2" points="{train_points}"/>
  <polyline fill="none" stroke="#d62728" stroke-width="2" points="{valid_points}"/>
  <rect x="{width-180}" y="35" width="12" height="12" fill="#1f77b4"/>
  <text x="{width-162}" y="45" font-family="sans-serif" font-size="12">train</text>
  <rect x="{width-100}" y="35" width="12" height="12" fill="#d62728"/>
  <text x="{width-82}" y="45" font-family="sans-serif" font-size="12">valid</text>
</svg>
"""
    with open(plot_path, "w", encoding="utf-8") as f:
        f.write(svg)


def maybe_save_loss_plot(history, plot_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _save_svg_loss_plot(history, plot_path.replace(".png", ".svg"))
        return False

    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    valid_losses = [row["valid_loss"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, valid_losses, label="valid")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return True


def run_epoch(predict, data_loader, device, optimizer=None, weight_norm=1):
    if optimizer:
        predict.train()
    else:
        predict.eval()

    mean_loss = 0
    n_samples_processed = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in data_loader:
            batch = batch.to(device)
            sol_ind = batch.nsols
            target_sols = []
            target_vals = []
            sol_end_ind = 0
            val_end_ind = 0

            for i in range(sol_ind.shape[0]):
                nvar = len(batch.varInds[i][0][0])
                sol_start_ind = sol_end_ind
                sol_end_ind = sol_ind[i] * nvar + sol_start_ind
                val_start_ind = val_end_ind
                val_end_ind = val_end_ind + sol_ind[i]
                sols = batch.solutions[sol_start_ind:sol_end_ind].reshape(-1, nvar)
                vals = batch.objVals[val_start_ind:val_end_ind]

                target_sols.append(sols)
                target_vals.append(vals)

            batch.constraint_features[torch.isinf(batch.constraint_features)] = 10
            bd = predict(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
            )
            bd = bd.sigmoid()

            loss = 0
            index_arrow = 0
            for ind, (sols, vals) in enumerate(zip(target_sols, target_vals)):
                weight = torch.softmax(-vals / weight_norm, dim=0)

                var_inds = batch.varInds[ind]
                varname_map = var_inds[0][0]
                b_vars = var_inds[1][0].long()
                sols = sols[:, varname_map][:, b_vars]

                n_var = batch.ntvars[ind]
                pre_sols = bd[index_arrow:index_arrow + n_var].squeeze()[b_vars]
                index_arrow = index_arrow + n_var

                pos_loss = -(pre_sols + 1e-8).log()[None, :] * (sols == 1).float()
                neg_loss = -(1 - pre_sols + 1e-8).log()[None, :] * (sols == 0).float()
                sample_loss = (pos_loss + neg_loss) * weight[:, None]
                loss += sample_loss.sum()

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            mean_loss += loss.item()
            n_samples_processed += batch.num_graphs

    mean_loss /= n_samples_processed
    return mean_loss


def main():
    args = parse_args()
    task_name = args.task.upper()
    train_task = f"{task_name}_train"

    os.makedirs("./train_logs", exist_ok=True)
    os.makedirs(f"./train_logs/{train_task}", exist_ok=True)
    os.makedirs("./pretrain", exist_ok=True)
    os.makedirs(f"./pretrain/{train_task}", exist_ok=True)

    model_save_path = f"./pretrain/{train_task}/"
    log_save_path = f"./train_logs/{train_task}/"
    log_file = open(f"{log_save_path}{train_task}_train.log", "wb")
    metrics_csv_path = os.path.join(log_save_path, "metrics.csv")
    plot_path = os.path.join(log_save_path, "loss_curve.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dirs, valid_dirs = resolve_dataset_split_dirs(task_name)
    train_files = build_sample_files(*train_dirs)
    valid_files = build_sample_files(*valid_dirs)
    print(
        f"Task={task_name} device={device} "
        f"train_samples={len(train_files)} valid_samples={len(valid_files)}"
    )

    random.seed(0)
    random.shuffle(train_files)

    if task_name == "IP":
        from GCN import GNNPolicy_position as GNNPolicy
        from GCN import GraphDataset_position as GraphDataset
    else:
        from GCN import GraphDataset, GNNPolicy

    train_data = GraphDataset(train_files)
    train_loader = torch_geometric.loader.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_data = GraphDataset(valid_files)
    valid_loader = torch_geometric.loader.DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    predict_model = GNNPolicy().to(device)
    optimizer = torch.optim.Adam(predict_model.parameters(), lr=args.lr)
    weight_norm = energy_weight_norm(task_name)
    best_val_loss = float("inf")
    history = []

    with open(metrics_csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=["epoch", "train_loss", "valid_loss", "epoch_time"]
        )
        writer.writeheader()

        for epoch in range(args.epochs):
            begin = time.time()
            train_loss = run_epoch(
                predict_model, train_loader, device, optimizer, weight_norm
            )
            valid_loss = run_epoch(predict_model, valid_loader, device, None, weight_norm)
            epoch_time = time.time() - begin

            print(f"Epoch {epoch} Train loss: {train_loss:0.3f}")
            print(f"Epoch {epoch} Valid loss: {valid_loss:0.3f}")

            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                torch.save(
                    predict_model.state_dict(),
                    os.path.join(model_save_path, "model_best.pth"),
                )

            torch.save(
                predict_model.state_dict(),
                os.path.join(model_save_path, "model_last.pth"),
            )

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "epoch_time": epoch_time,
            }
            history.append(row)
            writer.writerow(row)
            csv_file.flush()

            st = (
                f"@epoch{epoch}   Train loss:{train_loss}   "
                f"Valid loss:{valid_loss}    TIME:{epoch_time}\n"
            )
            log_file.write(st.encode())
            log_file.flush()

    maybe_save_loss_plot(history, plot_path)
    print("done")


if __name__ == "__main__":
    main()
