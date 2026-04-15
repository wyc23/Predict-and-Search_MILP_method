import argparse
import csv
import os
import random
import time

import torch
import torch_geometric

from GCN import GraphDataset, ImprovedGNNPolicy
from trainPredictModel import (
    build_sample_files,
    energy_weight_norm,
    maybe_save_loss_plot,
    resolve_dataset_split_dirs,
    run_epoch,
)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="CA")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    return parser.parse_args()


def main():
    args = parse_args()
    task_name = args.task.upper()
    train_task = f"{task_name}_improved_train"

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

    predict_model = ImprovedGNNPolicy().to(device)
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
