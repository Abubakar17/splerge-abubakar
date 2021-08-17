import os
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from libs.transforms import get_transform
from libs.dataloader import SplitTableDataset
from libs.model import SplitModel
from libs.losses import split_loss

import time

from termcolor import cprint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_dir",
        help="Path to training data.",
        required=True,
    )
    parser.add_argument(
        "--val_dir",
        help="Path to validation data.",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_weight_path",
        dest="output_weight_path",
        help="Output folder path for model checkpoints and summary.",
        required=True,
    )
    parser.add_argument(
        "-e",
        "--num_epochs",
        type=int,
        dest="num_epochs",
        help="Number of epochs.",
        default=10,
    )
    parser.add_argument(
        "--log_every",
        type=int,
        dest="log_every",
        help="Print logs after every given steps",
        default=10,
    )
    parser.add_argument(
        "--val_every",
        type=int,
        dest="val_every",
        help="perform validation after given steps",
        default=1,
    )
    parser.add_argument(
        "--lr",
        "--learning_rate",
        type=float,
        dest="learning_rate",
        help="learning rate",
        default=0.00075,
    )

    configs = parser.parse_args()

    print(25 * "=", "Configuration", 25 * "=")
    print("Train Directory:\t", configs.train_dir)
    print("Validation Directory:\t", configs.val_dir)
    print("Output Weights Path:\t", configs.output_weight_path)
    print("Number of Epochs:\t", configs.num_epochs)
    print("Log after:\t", configs.log_every)
    print("Validate after:\t", configs.val_every)
    print("Batch Size:\t", 1)
    print("Learning Rate:\t", configs.learning_rate)
    print(65 * "=")

    batch_size = 1

    MODEL_STORE_PATH = configs.output_weight_path

    # train_images_path = configs.train_images_dir
    # train_labels_path = configs.train_labels_dir

    cprint("Loading dataset...", "blue", attrs=["bold"])
    train_dataset = SplitTableDataset(configs.train_dir)
    val_dataset   = SplitTableDataset(configs.val_dir)

    torch.manual_seed(1)
    # define training and validation data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(dataset=val_dataset  , batch_size=batch_size, shuffle=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    cprint("Creating split model...", "blue", attrs=["bold"])
    model = SplitModel().to(device)

    criterion = split_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.learning_rate)

    os.makedirs(MODEL_STORE_PATH)

    with open(os.path.join(MODEL_STORE_PATH, "config.json"), 'w') as fp:
        json.dump(configs.__dict__, fp, sort_keys=True, indent=4)
    best_val_loss = 10000.

    # create the summary writer
    writer = SummaryWriter(os.path.join(MODEL_STORE_PATH, "summary"))

    total_step = len(train_loader)
    time_stamp = time.time()

    for epoch in range(configs.num_epochs):

        for i, (images, targets, img_path, _, _) in enumerate(train_loader):
            images = images.to(device)

            model.train()
            # incrementing step

            targets[0] = targets[0].long().to(device)
            targets[1] = targets[1].long().to(device)

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()

            # Run the forward pass
            outputs = model(images.to(device))
            rpn_loss, cpn_loss = criterion(outputs, targets)

            loss = rpn_loss + cpn_loss
            loss.backward()
            optimizer.step()

            writer.add_scalar(
                "total loss train", loss.item(), (epoch * total_step + i)
            )
            writer.add_scalar(
                "rpn loss train", rpn_loss.item(), (epoch * total_step + i)
            )
            writer.add_scalar(
                "cpn loss train", cpn_loss.item(), (epoch * total_step + i)
            )

            if (i + 1) % configs.log_every == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, RPN Loss: {:.4f}, CPN Loss: {:.4f}, Time taken: {:.2f}s".format(
                        epoch + 1,
                        configs.num_epochs,
                        i + 1,
                        total_step,
                        loss.item(),
                        rpn_loss.item(),
                        cpn_loss.item(),
                        time.time() - time_stamp
                    )
                )
                time_stamp = time.time()

        if (epoch + 1) % configs.val_every == 0:
            print(65 * "=")
            print("Saving model weights at epoch", epoch + 1)
            model.eval()
            val_loss_list = []
            cpn_loss_list = []
            rpn_loss_list = []
            for val_batch in val_loader:
                with torch.no_grad():
                    val_images, val_targets, _, _, _ = val_batch

                    val_targets[0] = val_targets[0].long().to(device)
                    val_targets[1] = val_targets[1].long().to(device)

                    val_outputs = model(val_images.to(device))
                    val_loss, val_rpn_loss, val_cpn_loss = criterion(
                        val_outputs, val_targets
                    )

                    val_loss_list.append(val_loss.item())
                    rpn_loss_list.append(val_rpn_loss.item())
                    cpn_loss_list.append(val_cpn_loss.item())

            writer.add_scalar("total loss val", sum(val_loss_list) / len(val_loss_list), epoch)
            writer.add_scalar("rpn loss val", sum(rpn_loss_list) / len(val_loss_list), epoch)
            writer.add_scalar("cpn loss val", sum(cpn_loss_list) / len(val_loss_list), epoch)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "best_val_loss": best_val_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(MODEL_STORE_PATH, "last_model.pth"),
            )

            print("-"*25)
            print("Validation Loss :", sum(val_loss_list) / len(val_loss_list))
            print("-"*25)

            if best_val_loss > sum(val_loss_list) / len(val_loss_list):
                with open(os.path.join(MODEL_STORE_PATH, "best_epoch.txt"), 'w') as f:
                    f.write(str(epoch))
                best_val_loss = sum(val_loss_list) / len(val_loss_list)   
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    os.path.join(MODEL_STORE_PATH, "best_model.pth"),
                )   

        print(65 * "=")

        torch.cuda.empty_cache()