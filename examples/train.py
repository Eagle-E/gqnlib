
"""Training method."""

import argparse
import json
import os
import pathlib
import random

import torch

import gqnlib
from experiment import Trainer


def main():

    # -------------------------------------------------------------------------
    # 1. Settings
    # -------------------------------------------------------------------------

    # Command line args
    args = init_args()

    # Configs
    config_path = pathlib.Path(
        os.getenv("CONFIG_PATH", "./examples/config.json"))
    with config_path.open() as f:
        config = json.load(f)

    # Path
    logdir = str(pathlib.Path(os.getenv("LOGDIR", "./logs/"),
                              os.getenv("EXPERIMENT_NAME", "tmp")))
    data_dir = pathlib.Path(
        os.getenv("DATASET_DIR", "./data/"),
        os.getenv("DATASET_NAME", "shepard_metzler_5_parts_torch"))
    train_dir = str(data_dir / "train")
    tset_dir = str(data_dir / "test")

    # Cuda setting
    use_cuda = torch.cuda.is_available() and args.cuda != "null"
    gpus = args.cuda if use_cuda else None

    # Random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # -------------------------------------------------------------------------
    # 2. Training
    # -------------------------------------------------------------------------

    # GQN model
    model = gqnlib.GenerativeQueryNetwork(**config["model_params"])

    # Trainer
    params = {
        "logdir": logdir,
        "gpus": gpus,
        "train_dir": train_dir,
        "test_dir": tset_dir,
    }
    params.update(config)
    params.update(vars(args))

    trainer = Trainer(model, params)
    trainer.run()


def init_args():
    parser = argparse.ArgumentParser(description="GQN training")
    parser.add_argument("--cuda", type=str, default="0",
                        help="Number of CUDA device with comma separation, "
                             "ex. '0,1'. 'null' means cpu device.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed.")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size.")
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of training steps.")
    parser.add_argument("--log-save-interval", type=int, default=5,
                        help="Interval epochs for saving logs.")

    return parser.parse_args()


if __name__ == "__main__":
    main()
