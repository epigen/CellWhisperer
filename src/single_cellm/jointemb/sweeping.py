"""
This is an untested script to perform wandb sweeps. Good luck! ;)
"""
import yaml
import sys
from single_cellm.jointemb.training import cli_main as training_main
from single_cellm.config import get_path, config
from pathlib import Path
from argparse import ArgumentParser, Namespace
import subprocess
import wandb
from functools import partial
import logging


def init_wandb(args):
    wandb_folder = get_path(["paths", "wandb_logs"])

    kwargs = {
        "entity": "single-cellm",
        "project": "JointEmbed_Training",
        "reinit": True,
        "job_type": "sweep-training",
        "mode": args.wandb_mode,
        "dir": str(wandb_folder),
    }
    run = wandb.init(**kwargs)  # this is apparently called from within the training
    return run


def wandb_config_to_args(wandb_config):
    """
    Convert wandb_config (dict-like) to args object that can be passed to LightningCLI (as a supplement to sys.argv).

    LightningCLI args expect one of two formats (see below). We implement the first one for now (it worked)

    1. list of strings, where each string is a command line argument (e.g. args = ["fit", --trainer.max_epochs=100", "--model.encoder_layers", "24"])
    2. A config dictionary resembling the config structure of the CLI arguments (see `single_cellm --print_config`). E.g.
        args = {
            "subcommand": "fit",
            "trainer": {
                "max_epochs": 100,
            },
            "model": {},
        }

    Args:
        wandb_config:

    Returns:
        Arguments in format suitable for LightningCLI
    """
    args = Namespace()
    for key, value in wandb_config.items():
        setattr(args, key, value)

    return ["fit"] + [f"--{key}={value}" for key, value in wandb_config.items()]


def sweep_train(args):
    run = init_wandb(args)
    try:
        # process the fetched wandb args
        for param, param_value in run.config.items():
            if param_value == "null":
                wandb.config[param] = None

        print(f"Performing sweep training run")

        cli_mimic_args = wandb_config_to_args(wandb.config)
        # Make sure that LightningCLI does not parse sys.argv (it prints a warning)
        sys_argv_backup = sys.argv
        sys.argv = sys.argv[:1]
        training_main(args=cli_mimic_args)
        sys.argv = sys_argv_backup

        wandb.finish(0)
    except Exception as e:
        # log errors before finishing job
        import traceback

        print(e)
        print(traceback.print_exc())
        wandb.finish(-1)

    if args.wandb_mode == "offline":
        run_dir = run.dir[:-6]
        command = f"wandb sync --id {run.id} {run_dir}"
        subprocess.run(command, shell=True)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--wandb_mode", choices=["offline", "online", "disabled"], default="online"
    )
    parser.add_argument(
        "--sweep_id",
        type=str,
        default=None,
        help="Start process attaching to provided sweep ID. If not provided, a new sweep is started",
    )
    parser.add_argument(
        "--sweep_config",
        type=Path,
        default=None,
        help="Path to sweep config file (yaml). If not provided, use the one defined in config.yaml",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of runs to perform before dying. If 0, runs forever",
    )

    args = parser.parse_args()

    if args.sweep_id is None:  # start new sweep
        logging.info("Starting new sweep")
        if args.sweep_config is None:
            logging.info(
                "Using sweep config from config.yaml (field SWEEP_HYPERPARAMETERS)"
            )
            sweep_configuration = config["SWEEP_HYPERPARAMETERS"]
        else:
            with open(args.sweep_config, "r") as f:
                sweep_configuration = yaml.safe_load(f)
        sweep_id = wandb.sweep(
            sweep=sweep_configuration,
            entity="single-cellm",
            project="JointEmbed_Training",
        )
        args.sweep_id = sweep_id
        print(f"W&B Sweep initialized with ID: {args.sweep_id}")
    else:
        assert args.sweep_id is not None, "No sweep ID provided"

    print(f"Starting {args.num_runs} runs in this instance")
    wandb.agent(
        args.sweep_id,
        function=partial(sweep_train, args=args),
        count=args.num_runs,
        entity="single-cellm",
        project="JointEmbed_Training",
    )


if __name__ == "__main__":
    main()
