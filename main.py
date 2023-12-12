import logging
import traceback
import tensorflow as tf
from task.utils import load_data
from hydra.utils import instantiate
from hydra.core.utils import _save_config
import os
import numpy as np
# import mlflow
import aim
from aim import Run
import argparse
from model.network import Network
from config.cdfa_multitask_config import cdfa_multitask_config
from run_cdfa_multitask import run_cdfa_experiment


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        default="cdfa_multitask",
        choices=["cdfa_multitask", 
                 "cdfa_continual", 
                 "pattern_learning_Fig2A", 
                 "pattern_learning_Fig2C",
                  "pattern_learning_Fig2D",
                   "pattern_learning_Fig2E",
                   "coincidence_detection"],
        help="Name of the experiment to run"
    )    

    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed to use for the experiment"
    )    

    parser.add_argument(
        "--data_base_path",
        type=str,
        default="data",
        help="Base path for the data"
    )

    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of runs to execute"
    )

    args = parser.parse_args()
    if args.experiment == "cdfa_multitask":
        config = cdfa_multitask_config
    # elif args.experiment == "cdfa_continual":
        # config = cdfa_continual_config
    results = []

    # Run experiments multiple times and report results
    for run_id in range(args.num_runs):
        print("Starting run {}".format(run_id))
        result = execute_cdfa_experiment(
            args.experiment, args.data_base_path, args.random_seed, config
        )
        results.append(result)
    
    print_results(args.experiment, results)


def print_results(experiment, results):
    pass



if __name__ == "__main__":
    main()