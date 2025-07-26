"""
Copyright (c) 2022 PiCor Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

PiCor Training Script

This module provides the main entry point for training PiCor agents.
It handles argument parsing and initializes the training runner.

Author: Fengshuo Bai
"""

import argparse

import torch._dynamo.config
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

from picor_runner import PiCorRunner

# Configure torch dynamo for better performance
torch._dynamo.config.cache_size_limit = 128

# Constants for default values
DEFAULT_ENV_NAME = "metaworld_mt10"
DEFAULT_EXPERIMENT = "picor"
DEFAULT_CONFIG_PATH = "config/picor.yaml"
DEFAULT_SEED = 42
DEFAULT_DEVICE = "0"
DEFAULT_NUM_TASK = 10
DEFAULT_BUFFER_SIZE = 1000000
DEFAULT_NUM_RANDOM_STEPS = 5000
DEFAULT_NUM_TRAIN_STEPS = 500000
DEFAULT_NUM_EVAL_EPISODES = 2
DEFAULT_LOG_FREQUENCY = 10000
DEFAULT_EVAL_FREQUENCY = 10000
DEFAULT_SAVE_FREQUENCY = 100000


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for PiCor training.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="PiCor Training Script")

    # Experiment settings
    parser.add_argument("--env_name", type=str, default=DEFAULT_ENV_NAME, help="Environment name")
    parser.add_argument(
        "--experiment", type=str, default=DEFAULT_EXPERIMENT, help="Experiment name"
    )
    parser.add_argument(
        "--config_path", type=str, default=DEFAULT_CONFIG_PATH, help="Path to configuration file"
    )
    parser.add_argument(
        "--model_path", type=str, default="", help="Path to pre-trained model (for loading)"
    )
    parser.add_argument(
        "--exec_type",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="Execution type: train, eval",
    )

    # Training parameters
    parser.add_argument(
        "--num_train_steps",
        type=int,
        default=DEFAULT_NUM_TRAIN_STEPS,
        help="Number of training steps",
    )
    parser.add_argument(
        "--num_random_steps",
        type=int,
        default=DEFAULT_NUM_RANDOM_STEPS,
        help="Number of random steps",
    )
    parser.add_argument(
        "--buffer_size", type=int, default=DEFAULT_BUFFER_SIZE, help="Buffer capacity"
    )
    parser.add_argument("--num_task", type=int, default=DEFAULT_NUM_TASK, help="Number of tasks")
    parser.add_argument(
        "--device", type=str, default=DEFAULT_DEVICE, help="Device to use for training"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument(
        "--num_eval_episodes",
        type=int,
        default=DEFAULT_NUM_EVAL_EPISODES,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--log_frequency", type=int, default=DEFAULT_LOG_FREQUENCY, help="Logging frequency"
    )
    parser.add_argument(
        "--eval_frequency", type=int, default=DEFAULT_EVAL_FREQUENCY, help="Evaluation frequency"
    )
    parser.add_argument(
        "--save_frequency", type=int, default=DEFAULT_SAVE_FREQUENCY, help="Save frequency"
    )
    parser.add_argument("--wandb", action="store_true", default=False, help="Enable wandb tracking")
    parser.add_argument(
        "--enable_corr", action="store_true", default=True, help="Enable policy correction"
    )
    parser.add_argument(
        "--compile", action="store_true", default=False, help="Enable torch.compile optimization"
    )
    parser.add_argument(
        "--cudagraphs", action="store_true", default=False, help="Enable CUDA Graph optimization"
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="Enable automatic mixed precision training",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead"],
        help="Torch compile mode",
    )
    parser.add_argument(
        "--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"], help="AMP dtype"
    )

    args = parser.parse_args()
    return args


def main() -> None:
    """Main entry point for PiCor training."""
    args = parse_args()
    runner = PiCorRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
