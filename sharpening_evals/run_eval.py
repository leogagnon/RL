# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pprint
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", message="Skipping import of cpp extensions", module="torchao")
warnings.filterwarnings("ignore", category=SyntaxWarning, module="torchao")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.datasets import AllTaskProcessedDataset, load_eval_dataset
from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.math_environment import MathEnvironment
from nemo_rl.evals.eval import MasterConfig, run_env_eval, setup
from nemo_rl.models.generation import configure_generation_config
def maybe_convert_checkpoint(model_name: str) -> str:
    """If model_name is a NeMo RL step directory, convert to HF and return the HF path.

    Detects the nemo_automodel checkpoint format by the presence of
    weights/model/.hf_metadata/fqn_to_file_index_mapping.json.
    The consolidated HF checkpoint is written to policy/hf/ and reused on subsequent runs.
    """
    import json
    import shutil
    from nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors import (
        consolidate_safetensors_files,
    )

    model_dir = os.path.join(model_name, "policy", "weights", "model")
    hf_metadata_dir = os.path.join(model_dir, ".hf_metadata")
    fqn_mapping_path = os.path.join(hf_metadata_dir, "fqn_to_file_index_mapping.json")

    if not os.path.isfile(fqn_mapping_path):
        return model_name  # not a nemo_automodel step dir, use as-is

    hf_path = os.path.join(model_name, "policy", "hf")
    if os.path.isdir(hf_path) and os.path.isfile(os.path.join(hf_path, "config.json")):
        print(f"Using cached HF checkpoint at {hf_path}")
        return hf_path

    print(f"Converting nemo_automodel checkpoint at {model_dir} to HF format...")
    os.makedirs(hf_path, exist_ok=True)

    with open(fqn_mapping_path) as f:
        fqn_to_index = json.load(f)

    consolidate_safetensors_files(
        input_dir=model_dir,
        output_dir=hf_path,
        fqn_to_index_mapping=fqn_to_index,
    )

    for fname in ("config.json", "generation_config.json"):
        src = os.path.join(hf_metadata_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, hf_path)

    tokenizer_dir = os.path.join(model_name, "policy", "tokenizer")
    if os.path.isdir(tokenizer_dir):
        for fname in os.listdir(tokenizer_dir):
            shutil.copy2(os.path.join(tokenizer_dir, fname), hf_path)

    print(f"Conversion complete. HF checkpoint at {hf_path}")
    return hf_path

TokenizerType = PreTrainedTokenizerBase


def setup_data(tokenizer: AutoTokenizer, data_config, env_configs):
    print("Setting up data...")

    # load dataset
    base_dataset = load_eval_dataset(data_config)
    rekeyed_ds = base_dataset.rekeyed_ds

    # hardcode math for now
    env_name = "math"

    if env_name == "math_multi_reward":
        raise NotImplementedError(
            "MathMultiRewardEnvironment is not supported for evaluation, "
            "please set env_name to a different environment. "
            "See https://github.com/NVIDIA-NeMo/RL/issues/2088 for more details."
        )

    env = MathEnvironment.options(
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.math_environment.MathEnvironment"
            )
        }
    ).remote(env_configs[env_name])

    dataset = AllTaskProcessedDataset(
        dataset=rekeyed_ds,
        tokenizer=tokenizer,
        default_task_data_spec=base_dataset.task_spec,
        task_data_processors=base_dataset.processor,
        max_seq_length=data_config["max_input_seq_length"],
    )

    return dataset, env, tokenizer


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="eval",
)
def main(cfg: DictConfig):
    config: MasterConfig = OmegaConf.to_container(cfg, resolve=True)

    print("Final config:")
    pprint.pprint(config)

    # Auto-detect number of GPUs if not explicitly set
    if config["cluster"].get("gpus_per_node") is None:
        import torch
        config["cluster"]["gpus_per_node"] = torch.cuda.device_count()
        print(f"Auto-detected {config['cluster']['gpus_per_node']} GPUs")

    # Append date subfolder to save_path so runs don't overwrite each other
    if config["eval"].get("save_path"):
        config["eval"]["save_path"] = os.path.join(
            config["eval"]["save_path"], datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )

    # Auto-convert NeMo RL checkpoint to HF format if needed
    config["generation"]["model_name"] = maybe_convert_checkpoint(
        config["generation"]["model_name"]
    )
    config["tokenizer"]["name"] = config["generation"]["model_name"]

    # Init ray
    ray_tmpdir = config["cluster"].get("ray_tmpdir")
    if ray_tmpdir is None:
        import getpass, glob
        matches = glob.glob(f"/tmp/{getpass.getuser()}.*")
        if matches:
            ray_tmpdir = matches[0]
    if ray_tmpdir is not None:
        # Use a fresh per-run subdirectory so stale sessions are never found.
        # Also set RAY_TMPDIR so Ray's auto-discovery doesn't pick up other users' clusters.
        ray_tmpdir = os.path.join(ray_tmpdir, f"ray_{os.getpid()}")
        os.makedirs(ray_tmpdir, exist_ok=True)
        os.environ["RAY_TMPDIR"] = ray_tmpdir
        print(f"Using ray_tmpdir: {ray_tmpdir}")
    init_ray(log_dir=ray_tmpdir)

    # Setup tokenizer
    tokenizer = get_tokenizer(config["tokenizer"])
    config["generation"] = configure_generation_config(
        config["generation"], tokenizer, is_eval=True
    )

    # Setup data
    dataset, env, tokenizer = setup_data(tokenizer, config["data"], config["env"])

    # Setup
    vllm_generation, dataloader, master_config = setup(config, tokenizer, dataset)

    # Run evaluation
    run_env_eval(vllm_generation, dataloader, env, master_config)


if __name__ == "__main__":
    main()
