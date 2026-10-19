"""Bridge to NVIDIA's Ising-Decoding code (Apache-2.0): configuration, syndrome windows, model inputs and the logical
error rate evaluation, used unchanged so that the FP16 reference, our INT8 model and the FPGA output are judged by
the same code path.

Author: Nasir Ali, C-DAC Noida.

The reference repository (a clone of github.com/NVIDIA/Ising-Decoding) is found at ISING_DECODING_DIR, by default
third_party/Ising-Decoding (scripts/setup_reference.sh).
torch.compile is disabled (TORCHDYNAMO_DISABLE=1): the host compiler is too old for it, and eager execution computes
the same function.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
HERE = Path(__file__).resolve().parent
REF = Path(os.environ.get("ISING_DECODING_DIR", HERE / ".." / "third_party" / "Ising-Decoding")).resolve()
RUNS = Path(os.environ.get("NPD_RUNS", HERE / ".." / "runs")).resolve()   # checkpoints, results, logs
sys.path.insert(0, str(REF / "code"))

import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402

from training.distributed import DistributedManager  # noqa: E402
from workflows.config_validator import apply_public_defaults_and_model, validate_public_config  # noqa: E402


def make_cfg(distance: int, rounds: int, model_id: int = 1, extra=()):
    """The configuration run.py builds for public surface code inference."""
    with initialize_config_dir(version_base="1.3", config_dir=str(REF / "conf")):
        cfg = compose(config_name="config_public",
                      overrides=[f"distance={distance}", f"n_rounds={rounds}", f"model_id={model_id}",
                                 "workflow.task=inference", *extra])
    spec = validate_public_config(cfg)
    return apply_public_defaults_and_model(cfg, spec)


def dist_manager():
    DistributedManager.initialize()
    return DistributedManager()


def load_reference_model(path: str):
    """NVIDIA's SafeTensors loader; returns (model in eval mode, metadata)."""
    from export.safetensors_utils import load_safetensors
    model, meta = load_safetensors(path, device="cpu")
    return model.eval(), meta


def sample_windows(cfg, basis: str, num: int):
    """Detector vectors (num, 2 * T * half) of the inference datapipe for one measurement basis, and the eval module
    that turns them into model inputs."""
    from omegaconf import open_dict
    from data.factory import DatapipeFactory
    with open_dict(cfg):
        cfg.test.meas_basis_test = basis
        cfg.test.num_samples = num
    ds = DatapipeFactory.create_datapipe_inference(cfg)
    return ds


def model_inputs(cfg, model, dets: torch.Tensor, basis: str):
    """trainX (B, 4, T, D, D) for a batch of detector vectors, built by NVIDIA's eval module."""
    from omegaconf import open_dict
    from evaluation.logical_error_rate import PreDecoderMemoryEvalModule, _build_stab_maps
    with open_dict(cfg):
        cfg.test.meas_basis_test = basis
    maps = _build_stab_maps(cfg.distance, getattr(cfg.data, "code_rotation", "XV"))
    mod = PreDecoderMemoryEvalModule(model, cfg, maps, torch.device("cpu"))
    trainX, *_ = mod._batch_to_trainx_and_syndromes(dets.float())
    return trainX
