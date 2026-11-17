"""GPUEnsemble placement planner: decide which GPU holds which layers."""

from .devices import Device, Host, detect_host, detect_nvidia, load_rig
from .gguf import read_model
from .model import LayerProfile, ModelProfile, profile_from_gguf
from .plan import Plan, PlanError, Stage, make_plan

__all__ = [
    "Device",
    "Host",
    "LayerProfile",
    "ModelProfile",
    "Plan",
    "PlanError",
    "Stage",
    "detect_host",
    "detect_nvidia",
    "load_rig",
    "make_plan",
    "profile_from_gguf",
    "read_model",
]

__version__ = "0.1.0"
