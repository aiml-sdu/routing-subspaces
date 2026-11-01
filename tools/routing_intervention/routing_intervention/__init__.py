"""routing-intervention: automated routing-direction causal intervention
pipeline on a fine-tuned LoRA checkpoint.

Public entry point:

    from routing_intervention import run_intervention
"""

from .direction import fit_routing_direction
from .runner import run_intervention
from .window import predict_window

__all__ = ["run_intervention", "predict_window", "fit_routing_direction"]
__version__ = "0.1.0"
