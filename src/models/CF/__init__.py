from collections import namedtuple
from .MF import MF

MF_Params = namedtuple(
    "MF_Params",
    [
        "n_factors",
        "epochs",
        "lr",
        "reg",
        "batch_size",
        "seed",
        "lr_decay_factor",
        "max_grad_norm",
    ],
)
