from typing import Optional
from dataclasses import dataclass, field

@dataclass
class CFG:

    """
    """

    num_tokens: int = field(
        default = 20000,
        metadata = {'help': 'number of tokens'}
    )

    dim: int = field(
        default = 1024,
        metadata = {'help': 'dimension of the embedding'}
    )

    depth: int = field(
        default = 6,
        metadata = {'help': 'depth of the transformer'}
    )

    heads: int = field(
        default = 16,
        metadata = {'help': 'number of heads in the transformer'}
    )

    dim_head: int = field(
        default = 512,
        metadata = {'help': 'dimension of the head'}
    )

