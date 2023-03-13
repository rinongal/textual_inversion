from typing import Union, Callable, Optional, Literal

import torch
from torch import Tensor

from ldm.util import default

ShuffleMode = Union[
    Literal["off"],
    Literal["on", "all"],
    Literal["trailing", "leading", "between"],
    Literal["progressive", "dynamic"]
]
ShuffleFn = Union[
    Callable[[Tensor], Tensor],
    Callable[[Tensor, Optional[int]], Tensor]
]

def idx_of(value: int, device: torch.device):
    """Helper that makes single-value tensors for some device."""
    return torch.tensor([value], dtype=torch.int64, device=device)

def shuffle_off(placeholder_embedding: Tensor, num_vectors: Optional[int]=None):
    """Performs no shuffling, but will still trim to the number of vectors."""
    num_vectors = default(num_vectors, placeholder_embedding.shape[0])
    if num_vectors == placeholder_embedding.shape[0]:
        return placeholder_embedding
    return placeholder_embedding[:num_vectors]

def shuffle_all(placeholder_embedding: Tensor, num_vectors: Optional[int]=None):
    """Shuffles all embeddings."""
    num_vectors = default(num_vectors, placeholder_embedding.shape[0])
    d = placeholder_embedding.device
    if num_vectors >= 2:
        trim_source = placeholder_embedding[:num_vectors]
        shuffle_idx = torch.randperm(num_vectors, device=d)
        return trim_source[shuffle_idx].view(trim_source.size())
    
    # No effect with fewer than 2 vectors.
    return shuffle_off(placeholder_embedding, num_vectors)

def shuffle_trailing(placeholder_embedding: Tensor, num_vectors: Optional[int]=None):
    """Shuffles everything after first embedding."""
    num_vectors = default(num_vectors, placeholder_embedding.shape[0])
    d = placeholder_embedding.device
    if num_vectors >= 3:
        trim_source = placeholder_embedding[:num_vectors]
        shuffle_idx = torch.randperm(num_vectors - 1, device=d) + 1
        shuffle_idx = torch.cat([idx_of(0, d), shuffle_idx])
        return trim_source[shuffle_idx].view(trim_source.size())

    # No effect with fewer than 3 vectors.
    return shuffle_off(placeholder_embedding, num_vectors)

def shuffle_leading(placeholder_embedding: Tensor, num_vectors: Optional[int]=None):
    """Shuffles everything before the last embedding."""
    num_vectors = default(num_vectors, placeholder_embedding.shape[0])
    d = placeholder_embedding.device
    if num_vectors >= 3:
        trim_source = placeholder_embedding[:num_vectors]
        shuffle_idx = torch.randperm(num_vectors - 1, device=d)
        shuffle_idx = torch.cat([shuffle_idx, idx_of(num_vectors - 1, d)])
        return trim_source[shuffle_idx].view(trim_source.size())

    # No effect with fewer than 3 vectors.
    return shuffle_off(placeholder_embedding, num_vectors)

def shuffle_between(placeholder_embedding: Tensor, num_vectors: Optional[int]=None):
    """Shuffles between the first and last embeddings."""
    num_vectors = default(num_vectors, placeholder_embedding.shape[0])
    d = placeholder_embedding.device
    if num_vectors >= 4:
        trim_source = placeholder_embedding[:num_vectors]
        shuffle_idx = torch.randperm(num_vectors - 2, device=d) + 1
        shuffle_idx = torch.cat([idx_of(0, d), shuffle_idx, idx_of(num_vectors - 1, d)])
        return trim_source[shuffle_idx].view(trim_source.size())

    # No effect with fewer than 4 vectors.
    return shuffle_off(placeholder_embedding, num_vectors)

def shuffle_progressive(placeholder_embedding: Tensor, num_vectors: Optional[int]=None):
    """
    Always includes the first and last embeddings (if `num_vectors` is large enough)
    while shuffling the embeddings in between.  Unlike `shuffle_dynamic`, this
    establishes stable intro and outro embeddings ASAP.

    This was made as an option for progressive words mode.
    """
    num_vectors = default(num_vectors, placeholder_embedding.shape[0])
    d = placeholder_embedding.device
    if num_vectors == 2:
        # Only `[<first>, <last>]`.
        last_idx = placeholder_embedding.shape[0] - 1
        shuffle_idx = torch.cat([idx_of(0, d), idx_of(last_idx, d)])
        return placeholder_embedding[shuffle_idx].view(num_vectors, -1)
    if num_vectors > 2:
        # Now `[<first>, ...<random[1:num_vectors-1]>, <last>]`
        last_idx = placeholder_embedding.shape[0] - 1
        shuffle_idx = torch.randperm(num_vectors-2, device=d) + 1
        shuffle_idx = torch.cat([idx_of(0, d), shuffle_idx, idx_of(last_idx, d)])
        return placeholder_embedding[shuffle_idx].view(num_vectors, -1)

    # No effect with fewer than 2 vectors.
    return shuffle_off(placeholder_embedding, num_vectors)

def shuffle_dynamic(placeholder_embedding: Tensor, num_vectors: Optional[int]=None):
    """
    Tries to always perform an embedding shuffle when possible.
    
    The type of shuffle done depends on the number of vectors:
    * 4 or more uses `between` shuffling.
    * 3 uses `trailing` shuffling.
    * 2 or less uses `all` shuffling.
    """
    num_vectors = default(num_vectors, placeholder_embedding.shape[0])
    if num_vectors >= 4: return shuffle_between(placeholder_embedding, num_vectors)
    if num_vectors == 3: return shuffle_trailing(placeholder_embedding, num_vectors)
    return shuffle_all(placeholder_embedding, num_vectors)

def get_shuffler(shuffle_mode: Union[bool, ShuffleMode]) -> ShuffleFn:
    if shuffle_mode == True: shuffle_mode = "all"
    elif shuffle_mode == "on": shuffle_mode = "all"
    elif shuffle_mode == False: shuffle_mode = "off"

    if shuffle_mode == "all": return shuffle_all
    if shuffle_mode == "dynamic": return shuffle_dynamic
    if shuffle_mode == "progressive": return shuffle_progressive
    if shuffle_mode == "between": return shuffle_between
    if shuffle_mode == "trailing": return shuffle_trailing
    if shuffle_mode == "leading": return shuffle_leading
    return shuffle_off