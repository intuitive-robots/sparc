"""Shared image batching for the Hugging Face detector wrappers."""

import numpy as np


def create_batches(bsz, images):
    if images.shape[0] % bsz != 0:
        num_chunks = (len(images) + bsz - 1) // bsz
        images_split = np.array_split(images, [bsz * i for i in range(1, num_chunks)])
    else:
        images_split = np.array_split(images, images.shape[0] // bsz)
    return images_split
