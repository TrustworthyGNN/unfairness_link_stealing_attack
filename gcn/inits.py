import torch
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def uniform(shape, scale=0.05):
    """Uniform init."""
    # tensorflow version: tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    # alternative pytorch: torch.empty(shape).uniform_(-scale, scale)
    return (-scale - scale) * torch.rand(shape, dtype=torch.float32) + scale


def glorot(shape):
    scale = np.sqrt(6.0 / (shape[0] + shape[1]))
    # tensorflow version: tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=torch.float32)
    return (-scale - scale) * torch.rand(shape, dtype=torch.float32) + scale


def zeros(shape):
    """All zeros."""
    return torch.zeros(shape, dtype=torch.float32)


def ones(shape):
    """All ones."""
    return torch.ones(shape, dtype=torch.float32)


if __name__ == "__main__":
    print("uniform:{}".format(uniform((2, 3))))
    print("glorot:{}".format(glorot((2, 3))))
