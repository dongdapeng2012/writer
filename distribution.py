import numpy as np
import random

def random_distribution(vocab_size):
    """Generate a random column of probabilities."""
    b = np.random.uniform(0.0, 1.0, size=[1, vocab_size])
    return b / np.sum(b, 1)[:, None]

def sample_distribution(distribution):# choose under the probabilities
    """Sample one element from a distribution assumed to be an array of normalized
    probabilities.
    """
    r = random.uniform(0, 1)
    s = 0
    for i in range(len(distribution[0])):
        s += distribution[0][i]
        if s >= r:
            return i
    return len(distribution) - 1
