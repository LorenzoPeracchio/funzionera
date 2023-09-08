import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier
from sklearn import tree

import plotly.graph_objects as go
import matplotlib.pyplot as plt


"""
relAI - Python library for reliability assessment of ML predictions.
"""

__version__ = "0.1.0"


def create_autoencoder(layer_sizes):
    """ Gets an autoencoder model with the specified sizes of the layers.

    Parameters
    ----------
    layer_sizes: list
        A list containing the number of nodes of each layer of the encoder (decoder built with symmetry).

    Returns
    -------
        numpy.ndarray
            An instance of the autoencoder model.
    """

    ae = AE(layer_sizes)
    return ae
