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
    """
    Gets an autoencoder model with the specified sizes of the layers.

    This function gets an autoencoder model using the `AE` class, implemented as a PyTorch module, with the specified
    layers' sizes.
    The autoencoder is used for the implementation of the Density Principle.

    :param list layer_sizes: A list containing the number of nodes of each layer of the encoder (decoder built with
     symmetry).

    :return: An instance of the autoencoder model.
    :rtype: AE
    """
    ae = AE(layer_sizes)
    return ae
