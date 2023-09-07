import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from ReliabilityPackage.ReliabilityClasses import AE, ReliabilityDetector, DensityPrincipleDetector
from ReliabilityPackage.ReliabilityPrivateFunctions import _train_one_epoch, _compute_synpts_accuracy, _val_scores_diff_mse, \
    _contains_only_integers, _extract_values_proportionally
import plotly.graph_objects as go
import matplotlib.pyplot as plt


# Functions


def create_autoencoder(layer_sizes):
    """
    Gets an autoencoder model with the specified sizes of the layers.

    This function gets an autoencoder model using the `AE` class, implemented as a PyTorch module, with the specified
    layers' sizes.
    The autoencoder is used for the implementation of the Density Principle.

    :param list layer_sizes: A list containing the number of nodes of each layer of the encoder (decoder built with
     symmetry).

    :return: An instance of the autoencoder model.
    :rtype: ReliabilityClasses.AE
    """
    ae = AE(layer_sizes)
    return ae
