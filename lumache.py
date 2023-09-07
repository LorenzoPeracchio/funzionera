


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
