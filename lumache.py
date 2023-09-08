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

# Classes


class CosineActivation(torch.nn.Module):
    """
    A custom activation function that applies the cosine function.

    The CosineActivation class is a PyTorch module that applies the cosine activation function
    to the input tensor.

    .. note::
        This class does not have any specific parameters or attributes.

    Methods:
        forward(x):
            Applies the cosine activation to the input tensor.

    """

    def __init__(self):
        """
        Initializes an instance of the CosineActivation class.

        """
        super().__init__()

    def forward(self, x):
        """
        Applies the cosine activation to the input tensor.

        The forward method takes an input tensor and applies the cosine activation function
        element-wise, subtracting the input value from its cosine.

        :param x: The input tensor.
        :type x: torch.Tensor

        :return: The tensor with the cosine activation applied.
        :rtype: torch.Tensor

        """
        return torch.cos(x) - x


class AE(torch.nn.Module):
    """
    Autoencoder model implemented as a PyTorch module.

    The AE class represents an autoencoder model with specified sizes of the layers.
    It consists of an encoder and a decoder, both utilizing the CosineActivation as
    the activation function.

    :param layer_sizes: A list containing the sizes of the layers of the encoder (decoder built with symmetry).
    :type layer_sizes: list[int]

    :ivar encoder: The encoder module.
    :vartype encoder: torch.nn.Sequential
    :ivar decoder: The decoder module.
    :vartype decoder: torch.nn.Sequential

    Methods:
        forward(x):
            Performs the forward pass of the autoencoder model.

    """

    def __init__(self, layer_sizes):
        """
        Initializes an instance of the AE class.

        :param layer_sizes: A list of integers containing the sizes of the layers.
        :type layer_sizes: list[int]
        """
        super().__init__()
        self.encoder = self.build_encoder(layer_sizes)
        self.decoder = self.build_decoder(layer_sizes)

    def build_encoder(self, layer_sizes):
        """
        Builds the encoder part of an autoencoder model based on the specified layer sizes.

        :param layer_sizes: A list of integers representing the number of nodes in each layer of the encoder.
        :type layer_sizes: list[int]

        :return: The encoder module of the autoencoder model.
        :rtype: torch.nn.Sequential
        """
        encoder_layers = []
        for i in range(len(layer_sizes) - 1):
            encoder_layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            encoder_layers.append(CosineActivation())
        return torch.nn.Sequential(*encoder_layers)

    def build_decoder(self, layer_sizes):
        """
        Builds the decoder part of an autoencoder model based on the specified layer sizes.

        :param layer_sizes: A list of integers representing the number of nodes in each layer of the decoder.
        :type layer_sizes: list[int]

        :return: The decoder module of the autoencoder model.
        :rtype: torch.nn.Sequential
        """
        decoder_layers = []
        for i in range(len(layer_sizes) - 1, 0, -1):
            decoder_layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i - 1]))
            decoder_layers.append(CosineActivation())
        return torch.nn.Sequential(*decoder_layers)

    def forward(self, x):
        """
        Performs the forward pass of the autoencoder model.

        The forward method takes an input tensor and passes it through the encoder,
        obtaining the encoded representation. The encoded representation is then passed
        through the decoder to reconstruct the original input.

        :param x: The input tensor.
        :type x: torch.Tensor

        :return: The reconstructed tensor.
        :rtype: torch.Tensor
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class ReliabilityDetector:
    """
    Reliability Detector for assessing the reliability of data points.

    The ReliabilityDetector class computes the reliability of data points based on
    a specified autoencoder (ae), a proxy model (clf), and an MSE threshold (mse_thresh).

    :param ae: The autoencoder model.
    :type ae: AE
    :param proxy_model: The proxy model used for the local fit reliability computation.
    :param float mse_thresh: The MSE threshold used for the density reliability computation.

    :ivar ae: The autoencoder model.
    :vartype ae: AE
    :ivar clf: The proxy model used for the local fit reliability computation.
    :ivar float mse_thresh: The MSE threshold for the density reliability computation.

    Methods:
        compute_density_reliability(x):
            Computes the density reliability of a data point.
        compute_localfit_reliability(x):
            Computes the local fit reliability of a data point.
        compute_total_reliability(x):
            Computes the combined reliability of a data point.

    """

    def __init__(self, ae, proxy_model, mse_thresh):
        """
        Initializes an instance of the ReliabilityDetector class.

        :param ae: The autoencoder model.
        :type ae: AE
        :param proxy_model: The proxy model used for the local fit reliability computation.
        :param float mse_thresh: The MSE threshold used for the density reliability computation.
        """
        self.ae = ae
        self.clf = proxy_model
        self.mse_thresh = mse_thresh

    def compute_density_reliability(self, x):
        """
        Computes the density reliability of a data point.

        The density reliability is determined by computing the mean squared error (MSE)
        between the input data point and its reconstructed representation obtained from
        the autoencoder. If the MSE is less than (or equal to) the specified MSE threshold,
        the data point is considered reliable (returns 1), otherwise unreliable (returns 0).

        :param x: The input data point.
        :type x: numpy.ndarray or torch.Tensor

        :return: The density reliability value (1 for reliable, 0 for unreliable).
        :rtype: int
        """
        mse = mean_squared_error(x, self.ae((torch.tensor(x)).float()).detach().numpy())
        return 1 if mse <= self.mse_thresh else 0

    def compute_localfit_reliability(self, x):
        """
        Computes the local fit reliability of a data point.

        The local fit reliability is determined by using the proxy model to predict the local fit
        reliability of the input data point. The input data point is reshaped to match the
        expected input format of the proxy model. The predicted reliability value is returned.

        :param x: The input data point.
        :type x: numpy.ndarray or torch.Tensor

        :return: The local fit reliability class predicted by the proxy model (1 for reliable, 0 for unreliable).
        :rtype: int
        """
        return self.clf.predict(x.reshape(1, -1))[0]

    def compute_total_reliability(self, x):
        """
        Computes the combined reliability of a data point.

        The combined reliability is determined by combining the density reliability and the
        local fit reliability. If both reliabilities are positive (1), the data point is
        considered reliable (returns True), otherwise unreliable (returns False).

        :param x: The input data point.
        :type x: numpy.ndarray or torch.Tensor

        :return: The combined reliability value (True for reliable, False for unreliable).
        :rtype: bool
        """
        density_rel = self.compute_density_reliability(x)
        localfit_rel = self.compute_localfit_reliability(x)
        return density_rel and localfit_rel


class DensityPrincipleDetector:
    """
    Density Principle Detector for assessing the density reliability of data points.

    The DensityPrincipleDetector class computes the density reliability of data points based on
    a specified autoencoder (autoencoder) and a threshold (threshold).

    :param autoencoder: The autoencoder model.
    :type autoencoder: AE
    :param float threshold: The threshold for determining the density reliability.

    :ivar ae: The autoencoder model.
    :vartype ae: AE
    :ivar float thresh: The threshold for determining the density reliability.

    Methods:
        compute_reliability(x):
            Computes the density reliability of a data point.

    """

    def __init__(self, autoencoder, threshold):
        """
        Initializes an instance of the DensityPrincipleDetector class.

        :param autoencoder: The autoencoder model.
        :type autoencoder: AE
        :param float threshold: The threshold for determining the density reliability.
        """
        self.ae = autoencoder
        self.thresh = threshold

    def compute_reliability(self, x):
        """
        Computes the density reliability of a data point.

        The density reliability is determined by computing the mean squared error (MSE)
        between the input data point and its reconstructed representation obtained from
        the autoencoder. If the MSE is less than or equal to the specified threshold,
        the data point is considered reliable (returns 1), otherwise unreliable (returns 0).

        :param x: The input data point.
        :type x: numpy.ndarray or torch.Tensor

        :return: The density reliability value (1 for reliable, 0 for unreliable).
        :rtype: int
        """
        mse = mean_squared_error(x, self.ae((torch.tensor(x)).float()).detach().numpy())
        return 1 if mse <= self.thresh else 0


# Private Functions

def _train_one_epoch(epoch_index, training_set, training_loader, optimizer, loss_function, ae):
    """
    Trains the autoencoder model for one epoch using the provided training set and loader.

    This function trains the autoencoder model for one epoch using the provided training set and data loader.
    It updates the model parameters, and calculates the training loss.

    :param int epoch_index: The index of the current epoch.
    :param numpy.ndarray training_set: The training set.
    :param torch.utils.data.DataLoader training_loader: The data loader for the training set.
    :param torch.optim.Optimizer optimizer: The optimizer used for parameter updates.
    :param torch.nn.Module loss_function: The loss function used for training.
    :param torch.nn.Module ae: The autoencoder model.

    :return: The average training loss per batch in the last epoch.
    :rtype: float
    """
    running_loss = 0.
    last_loss = 0.
    for i, data in enumerate(training_loader):
        inputs = data
        optimizer.zero_grad()
        outputs = ae(inputs.float())
        loss = loss_function(outputs, inputs.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_set) + i + 1
            print('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


def _compute_synpts_accuracy(predict_func, synpts, X_train, y_train, k=5):
    """
    Computes the accuracy of the synthetic points with the classifer.

    This function computes the accuracy on the set of synthetic points.
    The accuracy of each synthetic points is computed by comparing the predicted labels of its k nearest
    training samples to their actual labels.

    :param callable predict_func: The predict function of the classifier.
    :param numpy.ndarray synpts: The synthetic points with shape (n_synpts, n_features).
    :param numpy.ndarray X_train: The training data with shape (n_samples, n_features).
    :param numpy.ndarray y_train: The training labels with shape (n_samples,).
    :param int k: The number of nearest neighbors to consider (default: 5).

    :return: The accuracy scores associated with each synthetic point.
    :rtype: numpy.ndarray
    """
    acc_syn = []

    for i in range(len(synpts)):
        distances = np.linalg.norm(X_train - synpts[i], axis=1)
        nn = distances.argsort()[:k]
        acc_syn.append(accuracy_score(predict_func(X_train[nn, :]), y_train[nn]))
    acc_syn = np.asarray(acc_syn)

    return acc_syn


def _compute_metrics(y, ypred):
    """
    Computes various classification metrics based on the predicted and true labels.

    This function computes several classification metrics based on the predicted
    labels `ypred` and the true labels `y`. The metrics computed include accuracy,
    precision, recall, F1-score, Matthews correlation coefficient, and Brier score loss.

    :param 1d array-like y: The true labels.
    :param 1d array-like ypred: The predicted labels.

    :return: A list containing the computed metrics in the following order:
              [balanced_accuracy, precision, recall, F1-score, Matthews correlation coefficient, Brier score loss].
    :rtype: list
    """
    scores = [balanced_accuracy_score(y, ypred), precision_score(y, ypred), recall_score(y, ypred), f1_score(y, ypred),
              matthews_corrcoef(y, ypred), brier_score_loss(y, ypred)]

    return scores


def _dataset_density_reliability(X, density_principle_predictor):
    """
    Computes the density reliability of each data point in the dataset based on a density principle predictor.

    This function computes the density reliability of each data point in the dataset `X` based on a density
    principle predictor. It applies the `compute_reliability` function from the `density_principle_predictor`
    to each data point along axis 1 of `X`.

    :param array-like X: The input dataset.
    :param ReliabilityClasses.DensityPrinciplePredictor density_principle_predictor: An instance of a density principle
        predictor with a `compute_reliability` method.

    :return: An array containing the density reliability scores for each data point in `X`.
    :rtype: numpy.ndarray
    """
    return np.apply_along_axis(density_principle_predictor.compute_reliability, 1, X)


def _find_first_projection(x_i, autoencoder):
    """
    Computes the first projection of an autoencoder for a given input.

    This function computes the first projection of the `autoencoder` for a given input `x_i`. It first converts
    the input `x_i` into a tensor, passes it through the `autoencoder`, and then returns the projection as a numpy array.

    :param array-like x_i: The input data for which to compute the projection.
    :param torch.nn.Module autoencoder: The autoencoder used for projection.

    :return: The first projection of the `autoencoder` for the input `x_i` as a numpy array.
    :rtype: numpy.ndarray
    """
    pred = autoencoder((torch.tensor(x_i)).float())
    return pred.detach().numpy()


def _dataset_first_projections(X, autoencoder):
    """
    Computes the first projections of an autoencoder for each data point in a dataset.

    This function computes the first projection of the `autoencoder` for each data point in the dataset `X`.
    It applies the `find_first_projection` function along the axis 1 of `X` to compute the projections for each data point.

    :param array-like X: The dataset for which to compute the first projections.
    :param atorch.nn.Module autoencoder: The autoencoder used for projection.

    :return: An array containing the first projections of the `autoencoder` for each data point in `X`.
    :rtype: numpy.ndarray
    """
    return np.apply_along_axis(_find_first_projection, 1, X, autoencoder)


def _val_scores_diff_mse(ae, X_val, y_val, predict_func):
    """
    Computes different scores on the reliable and unreliable subset of the validation set based on different mean squared error (MSE) thresholds.

    This function computes different scores on the reliable and unreliable subset of the validation set based on different mean squared error (MSE) thresholds.
    It calculates the MSE for each data point in `X_val` and then generates a list of MSE thresholds using percentiles.
    For each threshold, it computes different scores for the reliable and unreliable samples obtained, and the number and percentage of unreliable samples.
    Finally, it returns the MSE threshold list and, for each threshold, the scores computed on the reliable and unreliable samples
    and the number and percentage of unreliable samples.

    :param torch.nn.Module ae: The autoencoder used for projection.
    :param array-like X_val: The validation dataset.
    :param array-like y_val: The validation labels.
    :param callable predict_func: The predict function of the classifier.

    :return: A tuple containing the MSE threshold list, and, for each threshold, the reliability scores, the unreliable scores,
          the number of unreliable samples and the percentage of unreliable samples
    :rtype: tuple
    """
    mse_val = []
    for i in range(len(X_val)):
        val_projections = _dataset_first_projections(X_val, ae)
        mse_val.append(mean_squared_error(X_val[i], val_projections[i]))

    mse_threshold_list = [np.percentile(mse_val, i) for i in range(2, 100)]

    rel_scores = []
    unrel_scores = []
    perc_unrel = []
    num_unrel = []
    for i in range(len(mse_threshold_list)):
        print('iterata', i + 1, "/", len(mse_threshold_list))
        DP = DensityPrincipleDetector(ae, mse_threshold_list[i])
        val_reliability = _dataset_density_reliability(X_val, DP)
        reliable_val = X_val[np.where(val_reliability == 1)]
        unreliable_val = X_val[np.where(val_reliability == 0)]
        y_reliable_val = y_val[np.where(val_reliability == 1)]
        y_unreliable_val = y_val[np.where(val_reliability == 0)]
        ypred_reliable_val = predict_func(reliable_val)
        ypred_unreliable_val = predict_func(unreliable_val)
        rel_scores.append(_compute_metrics(y_reliable_val, ypred_reliable_val))
        unrel_scores.append(_compute_metrics(y_unreliable_val, ypred_unreliable_val))
        num_unrel.append((len(X_val) - sum(val_reliability)))
        perc_unrel.append((len(X_val) - sum(val_reliability)) / len(X_val))

    return mse_threshold_list, rel_scores, unrel_scores, num_unrel, perc_unrel


def _generate_binary_vector(length):
    """
    Generates a binary vector of a specified length.

    This function generates a binary vector of a specified length, where each element in the vector is a randomly generated
    binary digit (0 or 1).

    :param int length: The desired length of the binary vector.

    :return: A list representing the generated binary vector.
    :rtype: list
    """
    binary_vector = []
    for _ in range(length):
        random_bit = random.randint(0, 1)
        binary_vector.append(random_bit)
    return binary_vector


def _contains_only_integers(array):
    """
    Checks if an array contains only integer values.

    :param array-like array: The array to be checked.

    :return: True if the array contains only integer values, False otherwise.
    :rtype: bool
    """
    integers_array = np.asarray(array).astype(int)
    check_array = np.unique(integers_array == array)
    if len(check_array) == 1 and check_array[0]:
        return True
    else:
        return False


def _extract_values_proportionally(array):
    """
    Extracts values from an array proportionally to their frequencies.

    :param array-like array: The array containing values.

    :return: A list of values extracted proportionally to their frequencies.
    :rtype: list
    """
    extracted_array = []
    frequencies = Counter(array)
    total_count = sum(frequencies.values())
    proportions = {value: count / total_count for value, count in frequencies.items()}
    for i in range(len(array)):
        extracted_array.append(random.choices(list(proportions.keys()), list(proportions.values()))[0])

    return extracted_array




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


def train_autoencoder(ae, training_set, validation_set, batchsize, epochs=1000, optimizer=None,
                      loss_function=torch.nn.MSELoss(),
                      ):
    """
    Trains the autoencoder model using the provided training and validation sets.

    This function trains the autoencoder model using the provided training and validation sets.
    It performs multiple epochs of training, updating the model parameters based on the specified optimizer
    and loss function. The training progress is evaluated on the validation set after each epoch, and the resulting
    validation loss is shown in the image.

    :param torch.nn.Module ae: The autoencoder model to be trained.
    :param numpy.ndarray training_set: The training set.
    :param numpy.ndarray validation_set: The validation set.
    :param int batchsize: The batch size used for training.
    :param int epochs: The number of training epochs (default: 1000).
    :param torch.optim.Optimizer optimizer: The optimizer used for parameter updates.
        If None, an Adam optimizer with default parameters will be used (default: None).
    :param torch.nn.Module loss_function: The loss function used for training.
        If None, the mean squared error (MSE) loss function will be used (default: torch.nn.MSELoss()).

    :return: The trained autoencoder model.
    :rtype: torch.nn.Module
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(ae.parameters(), lr=1e-4, weight_decay=1e-8)

    training_loader = DataLoader(dataset=training_set, batch_size=batchsize, shuffle=True)
    validation_loader = DataLoader(dataset=validation_set, batch_size=batchsize, shuffle=True)

    validation_loss = []
    epoch_number = 0

    for epoch in range(epochs):
        print('EPOCH {}'.format(epoch_number + 1))
        ae.train(True)
        avg_loss = _train_one_epoch(epoch_number, training_set, training_loader, optimizer, loss_function, ae)
        ae.train(False)
        running_vloss = 0.0
        for i, vdata in enumerate(validation_loader):
            vinputs = vdata
            voutputs = ae(vinputs.float())
            vloss = loss_function(voutputs, vinputs.float())
            running_vloss += vloss
        avg_vloss = running_vloss / (i + 1)
        validation_loss.append(avg_vloss.tolist())
        epoch_number += 1

    fig, ax = plt.subplots()
    plt.plot(validation_loss)
    plt.xlabel('epochs')
    plt.ylabel('Validation Loss')
    plt.title('Loss')
    plt.show()
    return ae


def get_and_train_autoencoder(training_set, validation_set, batchsize, layer_sizes=None, epochs=1000,
                                 optimizer=None, loss_function=torch.nn.MSELoss(),
                                 ):
    """
    Gets and trains an autoencoder model using the provided training and validation sets.

    This function gets an autoencoder model based on the specified layers' sizes and trains it using
    the provided training and validation sets. It performs multiple epochs of training, updating the model
    parameters based on the specified optimizer and loss function. The training progress is evaluated on
    the validation set after each epoch, and the resulting validation loss is shown in the image.

    :param numpy.ndarray training_set: The training set.
    :param numpy.ndarray validation_set: The validation set.
    :param int batchsize: The batch size used for training.
    :param list layer_sizes: A list containing the number of nodes of each layer of the encoder (decoder built with
     symmetry).
        If None, the default dimension of the encoder's layers is [dim_input, dim_input + 4, dim_input + 8,
        dim_input + 16, dim_input + 32]
    :param int epochs: The number of training epochs (default: 1000).
    :param torch.optim.Optimizer optimizer: The optimizer used for parameter updates.
        If None, an Adam optimizer with default parameters will be used (default: None).
    :param torch.nn.Module loss_function: The loss function used for training.
        If None, the mean squared error (MSE) loss function will be used (default: torch.nn.MSELoss()).

    :return: The trained autoencoder model.
    :rtype: torch.nn.Module
    """
    if layer_sizes is None:
        dim_input = training_set.shape[1]
        layer_sizes = [dim_input, dim_input + 4, dim_input + 8, dim_input + 16, dim_input + 32]
    ae = AE(layer_sizes)

    if optimizer is None:
        optimizer = torch.optim.Adam(ae.parameters(), lr=1e-4, weight_decay=1e-8)

    training_loader = DataLoader(dataset=training_set, batch_size=batchsize, shuffle=True)
    validation_loader = DataLoader(dataset=validation_set, batch_size=batchsize, shuffle=True)

    validation_loss = []
    epoch_number = 0

    for epoch in range(epochs):
        print('EPOCH {}'.format(epoch_number + 1))
        ae.train(True)
        avg_loss = _train_one_epoch(epoch_number, training_set, training_loader, optimizer, loss_function, ae)
        ae.train(False)
        running_vloss = 0.0
        for i, vdata in enumerate(validation_loader):
            vinputs = vdata
            voutputs = ae(vinputs.float())
            vloss = loss_function(voutputs, vinputs.float())
            running_vloss += vloss
        avg_vloss = running_vloss / (i + 1)
        validation_loss.append(avg_vloss.tolist())
        epoch_number += 1

    fig, ax = plt.subplots()
    plt.plot(validation_loss)
    plt.xlabel('epochs')
    plt.ylabel('Validation Loss')
    plt.title('Loss')
    plt.show()
    return ae


def compute_dataset_avg_mse(ae, X):
    """
    Compute the average mean squared error (MSE) for a given autoencoder model and dataset.

    :param torch.nn.Module ae: The autoencoder model.
    :param numpy.ndarray X: The dataset of interest

    :return: The average MSE value for the reconstructed samples.
    :rtype: float
    """
    mse = []
    for i in range(len(X)):
        mse.append(mean_squared_error(X[i],  ae((torch.tensor(X[i, :])).float()).detach().numpy()))
    return np.mean(mse)


def generate_synthetic_points(predict_func, X_train, y_train, method='GN', k=5):
    """
    Generates synthetic points based on the specified method.

    This function generates synthetic points based on the method specified in "method".
    'GN': the synthetic points are generated from the training set by adding gaussian random noise, with different
    values of variance, to the continous variables,
          and by randomly extracting, proportionally to their frequencies, the values of binary and integer variables.

    :param numpy.ndarray X_train: The training set with shape (n_samples, n_features).
    :param str method: The method used to generate synthetic points (default: 'GN').
        Currently, only the 'GN' (Gaussian Noise) method is supported.

    :return: The synthetic points generated with the specified method.
    :rtype: numpy.ndarray
    """
    allowed_methods = ['GN']
    if method not in allowed_methods:
        raise ValueError(f"Invalid value for method. Allowed values are {allowed_methods}.")

    if method == 'GN':
        noisy_data = X_train.copy()
        for j in range(2, 7):
            noisy_data_temp = X_train.copy()
            for i in range(X_train.shape[1]):
                if _contains_only_integers(X_train[:, i]):
                    noisy_data_temp[:, i] = _extract_values_proportionally(X_train[:, i])
                else:
                    noise = np.random.normal(0, j * 0.1, size=X_train.shape[0])
                    noisy_data_temp[:, i] += noise

            noisy_data = np.concatenate((noisy_data, noisy_data_temp))

    acc_syn = _compute_synpts_accuracy(predict_func, noisy_data, X_train, y_train, k)

    return noisy_data, acc_syn


def perc_mse_threshold(ae, validation_set, perc=95):
    """
    Computes the MSE threshold as a percentile of the MSE of the validation set.

    This function computes the MSE threshold as a percentile of the MSE of the validation set
    using an autoencoder model. It calculates the MSE for each sample in the validation set
    and returns the specified percentile threshold.

    :param torch.nn.Module ae: The autoencoder model.
    :param numpy.ndarray validation_set: The validation set with shape (n_samples, n_features).
    :param int perc: The percentile threshold to compute (default: 95).

    :return: The MSE threshold as the specified percentile of the MSE of the validation set.
    :rtype: float
    """
    val_projections = []
    mse_val = []
    for i in range(len(validation_set)):
        val_projections.append(ae((torch.tensor(validation_set[i, :])).float()))
        mse_val.append(mean_squared_error(validation_set[i], val_projections[i].detach().numpy()))

    return np.percentile(mse_val, perc)


def mse_threshold_plot(ae, X_val, y_val, predict_func, metric='f1_score'):
    """
    Generates a plot of performance metrics based on different MSE thresholds (selected as percentiles of the MSE of
    the validation set).

    This function generates a plot of performance metrics based on different Mean Squared Error (MSE) thresholds.
    It computes the number (and percentage) of the reliable and unreliable samples obtained with each threshold, and
    different performance metrics using the `val_scores_diff_mse` function.
    The plot shows the performance metric selected ('metric') (e.g., balanced_accuracy, precision, recall, F1-score,
    MCC, or Brier score) for reliable and unreliable samples at different MSE thresholds, and their number and
    percentage. A slider allows to move the x-axis.

    :param torch.nn.Module ae: The autoencoder used for projection.
    :param array-like X_val: The validation dataset.
    :param array-like y_val: The validation labels.
    :param callable predict_func: The predict function of the classifier.
    :param str metric: The performance metric to display on the plot. Available options: 'balanced_accuracy',
    'precision', 'recall', 'f1_score', 'mcc', 'brier_score'. Default is 'f1_score'.

    :return: A Plotly Figure object representing the MSE threshold plot.
    :rtype: go.Figure
    """
    allowed_metrics = ['balanced_accuracy', 'precision', 'recall', 'f1_score', 'mcc', 'brier_score']
    if metric not in allowed_metrics:
        raise ValueError(f"Invalid value for metric. Allowed values are {allowed_metrics}.")

    if metric == 'balanced_accuracy':
        metrx = 0
    elif metric == 'precision':
        metrx = 1
    elif metric == 'recall':
        metrx = 2
    elif metric == 'f1_score':
        metrx = 3
    elif metric == 'mcc':
        metrx = 4
    elif metric == 'brier_score':
        metrx = 5

    mse_threshold_list, rel_scores, unrel_scores, num_unrel, perc_unrel = _val_scores_diff_mse(ae, X_val, y_val,
                                                                                              predict_func)
    perc_rel = ['{:.2f}'.format((1 - perc_unrel[i]) * 100) for i in range(len(perc_unrel))]
    perc_unrel = ['{:.2f}'.format(perc_unrel[i] * 100) for i in range(len(perc_unrel))]
    num_rel = [X_val.shape[0] - i for i in num_unrel]
    percentiles = [i for i in range(2, 100)]
    y_rel = [lst[metrx] for lst in rel_scores]
    y_unrel = [lst[metrx] for lst in unrel_scores]

    htxt_rel = [str('{:.2f}'.format(perf)) for perf in y_rel]
    htxt_unrel = [str('{:.2f}'.format(perf)) for perf in y_unrel]

    # Create figure
    fig = go.Figure()
    fig.update_yaxes(range=[min(y_unrel + y_rel), max(y_unrel + y_rel)])
    # fig.update_xaxes(tickformat=".2e")

    for step in range(len(percentiles)):
        fig.add_trace(
            go.Scatter(
                x=percentiles[:step + 2],
                y=y_rel[:step + 2],
                visible=False,
                name='Reliable ' + metric,
                mode='lines',
                line=dict(color='lightgreen'),
                customdata=[[perc, num] for perc, num in zip(perc_rel[:step + 2], num_rel[:step + 2])],
                hovertemplate='%{y:.3f}<br>Reliable samples: %{customdata[1]} (%{customdata[0]}%)',
            )
        )
    for step in range(len(percentiles)):
        fig.add_trace(
            go.Scatter(
                x=percentiles[:step + 2],
                y=y_unrel[:step + 2],
                visible=False,
                name='Unreliable ' + metric,
                mode='lines',
                line=dict(color='salmon'),
                customdata=[[perc, num] for perc, num in zip(perc_unrel[:step + 2], num_unrel[:step + 2])],
                hovertemplate='%{y:.3f}<br>Unreliable samples: %{customdata[1]} (%{customdata[0]}%)',
            )
        )
    # Create and add slider
    steps = []
    for i in range(int(len(fig.data) / 2)):
        step = dict(
            method="update",
            label=str(percentiles[i]) + "°-P",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": str(metric) + " variation on the validation set at different values of the MSE threshold"},
                  {"x": [percentiles[:i + 2]]},  # Update x-axis data
                  ],
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        step["args"][0]["visible"][int(len(fig.data) / 2) + i] = True
        steps.append(step)

    # Make last traces visible
    fig.data[len(percentiles) - 1].visible = True
    fig.data[2 * len(percentiles) - 1].visible = True
    sliders = [dict(
        active=len(percentiles) - 1,
        # currentvalue={"prefix": "MSE x-limit: "},
        pad={"t": 20},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        hovermode="x unified",
        xaxis=dict(
            # tickformat='.2e',
            title='MSE threshold'
        ),
        title=str(metric) + " variation on the validation set at different values of the MSE threshold"
    )

    return fig


def mse_threshold_barplot(ae, X_val, y_val, predict_func):
    """
    Generates a bar plot of performance metrics based on different MSE thresholds.

    This function generates a bar plot of performance metrics based on different Mean Squared Error (MSE) thresholds
    (selected as percentiles of the MSE of the validation set).
    It computes different scores for the reliable and unreliable samples obtained, and the number and percentage of
    unreliable samples, using the `val_scores_diff_mse` function.
    The bar plot shows the percentage of unreliable samples, as well as various performance metrics (e.g.,
    balanced_accuracy, precision, recall, F1-score,  MCC, or Brier score) for reliable and unreliable samples at each
    MSE threshold.
    A slider allows selecting the MSE threshold and updating the plot accordingly.

    :param torch.nn.Module ae: The autoencoder used for projection.
    :param array-like X_val: The validation dataset.
    :param array-like y_val: The validation labels.
    :param callable predict_func: The predict function of the classifier.

    :return: A Plotly Figure object representing the MSE threshold bar plot.
    :rtype: go.Figure
    """
    mse_threshold_list, rel_scores, unrel_scores, num_unrel, perc_unrel = _val_scores_diff_mse(ae, X_val, y_val,
                                                                                              predict_func)
    percentiles = [i for i in range(2, 100)]

    # Creazione del layout con legenda
    hovertext = ["% Unreliable Validation Set",
                 "Balanced Accuracy Reliable set", "Balanced Accuracy Unreliable set",
                 "Precision Reliable set", "Precision Unreliable set",
                 "Recall Reliable set", "Recall Unreliable set",
                 "F1 Score Reliable set", "F1 Score Unreliable set",
                 "MCC Reliable set", "MCC Unreliable set",
                 "Brier Score Reliable set", "Brier Score Unreliable set", ]

    # Create figure
    fig = go.Figure()
    fig.update_yaxes(range=[0, 1])

    colors = ['black',
              'lightgreen', 'salmon',
              'lightgreen', 'salmon',
              'lightgreen', 'salmon',
              'lightgreen', 'salmon',
              'lightgreen', 'salmon',
              'lightgreen', 'salmon']

    # Add traces, one for each slider step
    for step in range(len(mse_threshold_list)):
        ybar = [perc_unrel[step],
                rel_scores[step][0], unrel_scores[step][0],
                rel_scores[step][1], unrel_scores[step][1],
                rel_scores[step][2], unrel_scores[step][2],
                rel_scores[step][3], unrel_scores[step][3],
                rel_scores[step][4], unrel_scores[step][4],
                rel_scores[step][5], unrel_scores[step][5],
                ]
        format_ybar = ["{:.3f}".format(val) for val in ybar]
        fig.add_trace(
            go.Bar(
                x=['% UR',
                   'R-Bal Accuracy', 'UR-Bal Accuracy',
                   'R-Precision', 'UR-Precision',
                   'R-Recall', 'UR-Recall',
                   'R-f1', 'UR-f1',
                   'R-MCC', 'UR-MCC',
                   'R-brier score', 'UR-brier score'
                   ],
                y=ybar,
                visible=False,
                marker=dict(color=colors),
                name='',
                width=0.8,
                text=format_ybar,
                showlegend=False,
                hovertext=hovertext,
                hoverinfo='text'
            )
        )

    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            label=str(i + 2) + "°-P",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": "MSE threshold: " + str('{:.4e}'.format(mse_threshold_list[i])) + ": " + str(
                      i + 1) + "°-percentile" +
                            " --- # Unreliable: " + str(num_unrel[i]) +
                            " (" + str('{:.2f}'.format(perc_unrel[i] * 100)) + "%)"}],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    # Make 10th trace visible
    fig.data[49].visible = True

    sliders = [dict(
        active=49,
        currentvalue={"prefix": "MSE threshold: "},
        pad={"t": 20},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        title="MSE threshold: " + str('{:.4e}'.format(mse_threshold_list[49])) + ": " + str(50) + "°-percentile" +
              " --- # Unreliable: " + str(num_unrel[49]) +
              " (" + str('{:.2f}'.format(perc_unrel[49] * 100)) + "%)"
    )

    return fig


def density_predictor(ae, mse_thresh):
    """
    Creates a DensityPrinciplePredictor object for a given autoencoder and MSE threshold.

    This function creates a DensityPrinciplePredictor object using the specified autoencoder and MSE threshold.
    The DensityPrinciplePredictor is a density-based predictor that assigns reliability scores to samples based on their
    reconstruction error (MSE) compared to the MSE threshold.

    :param torch.nn.Module ae: The autoencoder used for projection.
    :param float mse_thresh: The MSE threshold used for assigning reliability scores.

    :return: A DensityPrinciplePredictor object.
    :rtype: DensityPrincipleDetector
    """
    DP = DensityPrincipleDetector(ae, mse_thresh)
    return DP


def create_reliability_detector(ae, syn_pts, acc_syn, mse_thresh, acc_thresh, proxy_model='MLP'):
    """
    Gets a ReliabilityPredictor object for a given autoencoder, synthetic points, accuracy of the synthetic points,
    MSE threshold, and accuracy threshold.

    This function gets a ReliabilityPredictor object using the specified autoencoder, synthetic points, accuracy of
    the synthetic points, MSE threshold, and accuracy threshold. The ReliabilityPredictor assigns the density
    reliability of samples based on their reconstruction error (MSE), with respect to the MSE threshold, while assigns
    the local fit reliability based on the prediction of a model ('proxy_model'), trained on the synthetic points
    labelled as "local-fit" reliable/unreliable according to their associated accuracy with respect to the accuracy
    threshold.

    :param torch.nn.Module ae: The autoencoder used for projection.
    :param array-like syn_pts: The synthetic points used for training the "local-fit" reliability predictor.
    :param array-like acc_syn: The accuracy scores corresponding to the synthetic points.
    :param float mse_thresh: The MSE threshold used for assigning the density reliability scores.
    :param float acc_thresh: The accuracy threshold used for assigning the "local-fit" reliability scores.
    :param str proxy_model: The type of proxy model used for training the "local-fit"reliability predictor.
        Available options: 'MLP', 'tree'. Default is 'MLP' (Multi-Layer Perceptron).

    :return: A ReliabilityPackage object.
    :rtype: ReliabilityDetector
    """
    allowed_proxy_model = ['MLP', 'tree']
    if proxy_model not in allowed_proxy_model:
        raise ValueError(f"Invalid value for proxy_model. Allowed values are {allowed_proxy_model}.")
    y_syn_pts = []
    for i in range(len(acc_syn)):
        y_syn_pts.append(1) if acc_syn[i] >= acc_thresh else y_syn_pts.append(0)

    if proxy_model == 'MLP':
        clf = MLPClassifier(activation="tanh", random_state=42, max_iter=1000).fit(syn_pts, y_syn_pts)
    elif proxy_model == 'tree':
        clf = tree.DecisionTreeClassifier(random_state=42).fit(syn_pts, y_syn_pts)

    RP = ReliabilityDetector(ae, clf, mse_thresh)

    return RP


def compute_dataset_reliability(RD, X, mode='total'):
    """
    Computes the reliability of the samples in a dataset

    This function computes the density/local-fit/total reliability of the samples in the X dataset, based on the mode
    specified, with the ReliabilityPackage RD
    :param ReliabilityDetector RD: A ReliabilityPackage object.
    :param array-like X: the specified dataset
    :param str mode: the type of reliability to compute; Available options: 'density', 'local-fit', 'total'. Default is
    'total'
    :return: a numpy 1-D array containing the reliability of each sample (1 for reliable, 0 for unreliable)
    :rtype: numpy.ndarray
    """
    if mode == 'total':
        return np.asarray([RD.compute_total_reliability(x) for x in X])
    elif mode == 'density':
        return np.asarray([RD.compute_density_reliability(x) for x in X])
    elif mode == 'local-fit':
        return np.asarray([RD.compute_localfit_reliability(x) for x in X])
