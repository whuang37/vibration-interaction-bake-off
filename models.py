from typing import Any
from pathlib import Path

import pickle
import numpy as np
from sklearn.svm import SVC

from stream_analyzer import StreamAnalyzer


def get_ear(device: int) -> StreamAnalyzer:
    """Gets a stream analyzer object to capture the FFT of inputted audio waves.
    Investigate configurations here for different preprocessing options.

    Args:
        device (int): device index for input audio device to use.

    Returns:
        StreamAnalyzer: Class for capture and FFT processing of audio signals.
    """
    return StreamAnalyzer(device=device, n_frequency_bins=512, FFT_window_size_ms=50)


def make_dummy_spectrum(
    n_bins: int = 512, peaks: list[int] = [50, 150, 300], rate: int = 1000
) -> tuple[list[float], list[float]]:
    """Generates a dummy FFT spectrum.

    Args:
        n_bins (int, optional): Number of frequency bins to generate. Defaults to 512.
        peaks (list[int], optional): Specific bin locations for a peak. Defaults to [50, 150, 300].
        rate (int, optional): Rate of sampling for bins. Defaults to 1000.

    Returns:
        tuple[list[float],list[float]]: _description_
    """
    freqs = np.linspace(0, rate / 2, n_bins)
    amps = np.zeros_like(freqs)
    for p in peaks:
        amps += np.exp(-0.5 * ((freqs - p) / 5) ** 2) * 5
    amps += 0.2 * np.random.rand(n_bins)
    return freqs, amps


def get_audio_features(ear: StreamAnalyzer) -> tuple[list[float], list[float]]:
    """Function called every frame which retrieves the audio features generated
    from a given StreamAnalyzer class. Intended for students to make preprocessing changes.
    Will be visualized in the visualizer afterwards.

    Args:
        ear (StreamAnalyzer): Class for capture and FFT processing of audio signals

    Returns:
        tuple[list[float], list[float]]: List of frequency bin values and corresponding amplitudes
    """
    if ear is None:
        return make_dummy_spectrum()
    else:
        freqs, amps, _, _ = ear.get_audio_features()
        return freqs, amps


# model training
def preprocess_data(X: np.ndarray) -> np.ndarray:
    """Preprocessing before training and prediction. Data will not be visualized
    but will enable specific normalizations/preprocessing techniques for machine learning.

    Args:
        X (np.ndarray): (N,D) Array of data with N total trials for D bins in each trial.
        Collected from either the collected data in training or individual frames from prediction

    Returns:
        np.ndarray: (N, D) the preprocessed array in the same shape.
    """
    return X


def train_model(
    X: np.ndarray, y: list[int] | np.ndarray, class_names: list[str]
) -> tuple[Any, list[str]]:
    """Interface to train a specific model. Students should choose a model
    and then train it using the data from preprocess data sent in. Models can consist of interfaces
    from sklearn, pytorch, or other modules (from scratch or libraries).

    Args:
        X (np.ndarray): (N, D) Preprocessed data for N trials with D bin.
        y (list[int] | np.ndarray): (N,) Array of labels for each trial of X. Should be in integer form.
        class_names (list[str]): Decoded y values for a string name for each class.

    Returns:
        tuple[Any, list[str]]: The model and the list of class names for decoding later.
    """
    # port from java weka
    model = SVC(
        C=1.0,  # the same complexity parameter
        kernel="poly",  # polynomial kernel
        degree=1,  # exponent E = 1.0
        coef0=0.0,  # kernel constant term C = 0
        tol=0.001,  # stopping tolerance L = 0.001
        shrinking=True,  # use shrinking heuristics (Weka’s SMO always uses it)
        probability=False,  # disable probability estimates (–V -1 in Weka)
        break_ties=False,  # no direct Weka equivalent; default is fine
        random_state=1,  # seed for any randomized parts (–W 1)
        max_iter=-1,  # no limit on iterations (–V -1 / –N 0 implies full optimization)
    )

    model.fit(X, y)
    return model, class_names


def predict(model: Any, X: np.ndarray) -> np.ndarray:
    """Interface for using the model to predict some given data.

    Args:
        model (Any): Some model. Same model as one you trained in train_model
        X (np.ndarray): (N, D) N captures of D bins of data.

    Returns:
        np.ndarray: (N, ) Array of predicted labels for each trial. Array should have only integers.
    """

    # returns the index of some class name
    y_pred = model.predict(X)
    return y_pred


def save_model(model: Any, classes: list[str], save_path: Path):
    """Interface to save your generated model.

    Args:
        model (Any): Model created from train_model
        classes (list[str]): List of class names that are used to decode model prediction integers.
        save_path (Path): File path to save the model. Can only end in .model.
    """

    payload = {"model": model, "classes": classes}

    with open(save_path, "wb") as f:
        pickle.dump(payload, f)


def load_model(file_path: Path) -> tuple[Any, list[str]]:
    """Interface to load your model.

    Args:
        file_path (Path): Path of the model file to load. Can only end in .model

    Returns:
        tuple[Any, list[str]]: Model loaded and the list of class names for decoding
    """

    with open(file_path, "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload["classes"]
