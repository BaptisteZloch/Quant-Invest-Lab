import numpy as np
import numpy.typing as npt
import pandas as pd
from PyEMD import EMD
import pywt


def discrete_wavelet_smoothing(
    signal: pd.Series | npt.NDArray[np.float64],
    thresh: float = 0.05,
    wavelet: str = "db5",
) -> npt.NDArray[np.float64]:
    """Perform a discrete wavelet low pass filter to smoothen a signal.

    Args:
    ----
        signal (pd.Series | npt.NDArray[np.float64]): The signal to smooth, a pandas Series or numpy array containing float.

        thresh (float, optional): The wavelet threshold to smoothen data, the higher threshold the more denoised the signal will be. Defaults to 0.63.

        wavelet (str, optional): Wavelet type 'sym5', 'coif5', 'bior2.4': . Defaults to "db5".

    Returns:
    -----
        npt.NDArray[np.float64]: The smoothed signal.
    """
    if isinstance(signal, np.ndarray) is True:
        signal_np = signal
    elif isinstance(signal, pd.Series) is True:
        signal_np = signal.to_numpy()  # type: ignore
    else:
        raise TypeError("signal must be a pandas.Series or a numpy array")
    assert wavelet in pywt.wavelist(
        kind="discrete"
    ), f"Error provide a valid discrete wavelet : {pywt.wavelist(kind='discrete')}"
    coeff = pywt.wavedec(signal_np, wavelet, mode="per")
    coeff[1:] = (
        pywt.threshold(i, value=thresh * np.nanmax(signal_np), mode="soft")
        for i in coeff[1:]
    )
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per")
    if len(signal_np) % 2 == 0:
        return reconstructed_signal
    else:
        return reconstructed_signal[1:]


def emd_smoothing(
    signal: pd.Series | npt.NDArray[np.float64],
    layer_cutoff: int = 3,
) -> npt.NDArray[np.float64]:
    """Smooth a signal using Empirical Mode Decomposition (EMD), it will remove the high frequency components of the signal and return the sum of the remaining IMFs. It acts as a low pass filter.

    Args:
    -----
        signal (pd.Series | npt.NDArray[np.float64]): The signal to smooth, a pandas Series or numpy array containing float.

        layer_cutoff (int, optional): The number of the high frequency layer to remove, the higher this number is the smoother the signal resulting will be. Defaults to 3.

    Returns:
    -----
        npt.NDArray[np.float64]: The smoothed signal.
    """
    if isinstance(signal, np.ndarray) is True:
        signal_np = signal
    elif isinstance(signal, pd.Series) is True:
        signal_np = signal.to_numpy()  # type: ignore
    else:
        raise TypeError("signal must be a pandas.Series or a numpy array")
    imfs = __emd_decomposition(signal_np)  # type: ignore

    assert (
        layer_cutoff < imfs.shape[0]
        and layer_cutoff > 0
        and isinstance(layer_cutoff, int)
    ), f"layer_cutoff must an integer higher than 0 and less than {imfs.shape[0]}"

    return np.sum(imfs[layer_cutoff:, :], axis=0)


def emd_detrending(
    signal: pd.Series | npt.NDArray[np.float64],
    layer_cutoff: int = 3,
) -> npt.NDArray[np.float64]:
    """Smooth a signal using Empirical Mode Decomposition (EMD), it will remove the low frequency components of the signal and return the sum of the remaining IMFs. It acts as a low pass filter.

    Args:
    -----
        signal (pd.Series | npt.NDArray[np.float64]): The signal to smooth, a pandas Series or numpy array containing float.

        layer_cutoff (int, optional): The number of the low frequency layer to remove, the higher this number is the less detrended the resulting signal will be. Defaults to 3.

    Returns:
    -----
        npt.NDArray[np.float64]: The detrended signal.
    """
    if isinstance(signal, np.ndarray) is True:
        signal_np = signal
    elif isinstance(signal, pd.Series) is True:
        signal_np = signal.to_numpy()  # type: ignore
    else:
        raise TypeError("signal must be a pandas.Series or a numpy array")
    imfs = __emd_decomposition(signal_np)  # type: ignore

    assert (
        layer_cutoff < imfs.shape[0]
        and layer_cutoff > 0
        and isinstance(layer_cutoff, int)
    ), f"layer_cutoff must an integer higher than 0 and less than {imfs.shape[0]}"

    return np.sum(imfs[:-layer_cutoff, :], axis=0)


def __emd_decomposition(
    signal_array: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Decompose a signal using the Empirical Mode Decomposition (EMD) the imfs layers.

    Args:
    -----
        signal_array (npt.NDArray[np.float64]): The signal to perform the EMD on.

    Returns:
    -----
        npt.NDArray[np.float64]: The fft and the corresponding frequencies associated.
    """
    emd = EMD(DTYPE=np.float64, spline_kind="cubic")
    return emd(signal_array)


def fft_smoothing(
    signal: pd.Series | npt.NDArray[np.float64], frequency_cutoff: float = 0.1
) -> npt.NDArray[np.float64]:
    """Fast fourier transform smoother (low pass filter)

    Args:
    -----
        signal (pd.Series | npt.NDArray[np.float64]): The signal to smooth, a pandas Series or numpy array containing float.

        frequency_cutoff (float, optional): This frequency separates the low-frequency components that you want to keep from the high-frequency components that you want to remove. Defaults to 0.0005.

    Returns:
    -----
        npt.NDArray[np.float64]: The denoised data.
    """
    if isinstance(signal, np.ndarray) is True:
        signal_np = signal
    elif isinstance(signal, pd.Series) is True:
        signal_np = signal.to_numpy()  # type: ignore
    else:
        raise TypeError("signal must be a pandas.Series or a numpy array")

    fft_signal, freq = __fft_decomposition(signal_np)  # type: ignore
    mask = np.abs(freq) < frequency_cutoff
    return np.fft.ifft(mask * fft_signal).real


def fft_detrending(
    signal: pd.Series, frequency_cutoff: float = 0.1
) -> npt.NDArray[np.float64]:
    """Fast fourier transform detrending (high pass filter)

    Args:
    -----
        signal (pd.Series | npt.NDArray[np.float64]): The signal to smooth, a pandas Series or numpy array containing float.

        frequency_cutoff (float, optional): This frequency separates the low-frequency components that you want to remove from the high-frequency components that you want to keep. Defaults to 0.0005.

    Returns:
    -----
        npt.NDArray[np.float64]: The detrended data.
    """
    if isinstance(signal, np.ndarray) is True:
        signal_np = signal
    elif isinstance(signal, pd.Series) is True:
        signal_np = signal.to_numpy()  # type: ignore
    else:
        raise TypeError("signal must be a pandas.Series or a numpy array")

    fft_signal, freq = __fft_decomposition(signal_np)  # type: ignore
    mask = np.abs(freq) > frequency_cutoff
    return np.fft.ifft(mask * fft_signal).real


def __fft_decomposition(
    signal_array: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.float64]]:
    """Decompose a signal using the fast fourier transform and return the fft and the frequencies associated.

    Args:
    -----
        signal_array (npt.NDArray[np.float64]): The signal to perform the fft on.

    Returns:
    -----
        tuple[npt.NDArray[np.complex128], npt.NDArray[np.float64]]: The fft and the corresponding frequencies associated.
    """
    return np.fft.fft(signal_array), np.fft.fftfreq(signal_array.size)
