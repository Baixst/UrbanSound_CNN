from scipy.stats import entropy
import pandas as pd
import numpy as np
import statistics

def shannon_entropy(signal):
    pd_series = pd.Series(signal)
    counts = pd_series.value_counts()
    retval = entropy(counts)

    return retval


def sigmoid(x):
    z = np.exp(-x)
    sig = 1 / (1 + z)

    return sig

