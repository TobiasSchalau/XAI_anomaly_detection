"""Module to create Protodash explanations
"""
import numpy as np
import pandas as pd
from aix360.algorithms.protodash import ProtodashExplainer
from sklearn.preprocessing import OneHotEncoder
from IPython.utils import io

def generate_protodash_explanations(x_train_df: pd.DataFrame) -> pd.DataFrame:

    # following instructions from:
    # https://github.com/Trusted-AI/AIX360/blob/master/examples/protodash/Protodash-CDC.ipynb

    # convert to numpy
    x_train = x_train_df.to_numpy()

    #sort the rows by sequence numbers in 1st column
    idx = np.argsort(x_train[:, 0])
    x_train = x_train[idx, :]

    # replace nan's (missing values) with 0's
    x_train[np.isnan(x_train)] = 0

    # delete 1st column (sequence numbers)
    # x_train = x_train[:, 1:]

    # one hot encode all features as they are categorical
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(x_train)

    explainer = ProtodashExplainer()

    # call protodash explainer
    # S contains indices of the selected prototypes
    # W contains importance weights associated with the selected prototypes
    # suppress output
    with io.capture_output():
        (W, S, _) = explainer.explain(onehot_encoded, onehot_encoded, m=5)

    # Display the prototypes along with their computed weights
    inc_prototypes = x_train_df.iloc[S, :].copy()
    # Compute normalized importance weights for prototypes
    inc_prototypes["Weights of Prototypes"] = np.around(W/np.sum(W), 2)

    return inc_prototypes
