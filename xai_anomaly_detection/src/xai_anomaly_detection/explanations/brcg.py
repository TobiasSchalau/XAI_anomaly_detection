"""Module to create BRCG explanations
"""
from aix360.algorithms.rbm.boolean_rule_cg import BooleanRuleCG as BRCG
from aix360.algorithms.rbm import FeatureBinarizer
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
)
import pandas as pd
from IPython.utils import io
from typing import Any, Dict


def explain_rules(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
) -> Dict[str, Any]:
    """Trains a BRCG model on given data and prints boolean explanation

    Parameters
    ----------
    x_train : pd.DataFrame
        features of data
    y_train : pd.DataFrame
        label of data
    """

    fb = FeatureBinarizer(negations=True)
    x_train_fb = fb.fit_transform(x_train)
    x_test_fb = fb.transform(x_test)

    # Generate instance of BRCG
    explainer = BRCG(silent=True)

    # Train on data
    # suppress output
    with io.capture_output():
        explainer.fit(x_train_fb, y_train)

    # compute performance metrics on test set
    y_pred = explainer.predict(x_test_fb)

    # print performance
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Balanced accuracy:", balanced_accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, pos_label=1))
    print("Recall:", recall_score(y_test, y_pred, pos_label=1))

    return explainer.explain()
