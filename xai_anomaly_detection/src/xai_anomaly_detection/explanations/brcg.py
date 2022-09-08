"""Module to create BRCG explanations
"""
from aix360.algorithms.rbm.boolean_rule_cg import BooleanRuleCG as BRCG
import pandas as pd

def explain_rules(x_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
    """Trains a BRCG model on given data and prints boolean explanation

    Parameters
    ----------
    x_train : pd.DataFrame
        features of data
    y_train : pd.DataFrame
        label of data
    """
    # Generate instance of BRCG
    explainer = BRCG(silent=True)
    # Train on data
    explainer.fit(x_train, y_train)

    trxf_ruleset = explainer.explain()
    print(str(trxf_ruleset))
