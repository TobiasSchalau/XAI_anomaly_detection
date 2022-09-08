import lime
import lime.lime_tabular
import numpy as np

class lime_explanations:
    """Class for create explainer instance and generate LIME visualizations
    """
    def __init__(self, x_train: np.ndarray, columns: list) -> None:
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            x_train,
            feature_names=columns,
            verbose=True,
            mode='regression'
        )

    def generate_lime_explanation(self, model, elem: np.ndarray, num_features=10, show_table=True) -> None:
        """Generates lime explanation graph for given instance

        Parameters
        ----------
        model : Any
            Model to generate explanations for
        elem : np.ndarray
            Instance to create explanations for prediction
        num_features : int, optional
            number of features to visualize, by default 10
        show_table : bool, optional
            whether to show the table with feature values, by default True
        """
        exp = self.explainer.explain_instance(elem, model.predict, num_features)
        exp.show_in_notebook(show_table=show_table)
