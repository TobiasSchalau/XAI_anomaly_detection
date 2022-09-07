import shap
import numpy as np


class shap_explanations:
    """Class for generating shap values and graphs
    """
    def __init__(self, model, x_train: np.ndarray, x_test: np.ndarray) -> None:
        self.explainer = shap.DeepExplainer(
            model, x_train[np.random.randint(x_train.shape[0], size=50), :]
        )
        self.shap_values = self.explainer.shap_values(x_test)

    def generate_summary_plot(self, columns: list, bar=False):
        """Generates summary plot of all shap values

        Parameters
        ----------
        columns : list
            feature names
        bar : bool, optional
            whether to plot a bar or dot diagram, by default dot
        """
        plottype = "bar" if bar else "dot"
        shap.summary_plot(
            self.shap_values[0], plot_type=plottype, feature_names=columns
        )

    def generate_force_plot(self, columns: list):
        """Generates force plot for a randomly selected sample

        Parameters
        ----------
        columns : list
            feature names
        """
        shap.force_plot(
            self.explainer.expected_value[0].numpy(),
            self.shap_values[0][np.random.randint(self.shap_values[0].shape, size=1)],
            features=columns,
            matplotlib=True,
        )

    def generate_collective_force_plot(self, columns: list, x_test: np.ndarray):
        """Generates a force plot for multiple instances
        Currently not working!!

        Parameters
        ----------
        columns : list
            feature names
        x_test : np.ndarray
            test features
        """

        indices = np.random.randint(x_test.shape[0], size=50)
        shap.force_plot(
            self.explainer.expected_value[0].numpy(),
            list([self.shap_values[0][i] for i in indices]),
            x_test[indices, :],
            feature_names=columns,
            link="logit"
        )
