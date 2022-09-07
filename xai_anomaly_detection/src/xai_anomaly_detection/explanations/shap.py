import shap
import numpy as np

EXPLAINER_PATH = "/tmp/explainer"


class shap_explanations:
    def __init__(self, model, x_train: np.ndarray, x_test: np.ndarray) -> None:

        self.explainer = shap.DeepExplainer(
            model, x_train[np.random.randint(x_train.shape[0], size=50), :]
        )
        self.shap_values = self.explainer.shap_values(x_test)

    def generate_summary_plot(self, columns: list, bar=False):
        plottype = "bar" if bar else "dot"
        shap.summary_plot(
            self.shap_values[0], plot_type=plottype, feature_names=columns
        )

    def generate_force_plot(self, columns: list):
        # local explanation
        shap.initjs()
        shap.force_plot(
            self.explainer.expected_value[0].numpy(),
            self.shap_values[0][0],
            features=columns,
            matplotlib=True,
        )
