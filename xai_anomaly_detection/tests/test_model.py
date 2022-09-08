""" Tests for model class
"""

from xai_anomaly_detection.model.FCModel import FCModel, get_sequential_model
import tensorflow as tf


def test_FCModel():
    """Test models
    """
    model = FCModel(20)
    assert type(model) is FCModel

    seq_model = get_sequential_model(20)
    assert type(seq_model) is tf.keras.models.Sequential
