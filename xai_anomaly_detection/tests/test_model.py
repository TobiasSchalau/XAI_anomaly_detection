""" Tests for model class
"""

from typing_extensions import assert_type
from xai_anomaly_detection.model.FCModel import FCModel, get_sequential_model
import tensorflow as tf


def test_FCModel():
    model = FCModel(20)
    assert_type(model, FCModel)

    seq_model = get_sequential_model(20)
    assert_type(seq_model, tf.keras.models.Sequential)
