import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Dropout

DROPOUT_RATE = 0.2


class FCModel(tf.keras.Model):
    """Custom fully connected network"""

    def __init__(self, input_dim: int):
        super(FCModel, self).__init__()

        # first hidden layer
        self.hidden1 = Dense(
            1024, input_dim=input_dim, activation=tf.keras.activations.relu
        )

        # second hidden layer
        self.hidden2 = Dense(768, activation=tf.keras.activations.relu)

        # third hidden layer
        self.hidden3 = Dense(512, activation=tf.keras.activations.relu)

        # output layer
        self.output_layer = Dense(2, activation="softmax")

        self.drop = Dropout(DROPOUT_RATE)

    @tf.function
    def call(self, inputs):
        x = self.hidden1(inputs)
        x = self.drop(x)
        x = self.hidden2(x)
        x = self.drop(x)
        x = self.hidden3(x)
        x = self.drop(x)

        x = self.output_layer(x)
        return x


# metrics
# https://datascience.stackexchange.com/a/45166
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def get_sequential_model(input_dim: int):
    model = tf.keras.models.Sequential()
    model.add(Dense(1024, input_dim=input_dim, activation=tf.keras.activations.relu))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(768, activation=tf.keras.activations.relu))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(512, activation=tf.keras.activations.relu))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(2, activation="softmax"))

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        metrics=["accuracy", precision_m, recall_m, f1_m],
    )

    return model
