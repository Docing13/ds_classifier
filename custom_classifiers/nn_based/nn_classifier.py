import numpy as np
from custom_classifiers.base.base import CustomClassifierBase
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from custom_classifiers.nn_based.settings.constants import BATCH_SIZE, EPOCHS, \
    OUTPUT_SIZE, INPUT_SHAPE, OPTIMIZER, LOSS, METRICS


class CustomNNClassifier(CustomClassifierBase):
    """
    Neural net simple classification model
    """
    def __init__(self):
        self.__model = self.__nn_model

    def fit(self, x_train, y_train) -> None:

        self.__model.fit(x_train,
                         y_train,
                         batch_size=BATCH_SIZE,
                         epochs=EPOCHS)

    def predict(self, x) -> np.ndarray:
        one_hot_ans = self.__model.predict(x)
        ans = np.argmax(one_hot_ans, axis=1)
        return ans

    @property
    def __nn_model(self) -> Sequential:
        model = Sequential()

        model.add(Dense(units=300,
                        input_shape=INPUT_SHAPE))
        model.add(Activation('sigmoid'))
        model.add(Dropout(0.15))

        model.add(Dense(units=100))
        model.add(Dropout(0.15))
        model.add(Activation('sigmoid'))

        model.add(Dense(units=OUTPUT_SIZE))
        model.add(Activation('softmax'))

        model.compile(optimizer=OPTIMIZER,
                      loss=LOSS,
                      metrics=METRICS)
        return model

