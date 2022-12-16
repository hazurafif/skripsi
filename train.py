import os

import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import LSTM, Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from keras import backend as K
from data_collection import gestures, append_data
from keras.callbacks import EarlyStopping

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)


def lstm_model():
    model = Sequential()

    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(10, 63)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(gestures.shape[0], activation='softmax'))
    return model


# Create model instance
model = lstm_model()


def train_model():
    X = append_data()[0]
    y = append_data()[1]

    # Create an instance of the EarlyStopping callback
    early_stopping = EarlyStopping(monitor='acc', patience=20)

    # Load the weights
    # model.load_weights('weights.h5')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    # Compile the model
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc', f1_m, precision_m, recall_m])

    # Train the model, using the EarlyStopping callback
    history = model.fit(X_train, y_train, epochs=100, callbacks=[tb_callback])

    # Display the model's architecture
    # model.summary()

    yhat = model.predict(X_test)
    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()

    loss, accuracy, f1_score, precision, recall, = model.evaluate(X_test, y_test, verbose=0)
    print("Loss: {}, Accuracy: {:5.2f}%, F1 Score: {}, Precision: {}, Recall: {}".format(loss, 100 * accuracy,
                                                                                         f1_score,
                                                                                         precision, recall, ))
    # Confusion Matrix
    cm = multilabel_confusion_matrix(ytrue, yhat)
    print(cm)

    # Print the number of epochs that were run
    print("Number of epochs:", len(history.history['acc']))

    # Save the weights
    model.save('weights-coba.h5')

    return


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


if __name__ == "__main__":
    train_model()
