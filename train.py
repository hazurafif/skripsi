import os

from keras.callbacks import TensorBoard
from keras.layers import LSTM, Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split

from data_collection import gestures
from keypoint_data import append_data

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)


def lstm_model():
    model = Sequential()

    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(gestures.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model


# Create a basic model instance
model = lstm_model()

# Display the model's architecture
# model.summary()


def train_model():
    X = append_data()[0]
    y = append_data()[1]

    # Load the weights
    # model.load_weights('weights-1.h5')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model.fit(X_train, y_train, epochs=500, callbacks=[tb_callback])

    loss, acc = model.evaluate(X_test, y_test, verbose=2)
    print("Model, accuracy: {:5.2f}%".format(100 * acc))


    # Save the weights
    model.save('weights-500-2.h5')

    return


if __name__ == "__main__":
    train_model()
