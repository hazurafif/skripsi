import os

import numpy as np
import seaborn as sns
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from data_collection import gestures, append_data
from model import model


def train_model():
    X = append_data()[0]
    y = append_data()[1]

    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

    es_callback = [EarlyStopping(monitor='val_loss', patience=10),
                   ModelCheckpoint(filepath='skripsi.h5', monitor='val_loss', save_best_only=True)]

    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['categorical_accuracy', f1_m, precision_m, recall_m])

    history = model.fit(X_train, y_train, batch_size=32, epochs=100, callbacks=[tb_callback, es_callback],
                        validation_data=(X_val, y_val))

    loss, accuracy, f1_score, precision, recall, = model.evaluate(X_test, y_test, verbose=0)
    print("Loss: {}, Accuracy: {:5.2f}%, F1 Score: {}, Precision: {}, Recall: {}".format(loss, 100 * accuracy,
                                                                                         f1_score, precision,
                                                                                         recall, ))

    yhat = model.predict(X_test)
    ytrue = np.argmax(y_test, axis=1)
    yhat = np.argmax(yhat, axis=1)
    cm = confusion_matrix(ytrue, yhat)

    heatmap(cm)
    plot_history(history)


def heatmap(cm):
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
    plt.yticks(np.arange(len(gestures)), gestures.tolist())
    plt.xticks(np.arange(len(gestures)), gestures.tolist())
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


def plot_history(history):
    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    # Plot accuracy and validation accuracy on the first subplot
    ax1.plot(history.history['categorical_accuracy'], label='Training Accuracy', color='b', linestyle='solid')
    ax1.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy', color='m', linestyle='dashed')

    # Add title, labels, and legend to the first subplot
    ax1.set_title('Model Performance')
    ax1.set_xlabel('Number of Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Plot training and validation loss on the second subplot
    ax2.plot(history.history['loss'], label='Training Loss', color='r', linestyle='solid')
    ax2.plot(history.history['val_loss'], label='Validation Loss', color='b', linestyle='dashed')

    # Add title, labels, and legend to the second subplot
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Number of Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True)

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()


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
