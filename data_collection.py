import os

import cv2
import mediapipe as mp
import numpy as np

from keypoint_detection import draw_landmarks
from keypoint_detection import extract_keypoints
from keypoint_detection import mediapipe_detection

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

DATA_PATH = os.path.join('C:/Rafif/SKRIPSI/Proyek Skripsi - Pycharm/DATA')
gestures = np.array(['butuh makan', 'butuh minum', 'butuh obat', 'butuh bantuan', 'butuh ke toilet'])
no_sequence = 30
sequence_length = 30


def collect_data():
    isdir = os.path.isdir(DATA_PATH)

    if isdir:
        for gesture in gestures:
            dir_max = np.max(np.array(os.listdir(os.path.join(DATA_PATH, gesture))).astype(int))
            for sequence in range(1, no_sequence + 1):
                try:
                    os.makedirs(os.path.join(DATA_PATH, gesture, str(dir_max + sequence)))
                except:
                    pass

    else:
        for gesture in gestures:
            for sequence in range(no_sequence):
                try:
                    os.makedirs(os.path.join(DATA_PATH, gesture, str(sequence)))
                except:
                    pass

    cap = cv2.VideoCapture(0)
    cap.set(3, 800)
    cap.set(4, 600)
    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for gesture in gestures:
            for sequence in range(no_sequence):
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    image, results = mediapipe_detection(frame, holistic)
                    draw_landmarks(image, results)

                    if frame_num == 0:
                        cv2.putText(image, 'MULAI MENGAMBIL DATA', (300, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (33, 245, 22), 2, cv2.LINE_AA)
                        cv2.putText(image, 'MENGAMBIL FRAME: {} VIDEO NOMOR: {}'.format(gesture, sequence), (10, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (201, 141, 0), 1, cv2.LINE_AA)
                        cv2.imshow('Hand Gesture Recognition', image)
                        cv2.waitKey(500)

                    else:
                        cv2.putText(image, 'MENGAMBIL FRAME {} VIDEO NOMOR: {}'.format(gesture, sequence), (10, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (201, 141, 0), 1, cv2.LINE_AA)
                        cv2.imshow('Hand Gesture Recognition', image)

                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, gesture, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

        cap.release()
        cv2.destroyAllWindows()
    return

if __name__ == "__main__":
    collect_data()