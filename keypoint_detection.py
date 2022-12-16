import mediapipe as mp
import numpy as np
import cv2

mp_hands = mp.solutions.hands  # hands model
mp_drawing = mp.solutions.drawing_utils  # drawing utilities


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    # Get the list of detected hands
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)


def extract_keypoints(results):
    # Initialize arrays to store the keypoints
    handlandmark = np.zeros(21 * 3)
    # Iterate through the detected hands and extract their keypoints
    if results.multi_hand_landmarks:
        for i in results.multi_hand_landmarks:
            handlandmark = np.array([[res.x, res.y, res.z] for res in
                                     i.landmark]).flatten()

    return np.concatenate([handlandmark])
