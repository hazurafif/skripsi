import os
import time

import cv2
import numpy as np
import requests

from data_collection import gestures
from keypoint_detection import draw_landmarks, extract_keypoints, mediapipe_detection, mp_hands
from train import model

import cv2
import numpy as np

import cv2
import numpy as np


def visualize_prediction(image, prediction):
    y_offset = 30
    font_scale = 0.7
    font_thickness = 2
    text_color = (0, 0, 0)

    # Get the top three predictions
    top_3_indices = np.argpartition(prediction, -3)[-3:]
    top_3_indices = top_3_indices[np.argsort(prediction[top_3_indices])][::-1]
    top_3_predictions = prediction[top_3_indices]

    # Normalize the prediction values
    top_3_predictions = top_3_predictions / np.sum(top_3_predictions)

    # Overlay the prediction percentages on top of the image
    for i, pred in enumerate(top_3_predictions):
        if pred > 0.3:
            # Draw the label
            label = f'{gestures[top_3_indices[i]]}: {pred * 100:.2f}%'
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            label_x = 10
            label_y = y_offset + i * label_size[1]
            cv2.putText(image, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color,
                        font_thickness, cv2.LINE_AA)
    return image


def save_image(image):
    img_count = 0
    img_name = f'image_{img_count}.png'
    path_img = 'C:/Rafif/SKRIPSI/Proyek Skripsi - Pycharm/img/'
    img_count += 1
    cv2.imwrite(os.path.join(path_img, img_name), image)
    files = {'photo': open(path_img + img_name, 'rb')}
    return files


def send_msg(caption, files):
    token = "5870827651:AAH3AjqVoCO6zmKraw6a8kOlud8HCcCDLvc"
    chat_id = "1841767294"
    url = "https://api.telegram.org/bot"
    captions = {}
    file_path = 'caption.txt'
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            captions[i + 1] = line.strip()
    caption = captions[caption]
    url_req = requests.post(url + token + "/sendPhoto" + "?chat_id=" + chat_id + "&caption=" + caption + '',
                            files=files)
    return url_req


def main():
    model.load_weights('best_model.h5')
    sequence = []
    predictions = []
    threshold = 0.7
    output_label_counter = 0

    cap = cv2.VideoCapture(0)
    cap.set(3, 800)
    cap.set(4, 600)

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():

            start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, hands)
            if results.multi_hand_landmarks:
                draw_landmarks(image, results)
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-10:]

                if len(sequence) == 10:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    output_label = np.argmax(res)
                    predictions.append(output_label)
                    prediction_result = model.predict(np.expand_dims(sequence, axis=0))[0]
                    visualized_image = visualize_prediction(image, prediction_result)
                    visualized_image
                    # print(keypoints)
                    # print(gestures[output_label])
                    # print(res[np.argmax(res)])

                    # if res[np.argmax(res)] > threshold:
                    #
                    #     for label in range(24):
                    #         if output_label == label:
                    #             output_label_counter += 1
                    #             if output_label_counter >= 50:
                    #                 cv2.putText(image, 'PESAN DIKIRIM', (250, 200),
                    #                             cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
                    #                 files = save_image(image)
                    #                 send_msg(output_label + 1, files)
                    #                 output_label_counter = 0

            end = time.time()
            totalTime = end - start

            fps = 1 / totalTime
            # print("FPS: ", fps)

            cv2.putText(image, f'FPS: {int(fps)}', (550, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,
                        cv2.LINE_AA)

            cv2.imshow('Patient Hand Gesture Recognition', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
