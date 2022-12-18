import os
import time

import cv2
import numpy as np
import requests

from data_collection import gestures
from keypoint_detection import draw_landmarks, extract_keypoints, mediapipe_detection, mp_hands
from train import model


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
    url_req = requests.post(url + token + "/sendPhoto" + "?chat_id=" + chat_id + "&caption=" + caption + '',
                            files=files)
    return url_req


def main():
    model.load_weights('best_model.h5')
    sequence = []
    predictions = []
    threshold = 0.7
    output_label_counter = 0

    caption = ['pasien membutuhkan makananan', 'pasien membutuhkan minuman', 'pasien membutuhkan obat-obatan',
               'pasien membutuhkan bantuan', 'pasien ingin ke toilet']

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
                    #print(keypoints)
                    print(gestures[output_label])
                    print(res[np.argmax(res)])

                    if res[np.argmax(res)] > threshold:
                        cv2.putText(image, gestures[np.argmax(res)], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,
                                        cv2.LINE_AA)
                        pesan = cv2.putText(image, 'PESAN DIKIRIM', (250, 200),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (33, 245, 22), 2, cv2.LINE_AA)

                        # if output_label == 0:
                        #     output_label_counter += 1
                        #     if output_label_counter >= 30:
                        #         files = save_image(image)
                        #         send_msg(caption[0], files)
                        #         pesan
                        #         output_label_counter = 0
                        #
                        # if output_label == 1:
                        #     output_label_counter += 1
                        #     if output_label_counter >= 30:
                        #         files = save_image(image)
                        #         send_msg(caption[1], files)
                        #         pesan
                        #         output_label_counter = 0
                        #
                        # if output_label == 2:
                        #     output_label_counter += 1
                        #     if output_label_counter >= 30:
                        #         files = save_image(image)
                        #         send_msg(caption[2], files)
                        #         pesan
                        #         output_label_counter = 0
                        #
                        # if output_label == 3:
                        #     output_label_counter += 1
                        #     if output_label_counter >= 30:
                        #         files = save_image(image)
                        #         send_msg(caption[3], files)
                        #         pesan
                        #         output_label_counter = 0
                        #
                        # if output_label == 4:
                        #     output_label_counter += 1
                        #     if output_label_counter >= 30:
                        #         files = save_image(image)
                        #         send_msg(caption[4], files)
                        #         pesan
                        #         output_label_counter = 0

            end = time.time()
            totalTime = end - start

            fps = 1 / totalTime
            print("FPS: ", fps)

            cv2.putText(image, f'FPS: {int(fps)}', (550, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,
                        cv2.LINE_AA)

            cv2.imshow('OpenCV Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
