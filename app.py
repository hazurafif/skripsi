import time

import cv2
import numpy as np
from flask import Flask, render_template, Response, send_from_directory

from keypoint_detection import mp_hands, mediapipe_detection, draw_landmarks, extract_keypoints
from telegram_send import save_image, send_msg
from vis_prediction import visualize_prediction
from model import model

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


def gen():
    model.load_weights('skripsi.h5')
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
                    visualize_prediction(image, res)

                    if res[np.argmax(res)] > threshold:
                        for label in range(24):
                            if output_label == label:
                                output_label_counter += 1
                                if output_label_counter >= 50:
                                    cv2.putText(image, 'PESAN DIKIRIM', (250, 200),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
                                    files = save_image(image)
                                    send_msg(output_label + 1, files)
                                    output_label_counter = 0
                    else:
                        output_label_counter = 0

            end = time.time()
            totalTime = end - start

            fps = 1 / totalTime
            cv2.putText(image, f'FPS: {int(fps)}', (550, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,
                        cv2.LINE_AA)
            frame = cv2.imencode('.jpg', image)[1].tobytes()
            yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            key = cv2.waitKey(20)
            if key == 27:
                break


@app.route('/templates/<filename>')
def serve_image(filename):
    return send_from_directory('templates', filename, mimetype='image/jpeg')


@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
