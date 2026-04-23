# -------------------- IMPORT LIBRARIES --------------------
import cv2
import pygame
import numpy as np
import pandas as pd
import mediapipe as mp
from matplotlib import pyplot as plt
import time
import os

# -------------------- VARIABLES --------------------
WEBCAM = 0
DEFAULT_EYE_OPENNESS = 0.50

start_time = None
alerts = []

right_eye_distances = []
left_eye_distances = []

data = pd.DataFrame(columns=["Eye_Openness_Average"])

RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

mp_face_mesh = mp.solutions.face_mesh

# -------------------- CREATE ALERT FOLDER --------------------
if not os.path.exists("alerts"):
    os.makedirs("alerts")

# -------------------- CAMERA --------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# -------------------- FUNCTIONS --------------------
def calculate_eye_height(points):
    A = np.linalg.norm(np.array(points[15]) - np.array(points[1]))
    B = np.linalg.norm(np.array(points[14]) - np.array(points[2]))
    C = np.linalg.norm(np.array(points[13]) - np.array(points[3]))
    D = np.linalg.norm(np.array(points[12]) - np.array(points[4]))
    E = np.linalg.norm(np.array(points[11]) - np.array(points[5]))
    F = np.linalg.norm(np.array(points[10]) - np.array(points[6]))
    G = np.linalg.norm(np.array(points[9]) - np.array(points[7]))
    H = np.linalg.norm(np.array(points[0]) - np.array(points[8]))

    return (A + B + C + D + E + F + G) / (2 * H)

# -------------------- ALARM --------------------
pygame.init()
pygame.mixer.init()

sound_path = "Cod/Sample1.mp3"   # using mp3 file

if os.path.exists(sound_path):
    ALARM = pygame.mixer.Sound(sound_path)
    ALARM.set_volume(1.0)
else:
    print("⚠️ Alarm file not found!")
    ALARM = None

def sound_alert():
    if ALARM and not pygame.mixer.get_busy():
        ALARM.play(-1)

def stop_alarm():
    pygame.mixer.stop()

# -------------------- DETECTION --------------------
try:
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_h, img_w = frame.shape[:2]

            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                facial_landmarks = np.array(
                    [np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                     for p in results.multi_face_landmarks[0].landmark])

                right_eye = facial_landmarks[RIGHT_EYE]
                left_eye = facial_landmarks[LEFT_EYE]

                right_eye_distance = calculate_eye_height(right_eye)
                left_eye_distance = calculate_eye_height(left_eye)

                right_eye_distances.append(right_eye_distance)
                left_eye_distances.append(left_eye_distance)

                if len(right_eye_distances) > 10:
                    right_eye_distances.pop(0)
                if len(left_eye_distances) > 10:
                    left_eye_distances.pop(0)

                right_eye_avg = np.mean(right_eye_distances)
                left_eye_avg = np.mean(left_eye_distances)

                eye_avg = (right_eye_avg + left_eye_avg) / 2

                data = pd.concat(
                    [data, pd.DataFrame({"Eye_Openness_Average": [eye_avg]})],
                    ignore_index=True
                )

                # -------------------- GRAPH --------------------
                if len(data) % 10 == 0:
                    plt.clf()
                    plt.plot(data["Eye_Openness_Average"])
                    plt.title("Eye Openness")
                    plt.pause(0.001)

                # -------------------- DROWSINESS LOGIC --------------------
                if eye_avg < DEFAULT_EYE_OPENNESS:
                    if start_time is None:
                        start_time = time.time()

                    elapsed = time.time() - start_time

                    if elapsed > 2:
                        status = "DROWSY"
                        color = (0, 0, 255)
                        sound_alert()

                        cv2.putText(frame, '⚠️ DROWSINESS ALERT!', (100, 100), 0, 1, (0, 0, 255), 3)

                        filename = f"alerts/drowsy_{int(time.time())}.jpg"
                        cv2.imwrite(filename, frame)

                        alerts.append(f"Drowsy detected at {time.ctime()}")

                else:
                    start_time = None
                    status = "AWAKE"
                    color = (0, 255, 0)
                    stop_alarm()

                # -------------------- DISPLAY --------------------
                cv2.putText(frame, f'Status: {status}', (10, 30), 0, 0.8, color, 2)
                cv2.putText(frame, f'Openness: {eye_avg:.2f}', (10, 60), 0, 0.7, (255, 255, 255), 2)

                cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
                cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)

            cv2.imshow("Fatigue Detection System", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

except KeyboardInterrupt:
    print("Stopped manually")

# -------------------- CLEANUP --------------------
data.to_csv("dados_abertura_olhos.csv", index=False)

with open("alerts.txt", "w") as f:
    for a in alerts:
        f.write(a + "\n")

cap.release()
cv2.destroyAllWindows()
pygame.mixer.stop()
plt.close()

print("Program closed successfully")