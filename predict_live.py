import cv2
import mediapipe as mp
import joblib
import numpy as np

# Load trained model and label encoder
model = joblib.load("gesture_model.pkl")
le = joblib.load("label_encoder.pkl")

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()

print("Live Gesture Prediction Started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Can't receive frame.")
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    predicted_label = "No gesture"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                prediction = model.predict([landmarks])[0]
                predicted_label = le.inverse_transform([prediction])[0]

    # Show prediction on screen
    cv2.putText(frame, f"Prediction: {predicted_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Live Gesture Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()