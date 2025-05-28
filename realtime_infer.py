import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load trained model and label encoder
model = joblib.load('gesture_rf_model.joblib')
label_encoder = joblib.load('label_encoder.joblib')

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()

print("Webcam running. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Can't receive frame.")
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    prediction_text = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Predict only if we have all 63 values
            if len(landmarks) == 63:
                prediction = model.predict([landmarks])[0]
                gesture_name = label_encoder.inverse_transform([prediction])[0]
                prediction_text = gesture_name

    # Display prediction
    cv2.putText(frame, f'Gesture: {prediction_text}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

    cv2.imshow("Real-Time Sign Language Interpreter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()