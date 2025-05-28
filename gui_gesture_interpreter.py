import cv2
import mediapipe as mp
import joblib
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

# Load trained model and label encoder
model = joblib.load("gesture_model.pkl")
le = joblib.load("label_encoder.pkl")

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Set up webcam
cap = cv2.VideoCapture(0)

# GUI Setup
window = tk.Tk()
window.title(" AI Gesture Interpreter")
window.geometry("800x600")
window.resizable(False, False)

label = tk.Label(window, text="Prediction: ", font=("Arial", 24), fg="blue")
label.pack(pady=10)

canvas = tk.Label(window)
canvas.pack()

def update_frame():
    ret, frame = cap.read()
    if not ret:
        return

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

    label.config(text=f"Prediction: {predicted_label}")

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil)

    canvas.imgtk = img_tk
    canvas.configure(image=img_tk)
    window.after(10, update_frame)

def on_close():
    cap.release()
    window.destroy()

window.protocol("WM_DELETE_WINDOW", on_close)
update_frame()
window.mainloop()
