import cv2
import mediapipe as mp
import joblib
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Load model & encoder
model = joblib.load("gesture_model.pkl")
le = joblib.load("label_encoder.pkl")

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# 🎀 Cute Theme Colors
BG_COLOR = "#FFF0F5"  # lavender blush
TEXT_COLOR = "#FF69B4"  # hot pink

# 🪞 Tkinter window
window = tk.Tk()
window.title("✨ AI Sign Language Interpreter ✨")
window.geometry("900x680")
window.configure(bg=BG_COLOR)
window.resizable(False, False)

# 🧁 Title Label
title = tk.Label(window, text="🧠 Real-Time Gesture Predictor", 
                 font=("Comic Sans MS", 26, "bold"),
                 bg=BG_COLOR, fg="#8A2BE2")
title.pack(pady=20)

# 🫶 Prediction Bubble
prediction_label = tk.Label(window, text="Prediction: 🫶", 
                            font=("Comic Sans MS", 22),
                            bg="#FFE4E1", fg=TEXT_COLOR, 
                            bd=2, relief="solid", padx=20, pady=10)
prediction_label.pack(pady=10)

# 📸 Webcam Canvas
canvas = tk.Label(window, bd=0)
canvas.pack()

def update_frame():
    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    predicted_label = "🌸 No gesture detected"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                prediction = model.predict([landmarks])[0]
                gesture_name = le.inverse_transform([prediction])[0]
                emoji_map = {
                    "hi": "👋",
                    "thankyou": "🙏",
                    "bye": "👋",
                    "yes": "👍",
                    "no": "👎",
                    "i love you": "❤️‍🔥"
                }
                emoji = emoji_map.get(gesture_name.lower(), "🤖")
                predicted_label = f"{gesture_name.capitalize()} {emoji}"

    prediction_label.config(text=f"Prediction: {predicted_label}")

    img_pil = Image.fromarray(img_rgb)
    img_pil = img_pil.resize((800, 500))
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
