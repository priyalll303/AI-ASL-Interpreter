# 🤟 AI-Powered Word-Level Sign Language Interpreter

A real-time sign language gesture recognition system that interprets **word-level gestures** like *Hi*, *Thank You*, *I Love You*, etc., using a webcam, MediaPipe, and machine learning — all wrapped in a simple and aesthetic GUI.

## 📌 Project Highlights
- 🔍 Real-time gesture recognition using webcam
- ✋ Tracks 21 hand landmarks with MediaPipe
- 🤖 Machine learning classification (Random Forest)
- 🎨 Python Tkinter GUI with emoji feedback
- 🛠️ Custom training data with live capture
- 🧩 Easily extendable to more gestures

## 🎯 Recognized Gestures
- Hi 👋
- Thank You 🙏
- Bye 👋
- Yes 👍
- No 👎
- I Love You 🤟

## 🧪 How It Works
1. **Data Collection**  
   Collect hand gesture samples using webcam & save them as CSV files.

2. **Model Training**  
   Train a classifier on the hand landmark data (63 features per sample).

3. **Real-Time Prediction**  
   Run the interpreter to recognize gestures live from webcam feed.

4. **GUI Display**  
   Recognized gestures are shown with labels & emojis via a Python Tkinter interface.

## ⚙️ Tech Stack
- Python 3.10+
- MediaPipe
- OpenCV
- scikit-learn
- Tkinter
- joblib

## 🚀 Setup Instructions
```bash
# 1. Clone the repo
git clone https://github.com/yourusername/ai-sign-language-interpreter

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the GUI interpreter
python interpreter_gui.py
```

> ✨ Make sure your webcam is connected and working!

## 📂 File Structure
```
├── collect_data.py
├── merge_data.py
├── train_model.py
├── interpreter_gui.py
├── models/
│   ├── gesture_model.pkl
│   └── label_encoder.pkl
├── data/
│   ├── gesture_data_hi.csv
│   ├── gesture_data_no.csv
│   └── gesture_data_all.csv
└── README.md
```

## 🔒 License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## 🙏 Credits
- Google MediaPipe
- scikit-learn documentation
- OpenCV
- Community tutorials on YouTube & GitHub

## 💡 Future Scope
- Add text-to-speech output
- Enable multi-hand gesture support
- Let users train new gestures via GUI
- Gesture chaining to form full sentences
- Mobile or web app deployment

## 📞 Contact
@tpriyal2016@gmail.com 

Drop a ⭐ if you find this helpful!
