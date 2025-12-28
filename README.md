# 🧠 Acupressure Point Detector

A real-time AI application that detects acupressure points on the human hand using **MediaPipe**, **TensorFlow**, and **OpenCV**.  
The system predicts 3 therapeutic acupressure points (LI-4, HT-8, HT-7) and overlays glowing neon markers inside the detected hand boundary.

This project includes:
- ✔️ A deep learning **regression model** trained on synthetic hand landmark data  
- ✔️ A GUI-based real-time hand tracking + point detection app  
- ✔️ Mode filters for Cold Relief, Stress Relief, and Anxiety Relief  
- ✔️ Automatic boundary correction so points never go outside the hand  

---

## 🚀 Features

- Real-time webcam detection  
- TensorFlow model predicts 3 acupressure points from 21 hand landmarks  
- Beautiful neon visualization with labels + description  
- Points automatically constrained **inside** the hand boundary  
- Training pipeline generates 50,000 synthetic samples  
- GUI built with Tkinter  

---
