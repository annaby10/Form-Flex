# 🏋️‍♂️ FormFlex — AI Squat Analyzer

FormFlex is a real-time AI-powered posture correction system specifically designed for powerlifting exercises (focusing on the Squat). It uses **YOLOv5** to isolate the person and **MediaPipe** to track 33 3D skeletal landmarks.

It calculates 11 crucial joint angles in true 3D space and feeds them into a Machine Learning (Random Forest) model to classify your squat form—alerting you immediately to bad form (like caved-in knees or forward leaning) with visual and multi-modal audio feedback.

## 🚀 Features
- **YOLOv5 Person Isolation**: Robust background removal ensures accurate MediaPipe tracking.
- **3D Kinematics**: Computes 11 joint angles using real-world coordinates (meters) making it robust to different camera angles.
- **Machine Learning Classification**: Real-time inference using a trained `Random Forest` model for predicting bad postures (e.g., Spine Neutral vs. Knees Caving In).
- **Rep Counter Engine**: State-machine built-in that tracks squat depth intelligently.
- **Form Correction Feedback**: Get specific actionable text tips on-screen.
- **Audio Alerts**: Provides live beep tones for good reps, descents, and warnings.
- **Modern UI**: Streamlit-based web dashboard.

---

## 🛠️ Setup & Installation Guide (For Collaborators)

When cloning this project, you **do not** need to upload your heavy Python packages. The pre-trained models (`.pkl`, `.onnx`, `.task`) are already included. You just need to install the dependencies listed in `requirements.txt`.

### 1. Clone the repository
```bash
git clone https://github.com/annaby10/Form-Flex.git
cd Form-Flex
```

### 2. Create a Virtual Environment (Recommended)
This keeps the project dependencies isolated from your main system.
**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```
**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Run the following command to download all necessary libraries:
```bash
pip install -r requirements.txt
```

### 4. Run the Application
Start the Streamlit web dashboard:
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 📁 Repository Structure
*   `app.py`: Main Streamlit web application and UI.
*   `pose_estimation_module.py`: The 3D kinematics engine using MediaPipe.
*   `yolo_module.py`: YOLOv5 object detection isolation logic.
*   `audio_feedback.py`: Multimodal sound generation.
*   `train.py`: Script used to train the Random Forest model.
*   `requirements.txt`: List of required Python packages (`streamlit`, `mediapipe`, `ultralytics`, `torch`, `scikit-learn`, `pygame`, etc.).
*   `squat.pkl`: The trained Random Forest classifier.
*   `pose_landmarker_lite.task`: MediaPipe neural network weights.
*   `yolov5s.onnx`: Isolated YOLOv5 target detection weights.
*   `test_squat.mp4`: A test video you can use to verify your local installation works!
