import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import tempfile
import time

# --------------------------
# Activity Recognition Class
# --------------------------
class ActivityRecognizer:
    def _init_(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def get_angle(self, a, b, c):
        a = np.array(a); b = np.array(b); c = np.array(c)
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    def get_body_features(self, landmarks):
        features = {}
        mp_pose = self.mp_pose

        l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

        l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

        l_ank = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        r_ank = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        l_sh = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        r_sh = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

        l_wr = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        r_wr = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        features['left_knee_angle'] = self.get_angle(l_hip, l_knee, l_ank)
        features['right_knee_angle'] = self.get_angle(r_hip, r_knee, r_ank)
        features['left_hip_angle'] = self.get_angle(l_sh, l_hip, l_knee)
        features['right_hip_angle'] = self.get_angle(r_sh, r_hip, r_knee)

        features['avg_shoulder_height'] = (l_sh[1] + r_sh[1]) / 2
        features['avg_wrist_height'] = (l_wr[1] + r_wr[1]) / 2
        features['avg_hip_height'] = (l_hip[1] + r_hip[1]) / 2

        features['left_wrist'] = l_wr
        features['right_wrist'] = r_wr

        return features

    def recognize_activity(self, features):
        if not features:
            return "Unknown", 0.0

        if (features['left_knee_angle'] > 160 and features['right_knee_angle'] > 160 and
            features['left_hip_angle'] > 160 and features['right_hip_angle'] > 160):

            if features['avg_wrist_height'] < features['avg_shoulder_height']:
                return "Waving", 0.85

            if (abs(features['left_wrist'][0] - features['right_wrist'][0]) < 0.05 and
                abs(features['left_wrist'][1] - features['right_wrist'][1]) < 0.05 and
                abs(features['avg_wrist_height'] - features['avg_shoulder_height']) < 0.1):
                return "Clapping", 0.90

            return "Standing", 0.9

        elif (features['left_knee_angle'] < 120 and features['right_knee_angle'] < 120 and
              features['left_hip_angle'] < 120 and features['right_hip_angle'] < 120):
            return "Sitting", 0.88

        elif (140 < features['left_knee_angle'] < 170 or
              140 < features['right_knee_angle'] < 170):
            return "Walking", 0.75

        return "Unknown", 0.5

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        activity, confidence = "Unknown", 0.0

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            features = self.get_body_features(results.pose_landmarks.landmark)
            activity, confidence = self.recognize_activity(features)

        cv2.putText(frame, f"{activity} ({confidence:.2f})", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame, activity, confidence


# --------------------------
# Streamlit App (Styled)
# --------------------------

# Inject custom CSS for professional UI
st.markdown("""
    <style>
        /* Background */
        .stApp {
            background-color: #4d7c8a8a;
            color: white;
            font-family: 'Poppins', sans-serif;
        }
        .st-emotion-cache-155jwzh{
            background-color:#468a9ec4;
        }
        /* Title */
        h1 {
            color: #9b5de5;
            text-align: center;
            font-weight: 700;
        }

        /* Subtitles and sidebar */
        h3, h4 {
            color: #f15bb5;
        }
        .sidebar .sidebar-content {
            background-color: #;
        }

        /* Buttons and uploaders */
        .stButton>button {
            background-color: #9b5de5;
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6em 1.2em;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #f15bb5;
            transform: scale(1.05);
        }

        /* Activity display */
        .activity-box {
            background-color: #161b22;
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            margin-top: 15px;
            box-shadow: 0 0 10px #9b5de550;
        }
        
        .st-emotion-cache-14vh5up {
            display: flex;
            -webkit-box-align: center;
            align-items: center;
            height: 100%;
            width: 100%;
            padding: 0px;
            pointer-events: auto;
            background-color: #4d7c8a8a;
            position: relative;
            z-index: 999990;
            }

    </style>
""", unsafe_allow_html=True)

st.title("🎥 Human Activity Recognition (HAR)")
st.markdown("<h3>Detects: Standing | Sitting | Walking | Waving | Clapping</h3>", unsafe_allow_html=True)

option = st.sidebar.selectbox("🎬 Choose Mode", ["Webcam", "Upload Video"])
video_placeholder = st.empty()

recognizer = ActivityRecognizer()

activity_box = st.empty()

if option == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("📂 Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame, activity, conf = recognizer.process_frame(frame)
            video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            activity_box.markdown(
                f"<div class='activity-box'><h4>🏃 Activity: <b>{activity}</b> (Confidence: {conf:.2f})</h4></div>",
                unsafe_allow_html=True)
            time.sleep(0.03)
        cap.release()

elif option == "Webcam":
    cap = cv2.VideoCapture(0)
    run = st.checkbox("🎥 Start Webcam")

    while run and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame, activity, conf = recognizer.process_frame(frame)
        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        activity_box.markdown(
            f"<div class='activity-box'><h4>🏃 Activity: <b>{activity}</b> (Confidence: {conf:.2f})</h4></div>",
            unsafe_allow_html=True)
        time.sleep(0.03)
    cap.release()