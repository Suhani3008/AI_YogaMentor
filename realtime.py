import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import random

from voice import speak

# -----------------------------------
# Load Model Files
# -----------------------------------

model = pickle.load(
    open("model.pkl", "rb")
)

le = pickle.load(
    open("label_encoder.pkl", "rb")
)

scaler = pickle.load(
    open("scaler.pkl", "rb")
)

# -----------------------------------
# MediaPipe Setup
# -----------------------------------

mp_pose = mp.solutions.pose

mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------------
# Angle Function
# -----------------------------------

def calculate_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(
        c[1] - b[1],
        c[0] - b[0]
    ) - np.arctan2(
        a[1] - b[1],
        a[0] - b[0]
    )

    angle = np.abs(
        radians * 180.0 / np.pi
    )

    if angle > 180:
        angle = 360 - angle

    return angle

# -----------------------------------
# Voice Feedback Generator
# -----------------------------------

def get_feedback(
    pose_name,
    confidence
):

    if confidence < 60:

        return random.choice([

            "Move slightly back",
            "Adjust your body position",
            "Pose not clearly visible",
            "Try making the pose properly",
            "Stand fully inside camera"

        ])

    feedback_options = {

        "plank": [

            "Keep your back straight",
            "Engage your core",
            "Do not lower your hips"

        ],

        "tree": [

            "Maintain your balance",
            "Keep your posture stable",
            "Focus on body alignment"

        ],

        "warrior2": [

            "Stretch your arms properly",
            "Bend your front knee more",
            "Keep your shoulders aligned"

        ],

        "goddess": [

            "Lower your hips slightly",
            "Keep knees outward",
            "Maintain strong posture"

        ],

        "chair": [

            "Sit lower gently",
            "Keep your back straight",
            "Raise your arms properly"

        ]
    }

    if pose_name in feedback_options:

        return random.choice(
            feedback_options[pose_name]
        )

    return "Good posture keep holding"

# -----------------------------------
# Webcam
# -----------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Camera not opening")

    exit()

# -----------------------------------
# Variables
# -----------------------------------

prev_pose = ""

stable_count = 0

last_voice_time = 0

last_feedback = ""

# -----------------------------------
# Main Loop
# -----------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # -----------------------------------
    # Resize + Flip
    # -----------------------------------

    frame = cv2.resize(
        frame,
        (640, 480)
    )

    frame = cv2.flip(frame, 1)

    # -----------------------------------
    # RGB Conversion
    # -----------------------------------

    image_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # -----------------------------------
    # Pose Detection
    # -----------------------------------

    results = pose.process(image_rgb)

    display_pose = "No Pose"

    confidence_text = ""

    feedback_text = "Stand inside camera"

    # -----------------------------------
    # Pose Found
    # -----------------------------------

    if results.pose_landmarks:

        landmarks = results.pose_landmarks.landmark

        visibility_scores = [
            lm.visibility for lm in landmarks
        ]

        avg_visibility = np.mean(
            visibility_scores
        )

        if avg_visibility > 0.5:

            # -----------------------------------
            # Normalize
            # -----------------------------------

            hip_x = landmarks[23].x
            hip_y = landmarks[23].y
            hip_z = landmarks[23].z

            data = []

            for lm in landmarks:

                data.append(lm.x - hip_x)
                data.append(lm.y - hip_y)
                data.append(lm.z - hip_z)

            # -----------------------------------
            # Angle Features
            # -----------------------------------

            def get_point(i):

                return [
                    landmarks[i].x,
                    landmarks[i].y
                ]

            angles = [

                calculate_angle(
                    get_point(12),
                    get_point(14),
                    get_point(16)
                ),

                calculate_angle(
                    get_point(11),
                    get_point(13),
                    get_point(15)
                ),

                calculate_angle(
                    get_point(24),
                    get_point(26),
                    get_point(28)
                ),

                calculate_angle(
                    get_point(23),
                    get_point(25),
                    get_point(27)
                ),

                calculate_angle(
                    get_point(24),
                    get_point(12),
                    get_point(14)
                ),

                calculate_angle(
                    get_point(23),
                    get_point(11),
                    get_point(13)
                ),

                calculate_angle(
                    get_point(12),
                    get_point(24),
                    get_point(26)
                ),

                calculate_angle(
                    get_point(11),
                    get_point(23),
                    get_point(25)
                )
            ]

            data.extend(angles)

            # -----------------------------------
            # Scaling
            # -----------------------------------

            data = np.array(data).reshape(1, -1)

            data = scaler.transform(data)

            # -----------------------------------
            # Prediction
            # -----------------------------------

            pred = model.predict(data)[0]

            prob = np.max(
                model.predict_proba(data)
            )

            pose_name = le.inverse_transform(
                [pred]
            )[0]

            confidence = prob * 100

            confidence_text = f"{confidence:.2f}%"

            # -----------------------------------
            # Stability
            # -----------------------------------

            if pose_name == prev_pose:

                stable_count += 1

            else:

                stable_count = 0

            prev_pose = pose_name

            # -----------------------------------
            # Stable Pose
            # -----------------------------------

            if stable_count > 3:

                display_pose = pose_name

            else:

                display_pose = "Detecting..."

            # -----------------------------------
            # Feedback
            # -----------------------------------

            feedback_text = get_feedback(
                pose_name,
                confidence
            )

            # -----------------------------------
            # Voice Output
            # -----------------------------------

            current_time = time.time()

            if current_time - last_voice_time > 4:

                speak(feedback_text)

                last_voice_time = current_time

            # -----------------------------------
            # Draw Landmarks
            # -----------------------------------

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

    # -----------------------------------
    # Display Pose
    # -----------------------------------

    cv2.putText(
        frame,
        f"Pose: {display_pose}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # -----------------------------------
    # Display Confidence
    # -----------------------------------

    cv2.putText(
        frame,
        f"Confidence: {confidence_text}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 0),
        2
    )

    # -----------------------------------
    # Display Feedback
    # -----------------------------------

    cv2.putText(
        frame,
        f"Feedback: {feedback_text}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    # -----------------------------------
    # Show Webcam
    # -----------------------------------

    cv2.imshow(
        "Yoga AI Mentor",
        frame
    )

    # -----------------------------------
    # Exit
    # -----------------------------------

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break

# -----------------------------------
# Release Resources
# -----------------------------------

cap.release()

cv2.destroyAllWindows()

