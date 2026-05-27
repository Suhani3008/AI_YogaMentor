import cv2
import mediapipe as mp
import json
from angleUtils import calculate_angle

# load ideal angles
with open("angleStorage.json") as f:
    ideal_angles = json.load(f)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        landmarks = results.pose_landmarks.landmark

        def get_point(id):
            return [landmarks[id].x, landmarks[id].y]

        # calculate angles
        right_elbow = calculate_angle(get_point(12), get_point(14), get_point(16))
        left_elbow = calculate_angle(get_point(11), get_point(13), get_point(15))
        right_knee = calculate_angle(get_point(24), get_point(26), get_point(28))
        left_knee = calculate_angle(get_point(23), get_point(25), get_point(27))

        # TEMP: assume TreePose (we'll fix later)
        ideal = ideal_angles.get("TreePose", {})

        feedback = []

        if "right_elbow" in ideal and abs(right_elbow - ideal["right_elbow"]) > 20:
            feedback.append("Straighten right arm")

        if "left_knee" in ideal and abs(left_knee - ideal["left_knee"]) > 20:
            feedback.append("Adjust left knee")

        # show feedback
        y = 30
        for text in feedback:
            cv2.putText(frame, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            y += 30

    cv2.imshow("Yoga Correction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()