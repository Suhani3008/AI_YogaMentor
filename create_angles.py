import cv2
import mediapipe as mp
import os
import json
from angleUtils import calculate_angle

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

dataset_path = "AI_yogaPoseDataset"

angle_data = {}

for pose_name in os.listdir(dataset_path):

    pose_folder = os.path.join(dataset_path, pose_name)

    if not os.path.isdir(pose_folder):
        continue

    for image_name in os.listdir(pose_folder):

        image_path = os.path.join(pose_folder, image_name)
        image = cv2.imread(image_path)

        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:

            landmarks = results.pose_landmarks.landmark

            def get_point(id):
                return [landmarks[id].x, landmarks[id].y]

            # calculate angles
            angles = {
                "right_elbow": calculate_angle(get_point(12), get_point(14), get_point(16)),
                "left_elbow": calculate_angle(get_point(11), get_point(13), get_point(15)),
                "right_knee": calculate_angle(get_point(24), get_point(26), get_point(28)),
                "left_knee": calculate_angle(get_point(23), get_point(25), get_point(27)),
            }

            angle_data[pose_name] = angles
            break   # take only first good image

# save JSON
with open("angleStorage.json", "w") as f:
    json.dump(angle_data, f, indent=4)

print("angleStorage.json created!")