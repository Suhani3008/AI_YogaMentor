import cv2
import mediapipe as mp
import os
import csv

# initialize mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# dataset path
dataset_path = "AI_yogaPoseDataset"

# create CSV file
with open("landmarks.csv", mode="w", newline="") as file:
    writer = csv.writer(file)

    # header
    header = ["pose"]
    for i in range(33):
        header += [f"x{i}", f"y{i}", f"z{i}"]
    writer.writerow(header)

    # loop through each pose folder
    for pose_name in os.listdir(dataset_path):
        pose_folder = os.path.join(dataset_path, pose_name)

        if not os.path.isdir(pose_folder):
            continue

        print(f"Processing {pose_name}...")

        # loop through each image
        for image_name in os.listdir(pose_folder):
            image_path = os.path.join(pose_folder, image_name)

            image = cv2.imread(image_path)
            if image is None:
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                row = [pose_name]

                for lm in results.pose_landmarks.landmark:
                    row += [lm.x, lm.y, lm.z]

                writer.writerow(row)

print("DONE — landmarks saved in landmarks.csv")