
import cv2
import mediapipe as mp
import numpy as np
import os
import csv

# -------------------------------
# MediaPipe Pose Setup
# -------------------------------

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------------------------------
# Dataset Path
# -------------------------------

DATASET_PATH = "AI_yogaPoseDataset"

# CSV Output File
OUTPUT_FILE = "landmarks.csv"

# -------------------------------
# Angle Calculation Function
# -------------------------------

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

    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


# -------------------------------
# Store All Data
# -------------------------------

data = []

# -------------------------------
# Loop Through Pose Folders
# -------------------------------

for pose_name in os.listdir(DATASET_PATH):

    pose_folder = os.path.join(DATASET_PATH, pose_name)

    # Skip non folders
    if not os.path.isdir(pose_folder):
        continue

    print(f"\nProcessing Pose: {pose_name}")

    # -------------------------------
    # Loop Through Images
    # -------------------------------

    for image_name in os.listdir(pose_folder):

        image_path = os.path.join(
            pose_folder,
            image_name
        )

        try:

            # -------------------------------
            # Read Image
            # -------------------------------

            image = cv2.imread(image_path)

            if image is None:
                print(f"Could not read: {image_name}")
                continue

            # -------------------------------
            # Resize Image
            # -------------------------------

            image = cv2.resize(image, (640, 480))

            # -------------------------------
            # Blur Reduction
            # -------------------------------

            image = cv2.GaussianBlur(
                image,
                (3, 3),
                0
            )

            # -------------------------------
            # Convert BGR to RGB
            # -------------------------------

            image_rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            # -------------------------------
            # Pose Detection
            # -------------------------------

            results = pose.process(image_rgb)

            # -------------------------------
            # Check Landmarks
            # -------------------------------

            if not results.pose_landmarks:
                print(f"No pose detected: {image_name}")
                continue

            landmarks = results.pose_landmarks.landmark

            # -------------------------------
            # Visibility Check
            # -------------------------------

            visibility_scores = [
                lm.visibility for lm in landmarks
            ]

            avg_visibility = np.mean(
                visibility_scores
            )

            if avg_visibility < 0.5:
                print(f"Low visibility skipped: {image_name}")
                continue

            # -------------------------------
            # Hip Normalization
            # -------------------------------

            hip_x = landmarks[23].x
            hip_y = landmarks[23].y
            hip_z = landmarks[23].z

            row = []

            # -------------------------------
            # Extract Landmarks
            # -------------------------------

            for lm in landmarks:

                row.append(lm.x - hip_x)
                row.append(lm.y - hip_y)
                row.append(lm.z - hip_z)

            # -------------------------------
            # Helper Function
            # -------------------------------

            def get_point(index):

                return [
                    landmarks[index].x,
                    landmarks[index].y
                ]

            # -------------------------------
            # Angle Features
            # -------------------------------

            angles = [

                # Right Elbow
                calculate_angle(
                    get_point(12),
                    get_point(14),
                    get_point(16)
                ),

                # Left Elbow
                calculate_angle(
                    get_point(11),
                    get_point(13),
                    get_point(15)
                ),

                # Right Knee
                calculate_angle(
                    get_point(24),
                    get_point(26),
                    get_point(28)
                ),

                # Left Knee
                calculate_angle(
                    get_point(23),
                    get_point(25),
                    get_point(27)
                ),

                # Right Shoulder
                calculate_angle(
                    get_point(24),
                    get_point(12),
                    get_point(14)
                ),

                # Left Shoulder
                calculate_angle(
                    get_point(23),
                    get_point(11),
                    get_point(13)
                ),

                # Right Hip
                calculate_angle(
                    get_point(12),
                    get_point(24),
                    get_point(26)
                ),

                # Left Hip
                calculate_angle(
                    get_point(11),
                    get_point(23),
                    get_point(25)
                )
            ]

            # Add Angles
            row.extend(angles)

            # Add Label
            row.append(pose_name)

            # Save Row
            data.append(row)

            print(f"Processed: {image_name}")

        except Exception as e:

            print(f"Error in {image_name}: {e}")

# -------------------------------
# CSV Header
# -------------------------------

header = []

for i in range(33):

    header += [
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ]

# Angle Names

header += [

    "r_elbow",
    "l_elbow",

    "r_knee",
    "l_knee",

    "r_shoulder",
    "l_shoulder",

    "r_hip",
    "l_hip"
]

# Label Column

header.append("label")

# -------------------------------
# Save CSV File
# -------------------------------

with open(
    OUTPUT_FILE,
    mode="w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow(header)

    writer.writerows(data)

# -------------------------------
# Final Output
# -------------------------------

print("\n================================")
print("Landmark Extraction Completed")
print("================================")

print(f"Total Samples Saved: {len(data)}")
print(f"CSV File Saved: {OUTPUT_FILE}")

