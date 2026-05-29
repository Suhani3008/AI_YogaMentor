import csv
import json
import math

input_file = "landmarks.csv"
output_file = "angleStorage.json"

pose_data = {}

# angle function
def calculate_angle(a, b, c):
    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

with open(input_file, "r") as file:
    reader = csv.DictReader(file)

    headers = reader.fieldnames

    # detect label column
    if "label" in headers:
        label_key = "label"
    elif "pose" in headers:
        label_key = "pose"
    elif "class" in headers:
        label_key = "class"
    else:
        raise Exception("No pose label column found")

    for row in reader:
        pose_name = row[label_key]

        # extract landmarks (example indices)
        def get_point(i):
            return [float(row[f"x{i}"]), float(row[f"y{i}"])]

        try:
            # compute angles using mediapipe indices
            right_elbow = calculate_angle(get_point(12), get_point(14), get_point(16))
            left_elbow = calculate_angle(get_point(11), get_point(13), get_point(15))
            right_knee = calculate_angle(get_point(24), get_point(26), get_point(28))
            left_knee = calculate_angle(get_point(23), get_point(25), get_point(27))
        except:
            continue  # skip bad rows

        angles = {
            "right_elbow": right_elbow,
            "left_elbow": left_elbow,
            "right_knee": right_knee,
            "left_knee": left_knee
        }

        if pose_name not in pose_data:
            pose_data[pose_name] = []

        pose_data[pose_name].append(angles)

# save JSON
with open(output_file, "w") as f:
    json.dump(pose_data, f, indent=4)

print("angleStorage.json created with multiple samples")