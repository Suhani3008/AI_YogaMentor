import json
import numpy as np

# -----------------------------------
# Load Ideal Angles
# -----------------------------------

with open("angleStorage.json", "r") as f:

    ideal_angles = json.load(f)

# -----------------------------------
# Feedback Generator
# -----------------------------------

def generate_feedback(
    pose_name,
    user_angles
):

    # -----------------------------------
    # Pose Exists?
    # -----------------------------------

    if pose_name not in ideal_angles:

        return ["Pose not recognized"]

    ideal = ideal_angles[pose_name]

    # -----------------------------------
    # Handle Nested Dictionary
    # -----------------------------------

    if isinstance(ideal, dict):

        if "angles" in ideal:

            ideal = ideal["angles"]

    # -----------------------------------
    # Convert List → Dict
    # -----------------------------------

    if isinstance(ideal, list):

        joints = [

            "r_elbow",
            "l_elbow",

            "r_knee",
            "l_knee",

            "r_shoulder",
            "l_shoulder",

            "r_hip",
            "l_hip"
        ]

        ideal = dict(zip(joints, ideal))

    # -----------------------------------
    # Similarity Check
    # -----------------------------------

    total_difference = 0

    valid_joints = 0

    feedback = []

    # -----------------------------------
    # Compare Joint Angles
    # -----------------------------------

    for joint in ideal:

        ideal_angle = ideal[joint]

        if not isinstance(
            ideal_angle,
            (int, float)
        ):
            continue

        user_angle = user_angles.get(
            joint,
            None
        )

        if user_angle is None:
            continue

        diff = abs(
            user_angle - ideal_angle
        )

        total_difference += diff

        valid_joints += 1

        # -----------------------------------
        # Joint Corrections
        # -----------------------------------

        if diff > 25:

            if "elbow" in joint:

                if user_angle > ideal_angle:

                    feedback.append(
                        f"Straighten your {joint.replace('_', ' ')}"
                    )

                else:

                    feedback.append(
                        f"Bend your {joint.replace('_', ' ')} more"
                    )

            elif "knee" in joint:

                if user_angle > ideal_angle:

                    feedback.append(
                        f"Reduce bend in your {joint.replace('_', ' ')}"
                    )

                else:

                    feedback.append(
                        f"Bend your {joint.replace('_', ' ')} more"
                    )

            elif "shoulder" in joint:

                feedback.append(
                    "Align your shoulders properly"
                )

            elif "hip" in joint:

                feedback.append(
                    "Keep your hips balanced"
                )

    # -----------------------------------
    # No Valid Angles
    # -----------------------------------

    if valid_joints == 0:

        return ["No proper pose detected"]

    # -----------------------------------
    # Average Difference
    # -----------------------------------

    avg_difference = (
        total_difference / valid_joints
    )

    # -----------------------------------
    # Pose Validation
    # -----------------------------------

    # Very wrong pose
    if avg_difference > 45:

        return ["Pose not matching properly"]

    # Medium wrong
    elif avg_difference > 25:

        if len(feedback) == 0:

            return ["Adjust your posture"]

        return feedback[:2]

    # Good pose
    else:

        return [
            "Good posture keep holding the pose"
        ]

