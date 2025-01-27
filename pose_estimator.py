import mediapipe as mp
import cv2
import numpy as np

class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def process_frame(self, frame):
        """Process a frame and return pose landmarks."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.pose.process(frame_rgb)

    def draw_landmarks(self, frame, results):
        """Draw pose landmarks on the frame."""
        if results.pose_landmarks:
            # Custom drawing specifications
            custom_connections = self.mp_pose.POSE_CONNECTIONS
            landmark_drawing_spec = self.mp_drawing.DrawingSpec(
                color=(0, 180, 255),  # Neon blue color
                thickness=2,
                circle_radius=2
            )
            connection_drawing_spec = self.mp_drawing.DrawingSpec(
                color=(0, 255, 255),  # Neon yellow color
                thickness=2
            )

            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                custom_connections,
                landmark_drawing_spec,
                connection_drawing_spec
            )
        return frame

    def get_landmark_coordinates(self, results):
        """Extract landmark coordinates from results."""
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
            return landmarks
        return None

    def get_pose_confidence(self, results):
        """Calculate overall pose confidence score."""
        if results.pose_landmarks:
            visible_landmarks = [
                landmark.visibility
                for landmark in results.pose_landmarks.landmark
            ]
            return np.mean(visible_landmarks)
        return 0.0

    def calculate_joint_angles(self, landmarks):
        """Calculate key joint angles from landmarks."""
        if not landmarks:
            return {}

        def calculate_angle(p1, p2, p3):
            """Calculate angle between three points."""
            v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])
            v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])
            
            cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cosine, -1.0, 1.0))
            return np.degrees(angle)

        # Calculate key joint angles
        angles = {
            'Left Elbow': calculate_angle(
                landmarks[11],  # Left shoulder
                landmarks[13],  # Left elbow
                landmarks[15]   # Left wrist
            ),
            'Right Elbow': calculate_angle(
                landmarks[12],  # Right shoulder
                landmarks[14],  # Right elbow
                landmarks[16]   # Right wrist
            ),
            'Left Knee': calculate_angle(
                landmarks[23],  # Left hip
                landmarks[25],  # Left knee
                landmarks[27]   # Left ankle
            ),
            'Right Knee': calculate_angle(
                landmarks[24],  # Right hip
                landmarks[26],  # Right knee
                landmarks[28]   # Right ankle
            )
        }
        
        return angles
