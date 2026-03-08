"""
Pose Estimation Module — FormFlex (Squat Analyzer)
Uses MediaPipe Tasks API (compatible with mediapipe >= 0.10.x on Python 3.14)

Key design decisions:
- Uses direct submodule imports to avoid buggy lazy-loader on newer Python
- Draws full 33-landmark skeleton with colour-coded connections
- Calculates 11 key joint angles using atan2 trigonometry (per project abstract)
"""

import cv2
import mediapipe as mp
import math
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions as mp_BaseOptions

# ─────────────────────────────────────────────
# MediaPipe landmark index reference (33 points)
# ─────────────────────────────────────────────
# 0  = NOSE
# 11 = LEFT_SHOULDER   12 = RIGHT_SHOULDER
# 13 = LEFT_ELBOW      14 = RIGHT_ELBOW
# 15 = LEFT_WRIST      16 = RIGHT_WRIST
# 23 = LEFT_HIP        24 = RIGHT_HIP
# 25 = LEFT_KNEE       26 = RIGHT_KNEE
# 27 = LEFT_ANKLE      28 = RIGHT_ANKLE
# 29 = LEFT_HEEL       30 = RIGHT_HEEL
# 31 = LEFT_FOOT_INDEX 32 = RIGHT_FOOT_INDEX

# Skeleton connections to draw (pairs of landmark indices)
POSE_CONNECTIONS = [
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left Arm
    (11, 13), (13, 15),
    # Right Arm
    (12, 14), (14, 16),
    # Left Leg
    (23, 25), (25, 27), (27, 29), (27, 31),
    # Right Leg
    (24, 26), (26, 28), (28, 30), (28, 32),
    # Face
    (0, 11), (0, 12),
]

# Colour coding by body segment
SEGMENT_COLORS = {
    "torso": (0, 200, 255),    # Cyan
    "arm":   (255, 100, 0),    # Orange
    "leg":   (0, 255, 100),    # Green
    "face":  (200, 200, 200),  # Grey
}

CONNECTION_COLORS = {
    (11, 12): SEGMENT_COLORS["torso"],
    (11, 23): SEGMENT_COLORS["torso"],
    (12, 24): SEGMENT_COLORS["torso"],
    (23, 24): SEGMENT_COLORS["torso"],
    (11, 13): SEGMENT_COLORS["arm"],
    (13, 15): SEGMENT_COLORS["arm"],
    (12, 14): SEGMENT_COLORS["arm"],
    (14, 16): SEGMENT_COLORS["arm"],
    (23, 25): SEGMENT_COLORS["leg"],
    (25, 27): SEGMENT_COLORS["leg"],
    (27, 29): SEGMENT_COLORS["leg"],
    (27, 31): SEGMENT_COLORS["leg"],
    (24, 26): SEGMENT_COLORS["leg"],
    (26, 28): SEGMENT_COLORS["leg"],
    (28, 30): SEGMENT_COLORS["leg"],
    (28, 32): SEGMENT_COLORS["leg"],
    (0, 11):  SEGMENT_COLORS["face"],
    (0, 12):  SEGMENT_COLORS["face"],
}


class PoseDetector:
    def __init__(self, detectionCon=0.5, trackCon=0.5):
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        options = vision.PoseLandmarkerOptions(
            base_options=mp_BaseOptions(model_asset_path='pose_landmarker_lite.task'),
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=float(self.detectionCon),
            min_pose_presence_confidence=float(self.detectionCon),
            min_tracking_confidence=float(self.trackCon)
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.results = None
        self.lmList = []   # [[id, cx, cy], ...]

    # ── Detection ──────────────────────────────────────────────────────────
    def findPose(self, img):
        """Run inference on BGR image, store results. Returns same image."""
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        self.results = self.landmarker.detect(mp_image)
        return img

    # ── Landmark pixel positions ────────────────────────────────────────────
    def findPosition(self, img):
        """Populate self.lmList with pixel coords for all 33 landmarks, plus 3D world coords."""
        self.lmList = []
        self.worldList = []
        if self.results and self.results.pose_landmarks and self.results.pose_world_landmarks:
            h, w, _ = img.shape
            # Parse 2D pixel landmarks
            for idx, lm in enumerate(self.results.pose_landmarks[0]):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([idx, cx, cy])
            # Parse 3D true world coordinates (meters)
            for idx, wlm in enumerate(self.results.pose_world_landmarks[0]):
                self.worldList.append([idx, wlm.x, wlm.y, wlm.z])
        return self.lmList

    # ── Full 33-point skeleton drawing ─────────────────────────────────────
    def drawSkeleton(self, img, landmark_color=(255, 255, 255), dot_radius=5, line_thickness=2):
        """Draw colour-coded skeleton + filled circles on all 33 landmarks."""
        if not self.lmList:
            return img

        lm_dict = {lm[0]: (lm[1], lm[2]) for lm in self.lmList}

        # Draw connections
        for (p1, p2), color in CONNECTION_COLORS.items():
            if p1 in lm_dict and p2 in lm_dict:
                cv2.line(img, lm_dict[p1], lm_dict[p2], color, line_thickness + 1, cv2.LINE_AA)

        # Draw landmark dots
        for idx, cx, cy in self.lmList:
            cv2.circle(img, (cx, cy), dot_radius, (255, 255, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), dot_radius + 1, (0, 0, 0), 1)   # black border

        return img

    # ── Angle calculation (True 3D dot product, robust to camera angle) ────
    def findAngle(self, img, p1, p2, p3, draw=True, label=""):
        """
        Calculates 3D interior angle at vertex p2 between rays p2→p1 and p2→p3
        using MediaPipe pose_world_landmarks. This provides reliable kinematics 
        from ANY video angle (front, side, isometric).
        """
        if not hasattr(self, 'worldList') or len(self.worldList) <= max(p1, p2, p3):
            return 0.0

        # Get 3D real-world coordinates
        v1 = np.array(self.worldList[p1][1:4])
        v2 = np.array(self.worldList[p2][1:4])
        v3 = np.array(self.worldList[p3][1:4])

        # Vectors BA and BC
        ba = v1 - v2
        bc = v3 - v2
        
        # Cosine rule / Dot product
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        
        # Map to specific anatomical bends to simulate 2D flexion/extension logic where needed
        # (Standard 3D angle is always 0-180 absolute spatial angle)
        
        if draw and self.lmList:
            x1, y1 = self.lmList[p1][1:]
            x2, y2 = self.lmList[p2][1:]
            x3, y3 = self.lmList[p3][1:]
            # Draw the two limb segments on screen
            cv2.line(img, (x1, y1), (x2, y2), (200, 200, 0), 2, cv2.LINE_AA)
            cv2.line(img, (x3, y3), (x2, y2), (200, 200, 0), 2, cv2.LINE_AA)
            # Vertex dot
            cv2.circle(img, (x2, y2), 8, (0, 255, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 255, 255), 1)
            # Angle label
            text = f"{label}{int(angle)}°" if label else f"{int(angle)}°"
            cv2.putText(img, text, (x2 + 10, y2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
        
        return float(angle)

    # ── Raw feature vector for ML model ─────────────────────────────────────
    def get_raw_landmarks_features(self):
        """Returns flat list of [x,y,z,visibility] × 33 = 132 floats."""
        features = []
        if self.results and self.results.pose_landmarks:
            for lm in self.results.pose_landmarks[0]:
                v = getattr(lm, 'visibility', getattr(lm, 'presence', 0.0))
                features.extend([lm.x, lm.y, lm.z, float(v)])
        return features
