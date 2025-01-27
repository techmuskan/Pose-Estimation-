import cv2
import numpy as np
import time  # Added missing import

def draw_fps(frame, fps):
    """Draw FPS counter on frame."""
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )
    return frame

def apply_futuristic_overlay(frame):
    """Apply futuristic visual effects to the frame."""
    # Add a subtle blue tint
    blue_tint = frame.copy()
    blue_tint[:, :, 0] = blue_tint[:, :, 0] * 1.2  # Increase blue channel
    frame = cv2.addWeighted(frame, 0.8, blue_tint, 0.2, 0)

    # Add scanner line effect
    height, width = frame.shape[:2]
    scanner_pos = int((time.time() * 100) % height)
    cv2.line(frame, (0, scanner_pos), (width, scanner_pos), (0, 255, 255), 1)

    # Add grid overlay
    grid_size = 50
    alpha = 0.1

    overlay = frame.copy()
    for x in range(0, width, grid_size):
        cv2.line(overlay, (x, 0), (x, height), (0, 255, 255), 1)
    for y in range(0, height, grid_size):
        cv2.line(overlay, (0, y), (width, y), (0, 255, 255), 1)

    frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

    return frame