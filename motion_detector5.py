import cv2
import numpy as np
import time
import math

# open camera
cap = cv2.VideoCapture(0)

# Initialize the ball’s previous position and timestamp
prev_center = None
prev_time = time.time()

def calculate_angle(p1, p2):
    """Calculate the angle between two points"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = math.degrees(math.atan2(dy, dx))
    return angle

def calculate_speed(p1, p2, t1, t2):
    """Calculate the speed between two frames"""
    distance = np.linalg.norm(np.array(p2) - np.array(p1))
    time_diff = t2 - t1
    speed = distance / time_diff if time_diff > 0 else 0
    return speed

while True:
    # Read frame
    ret, frame = cap.read()

    if not ret:
        break

    # Convert frames from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Color filtering
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # current time
    current_time = time.time()

    # Traverse contours
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Adjust minimum area threshold
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            if radius > 5:  # Adjust minimum radius threshold
                current_center = (int(x), int(y))

                # draw center point
                cv2.circle(frame, current_center, int(radius), (0, 255, 0), 2)
                cv2.circle(frame, current_center, 5, (0, 0, 255), -1)

                # If there is a previous position, calculate the angle and velocity
                if prev_center is not None:
                    angle = calculate_angle(prev_center, current_center)
                    speed = calculate_speed(prev_center, current_center, prev_time, current_time)

                    # Display angle and speed
                    cv2.putText(frame, f"Angle: {angle:.2f} degrees", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame, f"Speed: {speed:.2f} px/s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Update location and time
                prev_center = current_center
                prev_time = current_time

                # show location
                cv2.putText(frame, f"Position: {current_center}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # display frame
    cv2.imshow("Frame", frame)

    # Detect keyboard keys
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
