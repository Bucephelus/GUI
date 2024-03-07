import cv2
import numpy as np
import time
import math
import serial

# open camera
cap = cv2.VideoCapture(0)

# Initialize serial communication
# ser = serial.Serial('COM3', 9600)  # Replace it with your serial port
# time.sleep(2)

# itialize target location
target_center = None
# new_target_cente = None

# itialize the ball’s previous position and timestamp
prev_center = None
prev_time = time.time()

i=0
grid_size = 16
turning_points = [(100,100), (200,200), (300,300), (400,400), (500,500)]

def detect_ball(frame):
    """Detect the position of the ball"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            if radius > 5:
                return int(x), int(y)
    return None

def has_reached_target(ball_pos, target_pos, box_size, next_target_pos, i):
    # 计算方框的边界
    box_half_size = box_size / 2
    left = target_pos[0] - box_half_size
    right = target_pos[0] + box_half_size
    top = target_pos[1] - box_half_size
    bottom = target_pos[1] + box_half_size

    # 判断小球是否在方框内
    if left <= ball_pos[0] <= right and top <= ball_pos[1] <= bottom:
        # 小球到达方框内，更新目标点为下一个目标点
        # target_pos = next_target_pos
        i = i+1
        return True, next_target_pos, i
    else:
        # 小球未到达方框内，保持当前目标点不变
        return False, target_pos, i
    
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # current_center = detect_ball(frame)
    current_center = (100,100)
    target_center = turning_points[i]
    next_target_center = turning_points[i+1]

    if current_center is not None:
        x, y = current_center

        reached, new_target_center, i = has_reached_target(current_center, target_center, grid_size, next_target_center, i)

        # Calculate the difference between current position and target position
        dx = new_target_center[0] - x
        dy = new_target_center[1] - y

        # Send ball position and target direction to Arduino
        # ser.write(f"{dx},{dy}\n".encode())
        print(dx,dy,new_target_center,next_target_center)
        # time.sleep(0.05)

     # show location
        # cv2.putText(frame, f"Position: {current_center}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # prev_center = current_center

    # display frame
    cv2.imshow("Frame", frame)

    # Detect keyboard keys
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()
# ser.close()
