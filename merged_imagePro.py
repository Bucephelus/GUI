
import cv2 #used 
import numpy as np #used 
import matplotlib.pyplot as plt #used 
import networkx as nx #used 
from PIL import Image
import math
# from matplotlib.path import Path


def detect_red_corners(image):
    # Convert the image from BGR to HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for red color in HSV
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red pixels
    red_mask1 = cv2.inRange(image_hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(image_hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Find contours in the red mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]

    # Get the centroids of the valid contours
    centroids = []
    for cnt in valid_contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))

    return centroids

def sort_corners(corners):
    # Sort the corners: top-left, top-right, bottom-right, bottom-left
    corners = sorted(corners, key=lambda x: (x[1], x[0]))
    top_corners = sorted(corners[:2], key=lambda x: x[0])
    bottom_corners = sorted(corners[2:], key=lambda x: x[0], reverse=True)
    return top_corners + bottom_corners

def interpolate_points(start, end):
    """Generate points between start and end using line equation"""
    points = []
    x0, y0 = start
    x1, y1 = end
    dx = x1 - x0
    dy = y1 - y0
    steps = max(abs(dx), abs(dy))
    if steps == 0:
        return [start]  # Start and end are the same
    x_inc = dx / steps
    y_inc = dy / steps
    x = x0
    y = y0
    for _ in range(steps + 1):
        points.append((int(round(x)), int(round(y))))
        x += x_inc
        y += y_inc
    return points

def detect_maze_obstacles(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Detect Trap Holes
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=31,
        minRadius=17,
        maxRadius=27
    )

    obstacle_list = [] # Circles + Frame coordinates
    wall_list = []
    frame_list = []

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            for i in range(y - r, y + r + 1):
                for j in range(x - r, x + r + 1):
                    obstacle_list.append((j, i))

    # Detect Walls
    blurred_walls = cv2.GaussianBlur(image, (5, 5), 80)
    gray_walls = cv2.cvtColor(blurred_walls, cv2.COLOR_BGR2GRAY)
    _, thresh_walls = cv2.threshold(gray_walls, 40.5, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh_walls, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    trim_size = 10
    valid_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if x > trim_size and y > trim_size and (x+w) < (gray_walls.shape[1]-trim_size) and (y+h) < (gray_walls.shape[0]-trim_size):
            valid_contours.append(contour)

    # Create an empty mask with the same dimensions as the input image
    mask = np.zeros_like(gray_walls)

    # Fill the mask with white where the contours are
    cv2.fillPoly(mask, valid_contours, 255)

    # Iterate over the mask and add coordinates to wall_list where the mask is white
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x] == 255:  # Check if the pixel is part of a wall
                wall_list.append((x, y))

    # Detect the frame of the maze
    red_corners = detect_red_corners(image)
    sorted_corners = sort_corners(red_corners)

# Draw lines to form a rectangle (frame of the maze) and store all coordinates
    for i in range(4):
        cv2.line(image, sorted_corners[i], sorted_corners[(i+1)%4], (255, 0, 0), 3)

        # Interpolate points between corners and add to the lists
        frame_line_points = interpolate_points(sorted_corners[i], sorted_corners[(i+1)%4])
        frame_list.extend(frame_line_points)

    # Display the result
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Detected Obstacles')
    plt.show()

    return obstacle_list, wall_list, frame_list

def detect_green_points(image):
    # Convert the image from BGR to HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for green color in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Create a mask for green pixels
    green_mask = cv2.inRange(image_hsv, lower_green, upper_green)

    # Find contours in the green mask
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]  # Adjust this threshold 100

    # Get the centroids of the valid contours
    centroids = []
    for cnt in valid_contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))

    return centroids

# Load the maze image
immaze = cv2.imread("maze_red2.jpeg")  # Replace with the actual path to your maze image
image_path = 'maze_red2.jpeg'

# Resize the image using PIL and save it
image = Image.open(image_path)

# Convert image to array
maze_array = np.array(image)

# Determine the size of the image
width, height = image.size

# calls function to detect obstacles 
obstacle_list, wall_list, frame_list = detect_maze_obstacles(image_path)
# Iterate through the obstacle list and display the obstacles
for obstacle in obstacle_list:
    print(obstacle)

# Detect green points in the resized image
green_points = detect_green_points(immaze)

# Handle the case where not enough green points are detected
if len(green_points) < 2:
    raise ValueError("Not enough green points detected to determine start and end")

# Find the start and end points
start_point = tuple(map(int, min(green_points, key=lambda x: x[0])))
end_point = tuple(map(int, max(green_points, key=lambda x: x[0])))

# Define the grid size - we'll aim for around 50 cells across the width
grid_size = 16 #width // 50
print(grid_size)
# Create a copy of the image to draw the grid on
maze_with_grid = maze_array.copy()

# Draw the grid lines
for x in range(0, width, grid_size):
    cv2.line(maze_with_grid, (x, 0), (x, height), color=(255, 0, 0), thickness=1)
for y in range(0, height, grid_size):
    cv2.line(maze_with_grid, (0, y), (width, y), color=(255, 0, 0), thickness=1)

# Convert array back to Image for display
maze_with_grid_image = Image.fromarray(maze_with_grid)

# Display the image with the grid
plt.imshow(maze_with_grid_image)
plt.axis('on')
plt.title('Maze Image with Grid')
plt.show()

# Save the image with the grid
output_path = 'image_grid.jpeg'
maze_with_grid_image.save(output_path)

maze_with_grid = cv2.imread(output_path)

# Initialize an empty list to store the grid coordinates
grid_coords = []

# Calculate the center of each square in the grid
for y in range(grid_size // 2, height, grid_size):
    for x in range(grid_size // 2, width, grid_size):
        # Append the (x, y) coordinates to the grid_coords list
        grid_coords.append((x, y))

# Display the length of the grid_coords list to verify
print(len(grid_coords), grid_coords[:10] ) # Show the first 10 entries for brevity

# Create a copy of the image to draw the points on
maze_with_points = maze_with_grid.copy()

# Iterate through the grid_coords list and draw each point
for coord in grid_coords:
    cv2.circle(maze_with_points, coord, radius=1, color=(0, 255, 0), thickness=-1)

# Convert array back to Image for display
maze_with_points_image = Image.fromarray(maze_with_points)

# Display the image with the points
plt.imshow(maze_with_points_image)
plt.axis('on') # to include axis with ticks
plt.title('Maze Image with Grid Points')
plt.show()



########################################### OBSTACLE: CIRCLES ###########################################
# REMOVES OBSTACLE COORDINATES FROM GRID_COORDINATES

d0 = 10.5  # Example value, adjust as needed.

# Function to check if an obstacle is within a small square around a grid coordinate
def is_obstacle_near(coord, obstacle_list, distance):
    x, y = coord
    # Define the square boundaries around the grid coordinate
    left = x - distance
    right = x + distance
    top = y + distance
    bottom = y - distance
    
    # Check each obstacle to see if it falls within the square boundaries
    for obstacle in obstacle_list:
        ox, oy = obstacle
        if left <= ox <= right and bottom <= oy <= top:
            return True
    return False

# Function to remove grid coordinates that are too close to obstacles
def remove_close_coords(grid_coords, obstacle_list, d0):
    # Filter out coordinates that are too close to any obstacle
    return [coord for coord in grid_coords if not is_obstacle_near(coord, obstacle_list, d0)]

########################################### WALL LIST ###########################################

d1 = 8.5 # Example value, adjust as needed.

# Function to check if an obstacle is within a small square around a grid coordinate
def is_wall_near(coord, wall_list, distance):
    x, y = coord
    # Define the square boundaries around the grid coordinate
    left = x - distance
    right = x + distance
    top = y + distance
    bottom = y - distance
    
    # Check each obstacle to see if it falls within the square boundaries
    for wall in wall_list:
        ox, oy = wall
        if left <= ox <= right and bottom <= oy <= top:
            return True
    return False

# Function to remove grid coordinates that are too close to obstacles
def remove_wall_coords(grid_coords, wall_list, d1):
    # Filter out coordinates that are too close to any obstacle
    return [coord for coord in grid_coords if not is_wall_near(coord, wall_list, d1)]

########################################### FRAME LIST ###########################################

d2 = 9 # Example value, adjust as needed.

# Function to check if an obstacle is within a small square around a grid coordinate
def is_frame_near(coord, frame_list, distance):
    x, y = coord
    # Define the square boundaries around the grid coordinate
    left = x - distance
    right = x + distance
    top = y + distance
    bottom = y - distance
    
    # Check each obstacle to see if it falls within the square boundaries
    for frame in frame_list:
        ox, oy = frame
        if left <= ox <= right and bottom <= oy <= top:
            return True
    return False

# Function to remove grid coordinates that are too close to obstacles
def remove_frame_coords(grid_coords, frame_list, d2):
    # Filter out coordinates that are too close to any obstacle
    return [coord for coord in grid_coords if not is_wall_near(coord, frame_list, d2)]

##################################################################################################

# Apply the function to filter the grid coordinates
grid_coords = remove_close_coords(grid_coords, obstacle_list, d0)
grid_coords = remove_wall_coords(grid_coords, wall_list, d1)
grid_coords = remove_wall_coords(grid_coords, frame_list, d2)

# Add the coordinates from green_points to grid_coords
grid_coords.extend(green_points)

##################################################################################################

# Create a graph using networkx
G = nx.Graph()

# Add nodes and edges for each green point
for coord in grid_coords:
    G.add_node(coord)
    for other_coord in grid_coords:
        if coord != other_coord:
            dist = np.linalg.norm(np.array(coord) - np.array(other_coord))
            if dist < 25:  # Adjust the threshold as needed 500, 700, 24
                G.add_edge(coord, other_coord)


#############################################################################################################################
################################################# DISPLAYS ADDED NODES AND EDGES ############################################

# Modify this code to draw nodes and edges with transparency over the maze
def draw_graph_on_image(image, graph):
    # Create a transparent overlay
    overlay = image.copy()
    output = image.copy()
    
    # Draw the edges
    for edge in graph.edges():
        point1 = edge[0]
        point2 = edge[1]
        cv2.line(overlay, point1, point2, color=(255, 0, 0), thickness=2)
    
    # Draw the nodes
    for node in graph.nodes():
        cv2.circle(overlay, node, radius=5, color=(0, 255, 0), thickness=-1)
    
    # Add the overlay, with transparency, onto the original image
    alpha = 0.5 # Define transparency: 0 is fully transparent, 1 is no transparency
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    
    return output

# Now apply the drawing function
maze_with_graph = draw_graph_on_image(maze_with_points, G)

# Convert the image with the graph back to Image for display
maze_with_graph_image = Image.fromarray(maze_with_graph)

# Display the image with the graph
plt.figure(figsize=(10, 5)) # You might want to adjust the figure size as needed
plt.imshow(maze_with_graph_image)
plt.axis('on') # Include axis with ticks
plt.title('Maze Image with Graph')
plt.show()

############################################################################################################################

# Check if start and end points are connected
if not nx.has_path(G, start_point, end_point):
    raise ValueError("Start and end points are not connected in the graph")

# Find the shortest path using A* algorithm
shortest_path = nx.astar_path(G, start_point, end_point)

# Draw the path on the image
path_image = np.copy(maze_with_grid)
for i in range(len(shortest_path) - 1):
    cv2.line(path_image, shortest_path[i], shortest_path[i + 1], (0, 0, 255), 2)

# Mark the start and end points on the image
cv2.circle(path_image, start_point, 10, (0, 255, 0), -1)
cv2.circle(path_image, end_point, 10, (0, 0, 255), -1)

# Display the image with the detected green points and the shortest path
plt.imshow(cv2.cvtColor(path_image, cv2.COLOR_BGR2RGB))
plt.title('Detected Green Points with Shortest Path')
plt.show()

##############################################################################################################
# STORE COORDINATES OF THE DETECTED PATH:  

def save_path_coordinates(path, filename):
    with open(filename, 'w') as file:
        for point in path:
            file.write(f"{point[0]}, {point[1]}\n")

# Call this function to save the coordinates of the shortest path to the file
save_path_coordinates(shortest_path, 'detected_path_coordinates.txt')

####################################################################################################################

def calculate_angle(p1, p2, p3):
    dx21 = p2[0] - p1[0]
    dy21 = p2[1] - p1[1]
    dx32 = p3[0] - p2[0]
    dy32 = p3[1] - p2[1]
    angle = math.atan2(dy32, dx32) - math.atan2(dy21, dx21)
    return angle * 180 / math.pi

def find_turning_points(path):
    turning_points = []
    for i in range(1, len(path) - 1):
        angle = calculate_angle(path[i-1], path[i], path[i+1])
        if abs(angle) > 10:  
            turning_points.append(path[i])
    return turning_points

# find inflection point
turning_points = find_turning_points(shortest_path)

# Draw a circle for each corner point on the image
for point in turning_points:
    cv2.circle(maze_with_grid, point, radius=5, color=(0, 0, 255), thickness=-1)

# Display the image with marked corners
cv2.imshow('Corner Points', maze_with_grid)
cv2.waitKey(0)
cv2.destroyAllWindows()
