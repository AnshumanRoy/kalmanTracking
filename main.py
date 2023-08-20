import cv2
import numpy as np
import configparser
from kalman import KalmanFilter

# Read the config.ini file
config = configparser.ConfigParser()
config.read('config.ini')

# Load YOLO
net = cv2.dnn.readNet(config['Paths']['yolo_weights'], config['Paths']['yolo_cfg'])  # Replace with appropriate paths
classes = []
with open(config['Paths']['coco_names'], "r") as f:  # Replace with appropriate path
    classes = f.read().rstrip('\n').split('\n')

# Load video capture
video_path = config['Paths']['video_input']  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Initialize Kalman Filter
dt = int(config['KalmanFilter']['dt'])  # Sampling time
ux = float(config['KalmanFilter']['ux'])  # Acceleration in x-direction
uy = float(config['KalmanFilter']['uy'])  # Acceleration in y-direction
std_acc = float(config['KalmanFilter']['std_acc'])  # Process noise magnitude
x_measuredSD = float(config['KalmanFilter']['x_measuredSD'])  # Standard deviation of measurement in x-direction
y_measuredSD = float(config['KalmanFilter']['y_measuredSD'])  # Standard deviation of measurement in y-direction


kf = KalmanFilter(dt, ux, uy, std_acc, x_measuredSD, y_measuredSD)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Create a blob from the frame to feed to the network
    blob = cv2.dnn.blobFromImage(frame, scalefactor=0.00392, size=(416, 416), swapRB=True, crop=False)

    # Set the input of the network
    net.setInput(blob)

    # Get the output layer names
    output_layers = net.getUnconnectedOutLayersNames()

    # Run forward pass to get detections
    detections = net.forward(output_layers)

    highest_confidence = 0
    detected_class_id = -1
    bounding_box = None

    # Loop through each detection and find the highest confidence detection
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > highest_confidence and classes[class_id] == 'car':
                highest_confidence = confidence
                detected_class_id = class_id
                bounding_box = obj[0:4] * np.array([width, height, width, height])

    if bounding_box is not None:
        x, y, w, h = bounding_box
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)

        # Draw yellow bounding box
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        # cv2.putText(frame, f"Car: {highest_confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        car_x = int((x1 + x2) / 2)
        car_y = int((y1 + y2) / 2)
        car_position = np.matrix([[car_x], [car_y]])

        # Predict using Kalman Filter
        predicted_position = kf.predict()

        # Update Kalman Filter with detected car position
        estimated_position = kf.update(car_position)

        # Draw circles for actual, estimated, and predicted positions
        cv2.circle(frame, (car_x, car_y), 5, (0, 0, 255), -1)  # Actual position (red)
        cv2.circle(frame, (int(estimated_position[0]), int(estimated_position[1])), 5, (0, 255, 0), -1)  # Estimated position (green)
        cv2.circle(frame, (int(predicted_position[0]), int(predicted_position[1])), 5, (255, 0, 0), -1)  # Predicted position (blue)

    # Display the result
    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()