import cv2
import numpy as np

# Load MobileNet-SSD model
prototxt_path = r"C:\Users\suppa\Desktop\coding\data_scince\Computer_Vision\people-counting-moving-up-and-down\deploy.prototxt"
model_path =  r"C:\Users\suppa\Desktop\coding\data_scince\Computer_Vision\people-counting-moving-up-and-down\mobilenet_iter_73000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Define classes (Only detecting "person")
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Open video file
video_path =r"C:\Users\suppa\Desktop\coding\data_scince\Computer_Vision\people-counting-moving-up-and-down\output.avi" #r"C:\Users\suppa\Desktop\coding\data_scince\Computer_Vision\people-counting-moving-up-and-down\test.mp4"  # Change to your video file path
cap = cv2.VideoCapture(video_path)

# Check if video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Initialize counters
up_count = 0
down_count = 0
trackers = []

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop when video ends

    (h, w) = frame.shape[:2]
    line_position = int(h * 0.5)  # Set the middle line dynamically

    # Prepare the frame for MobileNet-SSD
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    new_trackers = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:  # Process only confident detections
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] != "person":
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            center_y = (y1 + y2) // 2

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Person {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            new_trackers.append((center_y, (x1, y1, x2, y2)))

    # Counting logic
    for prev_center, prev_box in trackers:
        for new_center, new_box in new_trackers:
            if abs(prev_center - new_center) < 20:
                if prev_center < line_position and new_center >= line_position:
                    down_count += 1
                elif prev_center > line_position and new_center <= line_position:
                    up_count += 1

    trackers = new_trackers

    # Draw middle counting line
    cv2.line(frame, (0, line_position), (w, line_position), (0, 0, 255), 3)  # Red, thicker line
    cv2.putText(frame, f"Up: {up_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Down: {down_count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show output frame
    cv2.imshow("People Counting", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
