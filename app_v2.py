import torch
from preprocessing.preprocessing import find_faces, preprocessing_pipeline
from prediction.prediction import predict_happiness
import cv2

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # First check if 'q' is pressed for quitting
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break

    # Detect faces in the current frame
    images, location = find_faces(frame)

    # Loop through detected faces and check if they are happy
    for i, img in enumerate(images):
        # Predict happiness for each face
        is_happy = predict_happiness(img)
        
        # Get the coordinates of the detected face
        x, y, w, h = location[i]
        
        # If the model predicts happiness, draw a rectangle around the face
        if is_happy:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Blue rectangle for happy faces
            cv2.putText(frame, 'Happy', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)  # "Happy" text
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red rectangle for non-happy faces

    # Display the frame with detected faces and rectangles
    cv2.imshow('Happiness Detection (Press Q to quit)', frame)

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
