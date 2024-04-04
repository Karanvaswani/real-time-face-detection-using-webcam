import cv2
import logging
import os
import face_recognition
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(r'F:\Downloads\haarcascade_frontalface_default.xml')


# Function to update or create log file with face name and current date-time
def update_or_create_log_file(face_name):
    # Get the current date and time
    now = datetime.now()
    current_datetime = now.strftime("%Y-%m-%d %H:%M:%S")

    # Construct the log message
    log_message = f"Detected: {face_name}\nDate-Time: {current_datetime}\n"

    # Construct the log file name
    log_file_name = f"{face_name}.log"

    # Write the log message to the log file
    with open(log_file_name, 'w') as log_file:
        log_file.write(log_message)


# Load the images of the faces you want to compare
image_paths = [
       "C:\\Users\\DELL\\Desktop\\karan.jpeg"
]
known_face_encodings = []
known_face_names = []

for image_path in image_paths:
    image_of_person = face_recognition.load_image_file(image_path)
    person_face_encoding = face_recognition.face_encodings(image_of_person)[0]
    known_face_encodings.append(person_face_encoding)
    known_face_names.append(os.path.basename(image_path).split('.')[0])

# Open a connection to the default camera (usually camera index 0)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale (required by the classifier)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=1)

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face
        face = frame[y:y + h, x:x + w, :]

        # Perform face recognition
        face_encodings = face_recognition.face_encodings(frame, [(y, x + w, y + h, x)])

        # If a face is detected, compare with known faces
        if len(face_encodings) > 0:
            # Assume only one face per frame for simplicity
            face_encoding = face_encodings[0]
            # Compare with known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            if True in matches:
                # Known face detected, update log file
                match_index = matches.index(True)
                update_or_create_log_file(known_face_names[match_index])
            else:
                # Unknown face detected, create log file
                face_name = f"unknown_face_{x}_{y}"
                update_or_create_log_file(face_name)

        # Draw rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the result
    cv2.imshow('Face Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
