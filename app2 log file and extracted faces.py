import cv2
import face_recognition
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(r'F:\Downloads\haarcascade_frontalface_default.xml')

# Open a connection to the default camera (usually camera index 0)
cap = cv2.VideoCapture(0)

# Define a dictionary to store seen faces and their areas
seen_faces = {}

while True:
  # Read a frame from the webcam
  ret, frame = cap.read()

  # Convert the frame to grayscale (required by the classifier)
  gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Perform face detection
  faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=1)

  for (x, y, w, h) in faces:
    face_area = w * h  # Calculate area of the bounding box

    # Check if a similar face (based on area) has been seen before
    similar_face_seen = False
    for seen_face_area in seen_faces.values():
      if abs(face_area - seen_face_area) < 500:  # Define a threshold for area difference (adjust as needed)
        similar_face_seen = True
        break

    if not similar_face_seen:  # If it's a new face
      # Save the extracted face and log details
      face = frame[y:y+h, x:x+w, :]
      cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

      face_filename = f"extracted_face_{x}_{y}.png"  # You can customize the filename as needed
      cv2.imwrite(face_filename, face)

      logging.info(f"Extracted face saved as: {face_filename}")

      # Create a separate log file for each extracted face
      log_file_name = f"{face_filename}.log"
      log_file = open(log_file_name, 'w')
      log_file.write(f"Extracted face saved as: {face_filename}\n")
      log_file.close()

      # Update seen_faces dictionary
      seen_faces[face_filename] = face_area

  # Display the result with rectangles
  cv2.imshow('Face Detection', frame)

  # Display the extracted face if at least one face is detected
  if len(faces) > 0:
    cv2.imshow('Extracted Face', face)

  # Break the loop if 'q' key is pressed
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
