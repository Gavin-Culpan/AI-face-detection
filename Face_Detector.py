import cv2
from random import randrange


# Load pre-trained data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose img
#img = cv2.imread('RDJ.png')

# To capture on webcam
webcam = cv2.VideoCapture(0)

# While loop over frames
while True: 

        # Read the current frame
        successful_frame_read, frame = webcam.read()

        # Convert to grayscale
        grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

        # Draw rectangles around faces
        for (x, y, w, h) in face_coordinates:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)

        cv2.imshow('Face Detector', frame)
        key = cv2.waitKey(1)

        #### Stop video if Q is pressed
        if key==81 or key==113:
            break
       

print("Code Completed")
