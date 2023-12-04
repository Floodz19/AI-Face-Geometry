import cv2
import dlib
import numpy as np

print('Loading...')

# Face detector [Switched to Dlib facial landmark predictor]
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(r'C:\Users\grant\OneDrive\Desktop\Java repos\AI Face Analyzer draft\AI Face analyzer\Face DATA for AI\shape_predictor_68_face_landmarks.dat')

# Signal that indicates whether the initial head position has been captured or not
initial_position_captured = False
initial_head_position = None # Initialize head
initial_eye_cantal_tilt = None # Initialize tilt

def get_eye_cantal_tilt(landmarks):
    left_eye_outer_corner = landmarks[36]
    right_eye_outer_corner = landmarks[45]

    # Cantal tilt equation
    eye_cantal_tilt = np.arctan2(right_eye_outer_corner[1] - left_eye_outer_corner[1],
                                 right_eye_outer_corner[0] - left_eye_outer_corner[0]) * 180 / np.pi

    return eye_cantal_tilt

def get_face_halves(img, landmarks):
    
    # Left and Right Halves of faces
    left_face_indices = list(range(0, 17)) + list(range(26, 16, -1))
    right_face_indices = list(range(16, 26)) + list(range(26, 16, -1))


    left_half = img[:, :img.shape[1] // 2] # Extract left
    right_half = img[:, img.shape[1] // 2:] # Extract right
    left_landmarks = landmarks[left_face_indices]
    right_landmarks = landmarks[right_face_indices]

    return left_half, right_half, left_landmarks, right_landmarks

# Opens Video Capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Converts frame to grayscale
    faces = face_detector(gray)

    for face in faces:
        # Retrieve facial landmarks
        landmarks = landmark_predictor(gray, face)
        landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()])

        # Capture initial head position
        if not initial_position_captured:
            initial_head_position = landmarks_np
            initial_position_captured = True

        # Create visual landmarks on the frame
        for (x, y) in landmarks_np:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Calls cantal tilt equation
        current_eye_cantal_tilt = get_eye_cantal_tilt(landmarks_np)

        # Calculates relative eye cantal tilt [adjusting for head position]
        if initial_head_position is not None:
            relative_eye_cantal_tilt = get_eye_cantal_tilt(landmarks_np) - get_eye_cantal_tilt(initial_head_position)

            # Displays relative eye cantal tilt on the frame
            cv2.putText(frame, f'Relative Eye Cantal Tilt: {relative_eye_cantal_tilt:.2f}', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        
        left_half, right_half, _, _ = get_face_halves(frame, landmarks_np) # Retrieves facial halves
        diff = cv2.absdiff(left_half, right_half) # Calculate pixel-wise differences in facial halves

        # Calculate the percentage of similar pixels 
        similarity_percentage = 100 - ((np.sum(diff < 20) / diff.size) * 100)
        # (high values indicate higher similarity) (needs work)

        cv2.putText(frame, f'Bilateral Similarity: {similarity_percentage:.2f}%', (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Facial Analysis', np.hstack([frame, np.hstack([left_half, right_half])]))

    # Breaks loop if 'q key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
