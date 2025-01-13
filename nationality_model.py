import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from list import label_to_country  # Import the country dictionary
import cv2
from face_detection import detect_faces  # Function to detect faces in a frame

# Dummy dataset for training (replace with actual data)
X = np.random.rand(100, 128)  # 100 samples, 128 features each
y = np.random.randint(0, len(label_to_country), 100)  # Labels corresponding to dictionary keys

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier
classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)

# Test the classifier's accuracy
y_pred = classifier.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

def predict_nationality(frame):
    """
    Predicts nationality for faces detected in the frame.
    Overlays the predicted country name on the frame.
    """
    # Detect faces in the frame
    faces, frame_with_faces = detect_faces(frame)
    
    for (x, y, w, h) in faces:
        # Simulate feature extraction (replace with actual logic)
        features = np.random.rand(1, 128)
        
        # Predict the nationality label
        predicted_label = classifier.predict(features)[0]
        
        # Map the label to a country name
        predicted_country = label_to_country.get(predicted_label, "Unknown Country")
        
        # Display the predicted country name above the face
        text = predicted_country
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (0, 255, 0)  # Green
        thickness = 2
        text_x, text_y = x, y - 10  # Position the text slightly above the face
        cv2.putText(frame_with_faces, text, (text_x, text_y), font, font_scale, font_color, thickness)
        
    return frame_with_faces

if __name__ == "__main__":
    # Open the camera feed
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()  # Capture a frame
        if not ret:
            break
        
        # Predict and display nationalities
        frame_with_faces = predict_nationality(frame)
        
        # Show the frame in a window
        cv2.imshow('Live Camera Feed with Nationality Prediction', frame_with_faces)
        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

