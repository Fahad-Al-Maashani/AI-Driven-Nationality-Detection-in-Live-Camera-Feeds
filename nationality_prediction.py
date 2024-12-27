import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from face_detection import detect_faces

# Dummy dataset (replace with actual data)
X = np.random.rand(100, 128)  # 100 samples, 128 features each
y = np.random.randint(0, 5, 100)  # 5 possible nationalities

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier
classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)

# Test the classifier
y_pred = classifier.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

def predict_nationality(frame):
    faces, frame_with_faces = detect_faces(frame)
    for (x, y, w, h) in faces:
        # Simulated feature extraction (replace with actual feature extraction logic)
        features = np.random.rand(1, 128)
        nationality = classifier.predict(features)[0]
        print(f'Predicted Nationality: {nationality}')
    return frame_with_faces

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_with_faces = predict_nationality(frame)
        cv2.imshow('Live Camera Feed with Nationality Prediction', frame_with_faces)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
