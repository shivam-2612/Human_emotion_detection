import cv2
import numpy as np
from keras.models import model_from_json, Sequential

# Load model architecture and weights
# with open("facialemotionmodel.json", "r") as json_file:
#     model_json = json_file.read()

# model = model_from_json(model_json, custom_objects={"Sequential": Sequential})
# model.load_weights("facialemotionmodel.h5")
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")


haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

labels = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4:'neutral', 5:'sad', 6:'surprise'}

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature / 255.0

webcam = cv2.VideoCapture(0)

while True:
    ret, im = webcam.read()
    if not ret:
        continue

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        try:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img = extract_features(roi_gray)

            pred = model.predict(img, verbose=0)
            prediction_label = labels[pred.argmax()]

            cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(im, prediction_label, (x, y - 10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        except Exception as e:
            print("Prediction error:", e)
            continue

    cv2.imshow("Facial Emotion Detection", im)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
