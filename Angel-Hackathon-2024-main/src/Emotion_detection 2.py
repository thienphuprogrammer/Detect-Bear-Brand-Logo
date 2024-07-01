import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
import copy

face_classifier = cv2.CascadeClassifier(r"../model/haarcascade_frontalface_default.xml")

model_json_file = "../model/model.json"
model_weights_file = "../model/Latest_Model.h5"
with open(model_json_file, "r") as json_file:
    loaded_model_json = json_file.read()
    classifier = model_from_json(loaded_model_json)
    classifier.load_weights(model_weights_file)

# Read the image using cv2.imread
frame = cv2.imread("./images.jpg")
if frame is None:
    print("Error: Could not read the image.")
else:
    while True:
        img = copy.deepcopy(frame)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            fc = gray[y:y + h, x:x + w]

            roi = cv2.resize(fc, (48, 48))
            pred = classifier.predict(roi[np.newaxis, :, :, np.newaxis])
            text_idx = np.argmax(pred)
            text_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
            text = text_list[text_idx]

            cv2.putText(img, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow("frame", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
