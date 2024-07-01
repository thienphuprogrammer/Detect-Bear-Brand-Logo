import cv2
from fuzzywuzzy import fuzz
import streamlit as st
from ultralytics import YOLO
import tempfile
import numpy as np
from tensorflow.keras.models import model_from_json
import easyocr
import supervision as sv
from inference.models.yolo_world.yolo_world import YOLOWorld
from tensorflow.keras.models import load_model

class CountPeopleBeer:
    def __init__(self, model_path_pose, model_path_glass_beer):
        self.model_pose = YOLO(model_path_pose)
        self.model_glass_beer = YOLOWorld(model_id=model_path_glass_beer)
        self.quantities_people = 0
        self.quantities_glass_beer = 0
        self.quantities_people_use_beer = 0

    def detect_pose_human(self, __image):
        results = self.model_pose(__image)
        list_poses = []
        list_human = []
        for result in results:
            keypoints = result.keypoints.xy.tolist()
            boxes = result.boxes.xyxy.tolist()
            self.quantities_people = len(results[0].boxes.cls.tolist())
            for i in range(len(keypoints)):
                left_wrist = keypoints[i][9]
                left_wrist = [int(left_wrist[0]), int(left_wrist[1])]
                right_wrist = keypoints[i][10]
                right_wrist = [int(right_wrist[0]), int(right_wrist[1])]
                list_poses.append([left_wrist, right_wrist])
                list_human.append(boxes[i])
        return list_poses, list_human

    def count_glass_beer(self, image):
        classes = ["beer", "glass"]
        results = self.model_glass_beer.infer(image, text=classes, confidence=0.003)
        detections = sv.Detections.from_inference(results)

        list_positions = [position for position in detections.xyxy]
        list_conf = [confidence for confidence in detections.confidence]
        self.quantities_glass_beer = len(list_positions)
        return list_positions, list_conf

    def count_people_use_beer(self, image):
        # get image size
        height, width, _ = image.shape
        list_poses, list_human = self.detect_pose_human(image)
        list_positions, list_conf = self.count_glass_beer(image)
        print(f'list_poses: {list_poses} list_positions: {list_positions}')
        results_poses = []
        results_positions = []
        results_human = []

        check_added_pose = set()
        check_added_position = set()
        for i in range(len(list_poses)):
            for j in range(len(list_positions)):
                human_x1, human_y1, human_x2, human_y2 = list_human[i]
                bounded_size = (human_x2 - human_x1) // 2

                x1, y1, x2, y2 = list_positions[j]
                left_wrist, right_wrist = list_poses[i]
                rec_pose_left = [left_wrist[0] - bounded_size, left_wrist[0] + bounded_size,
                                 left_wrist[1] - bounded_size, left_wrist[1] + bounded_size]
                rec_pose_right = [right_wrist[0] - bounded_size, right_wrist[0] + bounded_size,
                                  right_wrist[1] - bounded_size, right_wrist[1] + bounded_size]

                if (
                        rec_pose_left[0] <= x1 <= rec_pose_left[1] and rec_pose_left[2] <= y1 <= rec_pose_left[
                    3] and i not in check_added_pose and j not in check_added_position
                ):
                    print(f'pose: {list_poses[i]} position: {list_positions[j]}')
                    check_added_pose.add(i)
                    check_added_position.add(j)
                    results_poses.append(list_poses[i])
                    results_positions.append(list_positions[j])
                    results_human.append(list_human[i])
                elif (
                        rec_pose_right[0] <= x1 <= rec_pose_right[1] and rec_pose_right[2] <= y1 <= rec_pose_right[
                    3] and i not in check_added_pose and j not in check_added_position
                ):
                    print(f'pose: {list_poses[i]} position: {list_positions[j]}')
                    check_added_pose.add(i)
                    check_added_position.add(j)
                    results_poses.append(list_poses[i])
                    results_positions.append(list_positions[j])
                    results_human.append(list_human[i])
                elif (
                        rec_pose_left[0] <= x2 <= rec_pose_left[1] and rec_pose_left[2] <= y2 <= rec_pose_left[
                    3] and i not in check_added_pose and j not in check_added_position
                ):
                    print(f'pose: {list_poses[i]} position: {list_positions[j]}')
                    check_added_pose.add(i)
                    check_added_position.add(j)
                    results_poses.append(list_poses[i])
                    results_positions.append(list_positions[j])
                    results_human.append(list_human[i])
                elif (
                        rec_pose_right[0] <= x2 <= rec_pose_right[1] and rec_pose_right[2] <= y2 <= rec_pose_right[
                    3] and i not in check_added_pose and j not in check_added_position
                ):
                    print(f'pose: {list_poses[i]} position: {list_positions[j]}')
                    check_added_pose.add(i)
                    check_added_position.add(j)
                    results_poses.append(list_poses[i])
                    results_positions.append(list_positions[j])
                    results_human.append(list_human[i])

        self.quantities_people_use_beer = len(results_poses)
        return results_poses, results_positions, results_human

    def get_average_people_use_beer(self, image_path=None):
        if self.quantities_people == 0 or self.quantities_glass_beer == 0:
            self.count_people_use_beer(image_path)

        return (self.quantities_glass_beer + self.quantities_people) / 2

class FaceDetection:
    def __init__(self, path_weight_emotion=None, path_weight_face_detection=None, number_class=8):
        self.path_weight_emotion = path_weight_emotion
        self.path_weight_face_detection = path_weight_face_detection
        self.number_class = number_class
        self.model_emotion = load_model(path_weight_emotion)
        self.model_face_detection = YOLO(path_weight_face_detection)

    def predict_emotion(self, image, x1, y1, x2, y2):
        cropped_img = image[y1:y2, x1:x2]
        cropped_img = cv2.resize(cropped_img, (48, 48))  # Assuming emotion model expects 48x48 input
        cropped_img = np.expand_dims(cropped_img, axis=0)
        cropped_img = np.expand_dims(cropped_img, axis=-1)  # If model expects grayscale images

        prediction = self.model_emotion.predict(cropped_img)
        return prediction

    def detect_face(self, image=None):
        result_return = {
            'face': [],
            'emotion': []
        }
        results = self.model_face_detection.predict(image)
        for result in results:
            cls, conf, xyxy = result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist()

            for i in range(len(cls)):
                if cls[i] == 0:
                    x1, y1, x2, y2 = int(xyxy[i][0]), int(xyxy[i][1]), int(xyxy[i][2]), int(xyxy[i][3])
                    cropped_img = image[y1:y2, x1:x2]
                    cropped_img = cv2.resize(cropped_img, (224, 224))
                    cropped_img = np.expand_dims(cropped_img, axis=0)
                    cropped_img = np.expand_dims(cropped_img, axis=-1)
                    # show image cropped
                    prediction = face_detection_model.model_emotion.predict(cropped_img, batch_size=1)
                    result_return['face'].append([x1, y1, x2, y2])
                    result_return['emotion'].append(prediction)
        return result_return
# Paths to models

logo_detection_model_path = "./model/angelhack_yolov9.pt"
face_classifier_path = "./model/haarcascade_frontalface_default.xml"
model_json_file = "./model/model.json"
model_weights_file = "./model/Latest_Model.h5"
human_detection_model_path = "./model/Human_detection.pt"

# Streamlit page configuration
st.set_page_config(
    page_title="Object Detection using YOLOv9",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.header("Video Detection")

    option = st.selectbox(
        "Choose detection type:",
        ("Emotion detect", "Human", "Quantities Person use beer", "Logo Detection","Object Detection"),
        placeholder="Select detection type..."
    )
    source_vid = st.file_uploader("Choose a file", type=["jpeg", "jpg", "png", "webp", "mp4"])

if option == "Emotion detect" and source_vid is not None:
    try:
        path_weights_emotion = 'model/emotion_model.h5'
        path_weights_face_detection = 'model/yolov8n-face.pt'
        face_detection_model = FaceDetection(path_weights_emotion, path_weights_face_detection, 8)
        st.success("Emotion detection model loaded successfully!")
    except Exception as ex:
        st.error(f"Unable to load model or classifier. Check the specified paths.")
        st.error(ex)

    if st.sidebar.button('Emotion Detect'):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(source_vid.read())
            temp_filename = temp_file.name

            image = cv2.imread(temp_filename)
            if image is not None:
                classes = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sadness', "surprise")
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_detection_model.detect_face(image)
                for i in range(len(faces['face'])):
                    x1, y1, x2, y2 = faces['face'][i]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    emotions = faces['emotion'][i]
                    max_acc = 0
                    emotion_text = ""
                    print(emotions)
                    for j in range(len(emotions[0])):
                        print(emotions[0][j])
                        if emotions[0][j] > max_acc:
                            max_acc = emotions[0][j]
                            emotion_text = classes[j]
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(image, emotion_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                                cv2.LINE_AA)

                st.image(image, caption='Detected Emotions', channels="BGR", use_column_width=True)
            else:
                st.error("Error: Unable to read the uploaded image.")
elif option == "Human" and source_vid is not None:
    try:
        model = YOLO(human_detection_model_path)
        st.success("Model loaded successfully!")
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {logo_detection_model_path}")
        st.error(ex)

    if st.sidebar.button('Logo Detect'):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(source_vid.read())
            temp_filename = temp_file.name

            vid_cap = cv2.VideoCapture(temp_filename)
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    image = cv2.resize(image, (720, int(1080 * (9 / 16))))
                    res = model.predict(image, conf=0.3)
                    result_tensor = res[0].boxes
                    res_plotted = res[0].plot()
                    st_frame.image(res_plotted, caption='Detected Video', channels="BGR", use_column_width=True)
                    st.write(f"Total number of humans detected: **{len(result_tensor)}**")
                else:
                    vid_cap.release()
                    break
elif option == "Quantities Person use beer" and source_vid is not None:
    try:
        model_path_pose = 'model/yolov8l-pose.pt'
        model_path_glass_beer = 'yolo_world/l'

        count_people_beer_model = CountPeopleBeer(model_path_pose, model_path_glass_beer)
        st.success("Model loaded successfully!")
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {logo_detection_model_path}")
        st.error(ex)

    if st.sidebar.button('Quantities Person use beer'):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(source_vid.read())
            temp_filename = temp_file.name

            vid_cap = cv2.VideoCapture(temp_filename)
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    count_people_beer = count_people_beer_model.count_people_use_beer(image)
                    result_tensor = count_people_beer[2]
                    new_image = image.copy()
                    for i in range(len(result_tensor)):
                        x1, y1, x2, y2 = result_tensor[i]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        new_image = cv2.rectangle(new_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    st_frame.image(new_image, caption='Detected Video', channels="BGR", use_column_width=True)
                    st.write(f"Total number of people using beer detected: **{count_people_beer_model.quantities_people_use_beer}**")
                    st.write(f"Average number of people using beer detected: **{count_people_beer_model.get_average_people_use_beer()}**")
                    st.write(f"Total number of people detected: **{count_people_beer_model.quantities_people}**")
                    st.write(f"Total number of glass beer detected: **{count_people_beer_model.quantities_glass_beer}**")

                else:
                    vid_cap.release()
                    break
elif option == "Logo Detection" and source_vid is not None:
    strong_list = ["Tiger", "Heineken", "BiaViet", "Strongbow", "Larue", "Bivina", "Edelweiss"]

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(source_vid.read())
        temp_filename = temp_file.name

    image = cv2.imread(temp_filename)

    if image is not None:
        reader = easyocr.Reader(['ch_sim', 'en'])
        result = reader.readtext(image)

        if st.sidebar.button('Logo Detection'):

            max_score = 0.0
            best_match = ""
            detected_text = ""

            for text in result:
                current_text = text[1]

                for item in strong_list:
                    similarity_score = fuzz.ratio(current_text.lower(), item.lower())
                    if similarity_score > max_score:
                        max_score = similarity_score
                        best_match = item
                        detected_text = current_text

            for text in result:
                if text[1] == detected_text:
                    bbox = text[0]
                    cv2.rectangle(image, (bbox[0][0], bbox[0][1]), (bbox[2][0], bbox[2][1]), (255, 0, 0), 2)
                    break

            st.image(image, caption='Detected Text with Highest Score', channels="BGR", use_column_width=True)

    else:
        st.error("Error: Unable to read the uploaded image.")
elif option == "Object Detection" and source_vid is not None:
    classes = ["carton","keg","bucket","umbrella","poster","standee","billboard","fridge","can","glass bottle"]
    try:
        model_glass_beer = YOLOWorld(model_id="yolo_world/l")
        st.success("Model loaded successfully!")
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {logo_detection_model_path}")
        st.error(ex)

    if st.sidebar.button('Object Detection'):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(source_vid.read())
            temp_filename = temp_file.name

            vid_cap = cv2.VideoCapture(temp_filename)
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    results = model_glass_beer.infer(image, text=classes, confidence=0.003)
                    detections = sv.Detections.from_inference(results)

                    list_positions = [position for position in detections.xyxy]
                    result_tensor = list_positions
                    labels = [classes[class_id] for class_id in detections.class_id]
                    for i in range(len(result_tensor)):
                        x1, y1, x2, y2 = result_tensor[i]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(image, labels[i], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                                cv2.LINE_AA)

                    st_frame.image(image, caption='Detected Video', channels="BGR", use_column_width=True)
                else:
                    vid_cap.release()
                    break