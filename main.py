import os

import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

import datetime

from deepface import DeepFace

yolomodel = YOLO("yolov8l.pt")
box_annotator = sv.BoxAnnotator()

yunet_weights = os.path.join(os.path.dirname(__file__), "yunet_n_640_640.onnx")
yunet_detector = cv2.FaceDetectorYN_create(yunet_weights, "", (0, 0))

def main():
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    while True:
        if (cv2.waitKey(30) == 27):
            break

        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)

        channels = 1 if len(frame.shape) == 2 else frame.shape[2]
        if channels == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if channels == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        current_second = datetime.datetime.now().second
        cnnframe = cv2.resize(frame, (640,360))
        if 0 <= current_second < 20:
            yolo(cnnframe, frame)
        elif 20 <= current_second < 40:
            deepface(cnnframe, frame)
        else:
            height, width, _ = cnnframe.shape
            yunet_detector.setInputSize((width, height))
            yunet(cnnframe, frame)

        cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.imshow("window", frame)

def yunet(cnnframe, frame):
    _, faces = yunet_detector.detect(cnnframe)
    faces = faces if faces is not None else []
    for face in faces:
        box = list(map(int, face[:4]))
        box = box[0]*3, box[1]*3, box[2]*3, box[3]*3
        color = (0, 0, 255)
        thickness = 2
        cv2.rectangle(frame, box, color, thickness, cv2.LINE_AA)
        landmarks = list(map(int, face[4:len(face)-1]))
        landmarks = np.array_split(landmarks, len(landmarks) / 2)
        landmarks = [(int(point[0]*3), int(point[1]*3)) for point in landmarks]
        for landmark in landmarks:
            radius = 5
            thickness = -1
            cv2.circle(frame, landmark, radius, color, thickness, cv2.LINE_AA)
            
        confidence = face[-1]
        confidence = "confidence: {:.2f}".format(confidence)
        position = (box[0], box[1] - 10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 2
        cv2.putText(frame, confidence, position, font, scale, color, thickness, cv2.LINE_AA)


def yolo(cnnframe, frame):
    result = yolomodel(cnnframe, agnostic_nms=True)[0]
    detections = sv.Detections.from_yolov8(result)
    labels = [
        f"class: {class_id}, confidence: {confidence:0.2f}"
        for _, _, confidence, class_id, _
        in detections
    ]
    detections.xyxy = (detections.xyxy * 3).astype(np.float32)
    frame = box_annotator.annotate(
        scene=frame,
        detections=detections,
        labels=labels
    )

def deepface(cnnframe, frame):
    detections = DeepFace.analyze(
        img_path = cnnframe, 
        actions = ['age', 'gender', 'race', 'emotion'],
        enforce_detection=False,
        detector_backend="opencv"
    )
    
    for detection in detections:
        top_left = (detection['region']['x']*3, detection['region']['y']*3)
        bottom_right = (
            detection['region']['x']*3 + detection['region']['w']*3, 
            detection['region']['y']*3 + detection['region']['h']*3)

        cv2.rectangle(frame, top_left, bottom_right, (0,255,0), 2)

        # Define the text and its position
        text = f"{detection['dominant_gender']} {str(detection['age'])} {detection['dominant_race']}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 1

        # Calculate text width & height to place the text background correctly
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

        # Draw background for the text (with some padding)
        padding = 3
        background_top_left = (top_left[0], bottom_right[1])
        background_bottom_right = (top_left[0] + text_width + 2 * padding, bottom_right[1] + 2 * (text_height + padding))
        cv2.rectangle(frame, background_top_left, background_bottom_right, (0,255,0), -1) 

        # Now put the text on top of the background
        text_pos = (top_left[0] + padding, bottom_right[1] + text_height + padding)
        cv2.putText(frame, text, text_pos, font, font_scale, (255, 255, 255), font_thickness)

        # Another line of text
        text2 = detection['dominant_emotion']
        text2_pos = (top_left[0] + padding, bottom_right[1] + 2 * text_height + 2 * padding)
        cv2.putText(frame, text2, text2_pos, font, font_scale, (255, 255, 255), font_thickness)

main()
