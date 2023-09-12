import os
import datetime

import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

yunet_weights = os.path.join(os.path.dirname(__file__), "yunet_n_640_640.onnx")
yunet_detector = cv2.FaceDetectorYN_create(yunet_weights, "", (0, 0))

yolomodel = YOLO("yolov8l.pt")
box_annotator = sv.BoxAnnotator()
from classes import classes

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

        cnnframe = cv2.resize(frame, (640,360))
        
        yolo(cnnframe, frame)
        
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
        confidence = f"confidence: {int(confidence * 100)}%"
        position = (box[0], box[1] - 10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 2
        cv2.putText(frame, confidence, position, font, scale, color, thickness, cv2.LINE_AA)


def yolo(cnnframe, frame):
    result = yolomodel(cnnframe, agnostic_nms=True)[0]
    detections = sv.Detections.from_yolov8(result)
    labels = [
        f"class: {classes[class_id]}, confidence: {int(confidence * 100)}%"
        for _, _, confidence, class_id, _
        in detections
    ]
    detections.xyxy = (detections.xyxy * 3).astype(np.float32)
    frame = box_annotator.annotate(
        scene=frame,
        detections=detections,
        labels=labels
    )

main()
