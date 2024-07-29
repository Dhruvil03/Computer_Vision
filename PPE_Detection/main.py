
from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import math
from sort import *

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
tracker1 = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
model = YOLO("best.pt")
model1 = YOLO("../Yolo-weights/yolov5x6u.pt")
cap = cv2.VideoCapture("2.jpeg")
# img = cv2.imread('2.jpeg')

totalCounts = []
totalCar = []
className = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                          "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                          "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                          "umbrella",
                          "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                          "baseball bat",
                          "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                          "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                          "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                          "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                          "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                          "teddy bear", "hair drier", "toothbrush"]
# It's important for YOLO; convert BGR (OpenCV default) to RGB
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
while True:
    success, img = cap.read()  # read the camera frame
    detections = np.empty((0, 5))
    detect = np.empty((0, 5))
    if not success:
        break
    else:
        results = model(source=img, stream=True)
        re = model1(source=img, stream=True)
        classNames = ["Helmet", "Mask", "Safety_Jacket"]
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
                w, h = x2 - x1, y2 - y1

                conf = math.ceil((box.conf * 100)) / 100  # Assuming this accesses the tensor correctly
                cls = int(box.cls)  # Assuming this accesses the tensor correctly
                currentClass = classNames[cls]

                if currentClass == "Helmet" and conf > 0.1:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 1)
                    cvzone.putTextRect(img, f'{currentClass}', (max(0, x1), max(35, y1)),
                                       scale=1, thickness=1, offset=1)
                    # cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=9, rt=5)

        for r in re:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
                w, h = x2 - x1, y2 - y1

                conf = math.ceil((box.conf * 100)) / 100  # Assuming this accesses the tensor correctly
                cls = int(box.cls)  # Assuming this accesses the tensor correctly
                currentClass = className[cls]

                if currentClass == "car" and conf > 0.1:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

                    cvzone.putTextRect(img, f'{currentClass}', (max(0, x1), max(35, y1)),
                                       scale=1, thickness=1, offset=1)
                    # cv2.putText(img, f'{currentClass}', (max(0, x1), max(35, y1)), cv2.FONT_HERSHEY_SIMPLEX,
                    #             scale=1, color, thickness=1, cv2.LINE_AA)
                    # cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=9, rt=5)
                    currentArr = np.array([x1, y1, x2, y2, conf])
                    detect = np.vstack((detect, currentArr))

                if currentClass == "person" and conf > 0.3:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                    #                    scale=0.6, thickness=1, offset=3)
                    # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))
        resultsTracker = tracker.update(detections)
        resultsTracking = tracker.update(detect)

        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(result)
            w, h = x2 - x1, y2 - y1
            # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))
            # cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)
            if totalCounts.count(id) == 0:
                totalCounts.append(int(id))
            else:
                if totalCounts.count(id) == 1:
                    totalCounts.remove(int(id))
            for r in resultsTracking:
                x1, y1, x2, y2, id = r
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                print(r)
                w, h = x2 - x1, y2 - y1
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))
                # cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)
                if totalCar.count(id) == 0:
                    totalCar.append(int(id))
                else:
                    if totalCar.count(id) == 1:
                        totalCar.remove(int(id))
            # cx, cy = (x1 + x2) / 2, y2
            # cx = math.ceil(cx)
            # cy = math.ceil(cy)
            # cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            #
            # # Entry from upper side
            # if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] < cy < limitsUp[1] + 150:
            #     if totalCounts.count(id) == 0:
            #         totalCounts.append(int(id))
            # else:
            #     if totalCounts.count(id) == 1:
            #         totalCounts.remove(int(id))
            #
            # cv2.putText(img, str(totalCounts), (37, 170), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 3)

        cv2.putText(img, "Total person count:" + str(len(totalCounts)), (37, 80), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 2)
        cv2.putText(img, "Person without helmet:1", (37, 120), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 2)
        cv2.putText(img, "Car count:" + str(len(totalCar)), (37, 160), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(0)

