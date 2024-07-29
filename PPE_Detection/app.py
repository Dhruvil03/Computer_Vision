from flask import Flask, render_template, request, Response
from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import math

app = Flask(__name__)

def gen_frames(video):
    model = YOLO("best.pt")
    cap = cv2.VideoCapture(video)
    while True:
        success, img = cap.read()  # read the camera frame
        if not success:
            break
        else:
            results = model(source=img, stream=True)

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

            cv2.imshow("Image", img)
            cv2.waitKey(1)
        ret, buffer = cv2.imencode('.jpg', img)
        img = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')
@app.route('/')
def index():
    return render_template('index.html')

ALLOWED_EXTENSION = ['mp4']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSION

@app.route("/upload", methods=['POST'])
def upload():
    if 'video' not in request.files:
        return 'No video found'
    video= request.files['video']
    if video.filename == '':
        return 'No video selected'
    if video and allowed_file(video.filename):
        video.save('static/video/'+video.filename)
        #return render_template('preview.html', video_name=video.filename)
        return Response(gen_frames(video.filename), mimetype='multipart/x-mixed-replace; boundary=frame')
    return "Invalid video type"

# @app.route('/preview')
# def video_feed(video):
#     return Response(gen_frames(video), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
