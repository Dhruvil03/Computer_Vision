# import required libraries
import cv2
import numpy as np
import easyocr

# Read input image
cap = cv2.VideoCapture("license_plate.mp4") # for video
# cap = cv2.imread("car.jpg")  #for image

reader = easyocr.Reader(['en'])
while True:
    success, img = cap.read()

    # convert input image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # read haarcascade for number plate detection
    cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number (1).xml')

    # Detect license number plates
    plates = cascade.detectMultiScale(gray, 1.2, 5)
    print('Number of detected license plates:', len(plates))

    # loop over all plates
    for (x, y, w, h) in plates:
        # draw bounding rectangle around the license number plate
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        gray_plates = gray[y:y + h, x:x + w]
        color_plates = img[y:y + h, x:x + w]
        # reader = easyocr.Reader(['en'])
        gray_plate_final = cv2.cvtColor(gray_plates, cv2.COLOR_BGR2GRAY)
        #_, license_plate_threshold = cv2.threshold(gray_plate_final, 1, 255, cv2.THRESH_BINARY_INV)
        output = reader.readtext(gray_plate_final)
        # print(output)
        for out in output:
            txt_bbox, text, txt_score = out
            if txt_score > 0.01:
                cv2.putText(img,text.upper(),(x-5,y-5),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

        # save number plate detected
        cv2.imwrite('Numberplate.jpg', gray_plate_final)
        cv2.imshow('Number Plate', gray_plate_final)
        cv2.imshow('Number Plate Image', img)
        cv2.waitKey(1)
