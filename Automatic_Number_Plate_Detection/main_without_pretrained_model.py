import cv2
import numpy as np
import easyocr


def detect_and_read_plate(img_path):
    # Create an OCR reader - this can be done once and reused for multiple images
    reader = easyocr.Reader(['en'], gpu=False)  # Set 'gpu=True' if you want to use GPU

    # Read the image file
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Image could not be read.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edged = cv2.Canny(blurred, 30, 200)

    # Find contours and sort them by size
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # Screen for potential number plates
    candidate_number_plate = None
    for contour in contours:
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # Contours with 4 points can potentially be the number plate
        if len(approx) == 4:
            candidate_number_plate = approx
            break

    if candidate_number_plate is not None:
        # Extract the number plate from the image
        x, y, w, h = cv2.boundingRect(candidate_number_plate)
        plate_img = img[y:y + h, x:x + w]

        # Use EasyOCR to read the text from the number plate
        result = reader.readtext(plate_img)
        plate_text = ' '.join([text[1] for text in result])
        print("Detected Number Plate:", plate_text.strip())

        # Draw the contour and text of the number plate detected
        cv2.drawContours(img, [candidate_number_plate], -1, (0, 255, 0), 3)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, plate_text.strip(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow("Number Plate Detection and Recognition", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No number plate detected.")


# Provide the path to the image
detect_and_read_plate('car1.jpg')
