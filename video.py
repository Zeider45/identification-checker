import cv2
import pytesseract
import re

square = 100
doc = 0


cap = cv2.VideoCapture(1)
cap.set(3,1280)
cap.set(4,720)


def texto(image):
    global doc
    # pytesseract direction
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # gray

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #filter

    umbral = cv2.adaptiveThreshold(gray,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,55,25)

    #OCR Configuration

    config = '--psm 1'
    text = pytesseract.image_to_string(umbral,config=config)

    #key word

    key_word = r'VENEZUELA'
    key_word2 = r'IDENTIFICACION'

    search = re.findall(key_word,text)
    search2 = re.findall(key_word2,text)

    if len(search)!=0 and len(search2)!=0:
        doc = 1


    print(text)

while True:
    # Lectura de VideoCapture
    ret, frame = cap.read()
    # read our keyboard

    #Interface
    cv2.putText(frame, 'Place the identification document', (458,80), cv2.FONT_HERSHEY_SIMPLEX, 0.71, (0,255,0), 2)
    cv2.rectangle(frame, (square,square), (1280 - square, 720 - square), (0,255,0), 2)

    #Opcions

    if doc == 0:
        cv2.putText(frame, 'Press S to identify', (458, 750), cv2.FONT_HERSHEY_SIMPLEX, 0.71, (0, 255, 0),
                    2)

    t = cv2.waitKey(5)
    cv2.imshow('Identification', frame)

    if t == 27:
        break
    elif t == 83 or t == 115:
        texto(frame)

cap.release()
cv2.destroyAllWindows()