import cv2
import numpy as np
import os
from keras.models import load_model

model1 = load_model('emojiClassifier.h5')
print(model1)

#mapping the output values to the types of emojis
dictionary = {1 : 'Open Plam',
              2 : 'thumbs up',
              3 : 'Dog',
              4 : 'Fist up',
              5 : 'One finger',
              6 : 'Peace',
              7 : 'OK',
              8 : 'arrow finger',
              9 : 'YO',
              10 : 'thumb and little'}

def keras_process_image(img):
    image_x = 50
    image_y = 50
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img

def keras_predict(model, image):
    processed = keras_process_image(image)
    print("processed: " + str(processed.shape))
    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class

cap = cv2.VideoCapture(0)
x, y, w, h = 300, 50, 350, 350

while(cap.isOpened()):
    ret, img = cap.read()
    if ret:
        img = cv2.flip(img, 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask2 = cv2.inRange(hsv, np.array([2,50,60]), np.array([25, 150, 255]))
        res = cv2.bitwise_and(img ,img, mask = mask2)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        median = cv2.GaussianBlur(gray, (5,5), 0)

        kernal_square = np.ones((5,5), np.uint8)
        dilation = cv2.dilate(median, kernal_square,iterations=2)
        opening = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernal_square)
        ret, thresh = cv2.threshold(opening,30, 255,cv2.THRESH_BINARY)

        thresh =  thresh[y:y+h, x:x+w]
        contours = cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[0]

        if len(contours) > 0:
            contour = max(contours, key = cv2.contourArea)
            if cv2.contourArea(contour) > 2500:
                x,y,w1,h1 = cv2.boundingRect(contour)
                newImage = thresh[y:y+h1, x:x+w1]
                newImage = cv2.resize(newImage, (50, 50))
                pred_probab, pred_class = keras_predict(model1, newImage)
                print(dictionary[pred_class])
        x ,y ,w ,h = 300, 50, 350, 350
        cv2.imshow("frame", img)
        cv2.imshow("Contours", thresh)
        k = cv2.waitKey(10)
        if k == 27:
            break

