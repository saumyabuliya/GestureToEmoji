import cv2
import os
import numpy as np

image_x, image_y = 50, 50
def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)


def store_images(gesture_id):
    cap = cv2.VideoCapture(0)
    total_pics = 1200
    x, y, w, h = 300, 50, 350, 350
    create_folder("gestures\\" + str(gesture_id))
    pic_no = 0
    flag_start_capturing = False
    frames = 0
    frame = 0
    while True:
        ret1, frame = cap.read()
        if ret1:
            frame = cv2.flip(frame, 1)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask2 = cv2.inRange(hsv, np.array([2,50,60]), np.array([25, 150, 255]))
            res = cv2.bitwise_and(frame, frame, mask = mask2)
            gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            median = cv2.GaussianBlur(gray, (5,5), 0)

            kernal_square = np.ones((5,5), np.uint8)
            dilation = cv2.dilate(median, kernal_square, iterations = 2)
            opening = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernal_square)

            ret, thresh = cv2.threshold(opening, 30, 255, cv2.THRESH_BINARY)
            thresh = thresh[y:y + h, x:x+w]
            contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
            if len(contours) > 0:
                contour = max(contours, key=cv2.contourArea)
               # print(cv2.contourArea(contour))
                if cv2.contourArea(contour) > 10000 and frames > 50:
                    x1, y1, w1, h1 = cv2.boundingRect(contour)
                    pic_no += 1
                    save_img = thresh[y1:y1+h1, x1:x1+w1]
                    save_img = cv2.resize(save_img, (image_x, image_y))
                    cv2.putText(frame, "Capturing....", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (127,255,255))
                    cv2.imwrite("gestures/" + str(gesture_id) + "/" + str(pic_no) + ".jpg", save_img)

                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(frame, str(pic_no), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
                cv2.imshow("Capturing gesture", frame)
                cv2.imshow("thresh", thresh)
                keypress = cv2.waitKey(1)
                print(flag_start_capturing)
                if keypress == ord('c'):
                    if flag_start_capturing == False:
                        flag_start_capturing = True
                    else:
                        flag_start_capturing = False
                        frames = 0
                if flag_start_capturing == True:
                    print(frames)
                    frames += 1
                if pic_no == total_pics:
                    break
    cv2.destroyAllWindows()
    cap.release()


for i in range(6, 11):
    store_images(i)