# import cv2 to capture videofeed
import cv2

import numpy as np

# attach camera indexed as 0
camera = cv2.VideoCapture(0)

# setting framewidth and frameheight as 640 X 480
camera.set(3 , 640)
camera.set(4 , 480)

# loading the mountain image
mountain = cv2.imread('mount_everest.jfif')

# resizing the mountain image as 640 X 480
background = cv2.resize(mountain, (640, 480))

while True:

    # read a frame from the attached camera
    status , frame = camera.read()

    # if we got the frame successfully
    if status:

        # flip it
        frame = cv2.flip(frame , 1)

        # converting the image to RGB for easy processing
        frame_rgb = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)

        # creating thresholds
        lower_bound = np.array([100, 100, 100])
        upper_bound = np.array([255, 255, 255])
        mask = cv2.inRange(frame_rgb, lower_bound, upper_bound)
        # thresholding image
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))       
        # inverting the mask
        mask_2 = cv2.bitwise_not(mask)
        # bitwise and operation to extract foreground / person
        res_1 = cv2.bitwise_and(frame, frame, mask=mask_2)
        res_2 = cv2.bitwise_and(background, background, mask=mask)
        # final image
        final_output = cv2.addWeighted(res_1, 1, res_2, 1, 0)
        final_output = cv2.morphologyEx(final_output, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        # show it 
        cv2.imshow('frame' , final_output)

        # wait of 1ms before displaying another frame
        code = cv2.waitKey(1)
        if code  ==  32:
            break

# release the camera and close all opened windows
camera.release()
cv2.destroyAllWindows()
