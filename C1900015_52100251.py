import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
path = r'8d3242e4-06d1-44aa-8f4b-074fb9977e35/1-20220629-web-bien-bao-cam-do-xe-tai-diem-trong-giu-xe-co-thu-phi-47-083934.jpg'

img = cv2.imread(path)
new_size = (img.shape[1],img.shape[0])

path = r'./jpg_image/'
count=0




red_upper = np.array([255, 255, 255], dtype=np.uint8)
red_lower = np.array([150, 50, 70], dtype=np.uint8)
blue_upper = np.array([128, 255, 255], dtype=np.uint8)
blue_lower = np.array([90, 50, 70], dtype=np.uint8)
black_lower = np.array([1,50,70], dtype=np.uint8)
black_upper = np.array([2, 255, 255], dtype=np.uint8)
lower_white = np.array([90, 50, 70])
upper_white = np.array([128, 255, 255])
for root, dirs, files in os.walk(path):
    for file in files:
        img = cv2.imread(path+file)
        print("read {0}".format(file))
        old_size = (img.shape[1],img.shape[0])
        img = cv2.resize(img,new_size)
        
        # image preprocessing
        img_copy = img.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))
        kernel3 = cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))
        img_copy = cv2.morphologyEx(img_copy,cv2.MORPH_CLOSE,kernel)
        img_copy = cv2.erode(img_copy,kernel,iterations=1)
        img_copy = cv2.dilate(img_copy,kernel3,iterations=2)      
        
        # detection image with hsv_space
        kernel = np.ones((3,3),dtype=np.uint8)
        img_hsv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img_hsv, red_lower, red_upper)
        mask2 = cv2.inRange(img_hsv, black_lower, black_upper)
        mask = mask + mask2
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)
        output1 = cv2.bitwise_and(img_copy, img_copy, mask=mask)
        largestArea = 0
        largestRect = None
        # 
        
        contours = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
        #print(contours.__len__())
        box_list = []
        if(len(contours)>0):
            for cnt in contours:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                sideOne = np.linalg.norm(box[0]-box[1])
                sideTwo = np.linalg.norm(box[0]-box[3])
                box_list.append(box)
                area = sideOne*sideTwo
                if area > largestArea:
                    largestArea = area
                    largestRect = box
                if ( sideTwo/sideOne<1.5 or sideOne /sideTwo<1.5):
                    cv2.drawContours(img,[box],0,(0,255,0),3)            
        else:
            continue
        # draw contours
        
        
        # show image
        '''
        plt.figure(figsize=(12,5))
        plt.subplot(2,1,1)
        plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.subplot(2,1,2)
        plt.imshow(mask)
        plt.axis('off')
        plt.show()
        '''
        # write file
        cv2.imwrite('./output/'+file, img)