import cv2
import numpy as np
import os

path = 'input'
dirs = os.listdir(path)
paths = [os.path.join(path,d) for d in dirs]

img_sample = cv2.imread(paths[40])
size_sample = (int(img_sample.shape[1]*1.5), int(img_sample.shape[0]*1.5))
#img = img_sample.copy()
color = {
    'red': {
        'lower': np.array([70, 60, 70], dtype=np.uint8),
        'upper': np.array([180, 255, 255], dtype=np.uint8)
        },
    'white': {
        'lower': np.array([0, 0, 231], dtype = np.uint8),
        'upper': np.array([180, 18, 255], dtype = np.uint8)
        },
    'gray': {
        'lower': np.array([40, 0, 40], dtype = np.uint8),
        'upper': np.array([110, 18, 230], dtype = np.uint8)
        }
    kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))

    '''
    thr = cv2.erode(thr,kernel,iterations=1)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel2)
    thr = cv2.erode(thr,kernel2,iterations=2)
    '''

    thr = cv2.bitwise_not(thr)
    thr = cv2.Canny(thr, 100, 200)
    thr = cv2.blur(thr, (3, 4))
    rows = thr.shape[0]
    circles = cv2.HoughCircles(thr, cv2.HOUGH_GRADIENT, 1, rows / 8,
                               param1=50, param2=50,
                               minRadius=20, maxRadius=200)
    print(circles)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(thr, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(thr, center, radius, (255, 0, 255), 3)
        
        
    
    '''
    contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    l = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
        if len(approx)>10:
            #cv2.drawContours(img_sample, [approx], 0, (127,255,255),1)
            (x,y,w,h) = cv2.boundingRect(contour)
            flag = 0
            for x1, w1, y1,h1 in l:
                if x in range(x1,x1+w1) or y in range(y1,y1+h1):
                    flag = 1
            if flag==1:
                continue           
            
            max, min = w, h
            if max<min:
                max, min = min, max
            if w > 30 and h > 30:              
                
                #print(min, max)
                cv2.rectangle(img_sample, (x,y), (x+w,y+h), (127, 255, 0), 2)
                l.append([x,x+w,y,y+h])
                
            #print(len(approx))
    '''

    cv2.imshow('d',thr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('output/'+d, img_sample)
    print('save: ', 'output/'+d)
    



'''
img_red = cv2.bitwise_and(img, img, mask=mask)
img = cv2.cvtColor(img_red, cv2.COLOR_BGR2GRAY)
_, thr = cv2.threshold(img,160, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))
thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
'''




'''
#thr = cv2.bitwise_not(thr)

'''


#print(contours)