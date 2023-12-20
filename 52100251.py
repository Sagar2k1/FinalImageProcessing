import cv2
import numpy as np

def red_detect(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define the lower and upper bounds for the red color in RGB
    lower_red = np.array([150, 0, 0])
    upper_red = np.array([255, 50, 50])

    # Create a mask to filter out the red color
    red_mask = cv2.inRange(image_rgb, lower_red, upper_red)

    # Apply the mask to the original image
    result = cv2.bitwise_and(image_rgb, image_rgb, mask=red_mask)

    # Convert the result back to BGR for display
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    return result_bgr

def brighten_image(image):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            avg = int(np.average(image[i][j]))
            # print(image[i][j], avg)
            for k in range(image.shape[2]):
                if image[i][j][k] > avg:
                    image[i][j][k] = min(image[i][j][k] + (image[i][j][k] - avg), 255)
                else:
                    image[i][j][k] = max(image[i][j][k] + (image[i][j][k] - avg), 0)
            # print(image[i][j],'/n')
    return image

def RGB_threshold(image):
    for i in range(3):
        _, image[:,:,i] = cv2.threshold(image[:,:,i],100,255,cv2.THRESH_BINARY)
    return image

def get_predict_area(path):
    # Read and resize image
    image = cv2.imread(path)
    image = cv2.resize(image, (1000, 600))

    # make color brighter
    result = brighten_image(image.copy())

    # get represent color of pixel
    result = RGB_threshold(result)

    # detect red for banning sign
    result = red_detect(result)


    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, result = cv2.threshold(result, 10, 255, cv2.THRESH_BINARY)

    # noise filter
    result = cv2.erode(result, np.ones((5,5), np.uint8), iterations=1)
    result = cv2.dilate(result, np.ones((3,3), np.uint8), iterations=1)

    # contour area
    contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sign_chain = list()
    for contour in contours:
        # Get the bounding rectangle for each contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Draw a rectangle on the original image
        if w*h > 5000:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            sign_chain.append(image[y:y+h, x:x+w, :])

    return (image, sign_chain)

    # cv2.imshow('Original Image', image)
    # cv2.imshow('Local Histogram Equalized Image', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
image, sign_list = get_predict_area('../FinalImageProcessing/8d3242e4-06d1-44aa-8f4b-074fb9977e35/img_2217.jpg')

cv2.imshow('Local Histogram Equalized Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
for i in sign_list:
    cv2.imshow('Local Histogram Equalized Image', i)
    cv2.waitKey(0)
    cv2.destroyAllWindows()