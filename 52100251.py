import cv2
import numpy as np

def red_detect(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define the lower and upper bounds for the red color in RGB
    lower_red = np.array([200, 0, 0])
    upper_red = np.array([255, 100, 100])

    # Create a mask to filter out the red color
    red_mask = cv2.inRange(image_rgb, lower_red, upper_red)

    # Apply the mask to the original image
    result = cv2.bitwise_and(image_rgb, image_rgb, mask=red_mask)

    # Convert the result back to BGR for display
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    return result_bgr

def RGB_threshold(image):
    for i in range(3):
        _, image[:,:,i] = cv2.threshold(image[:,:,i],160,255,cv2.THRESH_BINARY)
    return image

def adaptive_histogram(image):
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    return clahe.apply(image)

def adaptive_histogram_for_RGB(image):
    for i in range(3):
        image[:,:,i] = adaptive_histogram(image[:,:,i])
    return image
def local_histogram_equalization(image, tile_size):
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to ensure it can be evenly divided into tiles
    height, width = image.shape
    height_resized = height - (height % tile_size[0])
    width_resized = width - (width % tile_size[1])
    print(height_resized, width_resized)
    image = cv2.resize(image, (width_resized, height_resized))
    # Divide the resized image into tiles
    tiles = [image[x:x + tile_size[0], y:y + tile_size[1]] for x in range(0, height_resized, tile_size[0]) for y in range(0, width_resized, tile_size[1])]
    # Apply histogram equalization to each tile
    equalized_tiles = [cv2.equalizeHist(tile) for tile in tiles]
    # Reconstruct the image from the equalized tiles
    equalized_rows = [np.concatenate(equalized_tiles[i:i + int(width_resized / tile_size[1])], axis=1) for i in range(0, len(equalized_tiles), int(width_resized / tile_size[1]))]
    for i in equalized_rows:
        print(i.shape)
    equalized_image = np.concatenate(equalized_rows, axis=0)
    # equalized_image = np.concatenate([np.concatenate(equalized_tiles[i:i + int(width_resized / tile_size[1])], axis=0) for i in range(0, len(equalized_tiles), int(height_resized / tile_size[0]))], axis=1)

    return equalized_image

# Read an image
image = cv2.imread('D:/Project/ImgProcessing/FinalImageProcessing/8d3242e4-06d1-44aa-8f4b-074fb9977e35/3sCBWxeq.jpg')
# Set the tile size (adjust as needed)
tile_size = (50, 50)

result = cv2.resize(image.copy(), (600,600))
result = adaptive_histogram_for_RGB(result)
result = RGB_threshold(result)
result = red_detect(result)
# result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
# result = cv2.Canny(result, 100, 200)
# circles = cv2.HoughCircles(result, cv2.HOUGH_GRADIENT, 1, 80,
#                                param1=150, param2=35,
#                                minRadius=40, maxRadius=100)
# if circles is not None:
#     circles = np.uint16(np.around(circles))
#     for i in circles[0, :]:
#         center = (i[0], i[1])
#         # circle center
#         cv2.circle(result, center, 1, (0, 100, 100), 3)
#         # circle outline
#         radius = i[2]
#         cv2.circle(result, center, radius, (255, 0, 255), 3)
cv2.imshow('Original Image', image)
cv2.imshow('Local Histogram Equalized Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
