import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from matplotlib.animation import FuncAnimation
from skimage.morphology import skeletonize
from skimage import img_as_ubyte

#Link naar video
video_path = r'C:\Users\jespe\OneDrive\Bureaublad\Wormen\kamertemperatuur1.avi'

#Bijsnijden video(Om andere wormen uit te knippen bijvoorbeeld)
min_height = 300
max_height = 900
min_width = 500
max_width = 1500

#Waarde van contrast
threshold_value = 28
x_values = []
y_values = []
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

             
def resize_frame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return resized_frame    

def adjust_frame(frame, min_height, max_heigth, min_width, max_width):
    return frame[min_height:max_height, min_width:max_width]
def processing(frame):
    red_channel = frame[:,:,2]
    green_channel = frame[:,:,1]
    blue_channel = frame[:,:,0]


    red_adjusted = red_channel.astype(np.float32)  
    red_adjusted = np.clip(red_adjusted, 0, 255).astype(np.uint8)
    red_adjusted = red_adjusted - blue_channel.astype(np.float32)
    red_adjusted = np.clip(red_adjusted, 0, 255).astype(np.uint8)

    red_adjusted = red_adjusted.astype(np.uint8)
    red_adjusted = adjust_frame(red_adjusted, min_height, max_height, min_width, max_width)
    _, red_adjusted = cv2.threshold(red_adjusted, threshold_value, 255, cv2.THRESH_BINARY)


    kernel = np.ones((3,3),np.uint8)
    kernel_close = np.ones((45,45),np.uint8)


    opening = cv2.morphologyEx(red_adjusted, cv2.MORPH_OPEN, kernel = kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_close)
    skeleton = skeletonize(closing)
    skeleton_uint8 = img_as_ubyte(skeleton)
    return skeleton_uint8
for i in range(0, total_frames):
    print(i)
    ret, frame = cap.read()
    new_frame = processing(frame)
    contours, _ = cv2.findContours(new_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        longest_contour = contours[0]
        endpoint = longest_contour[-1][0]  
    
        original_image = cv2.cvtColor(new_frame, cv2.COLOR_GRAY2BGR)  
        x, y = endpoint
        square_size = 10
        cv2.rectangle(original_image, (x-square_size, y-square_size), (x+square_size, y+square_size), (128, 128, 128), -1) 
        print("Endpoint:", endpoint)
    else:
        print("No contours found")
    cv2.imshow('Processed Frames', resize_frame(original_image))

    cv2.waitKey(25)

cv2.destroyAllWindows()



  


        

       

