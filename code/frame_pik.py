import cv2

video_path = r'D:\officiele_metinen\9_o_4.avi'
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cv2.imwrite('saved_frame.jpg', frame)
