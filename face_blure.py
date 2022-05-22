import cv2 as cv
import mediapipe as mp
import numpy as np
from scipy.misc import face
from facial_landmarks import FaceLandmarks

# Load face landmarks
fl = FaceLandmarks()

cap = cv.VideoCapture("person_walking.mp4")

while True:
  
  ret, frame = cap.read()
  frame = cv.resize(frame, None, fx = 0.5, fy = 0.5)
  frame_copy = frame.copy()
  height, width, _ = frame.shape

  # 1 Face landmark detection
  landmarks = fl.get_facial_landmarks(frame)
  convexhull = cv.convexHull(landmarks)

  # 2 Face blurring
  mask = np.zeros((height, width), np.uint8)
  cv.polylines(mask ,[convexhull], True, 255, 3)
  cv.fillConvexPoly(mask, convexhull, 255)

  # Extract the Face
  frame_copy = cv.blur(frame_copy, (27, 27))  
  face_extracted = cv.bitwise_and(frame_copy, frame_copy, mask = mask)
  # blureed_frame = cv.GaussianBlur(face_extracted,(27, 27), 0)

  # Extract background
  background_mask = cv.bitwise_not(mask)
  background = cv.bitwise_and(frame, frame, mask= background_mask)

  # Final result
  result = cv.add(background, face_extracted)

  cv.imshow('Result', result)
  cv.imshow('Frame', frame) 
  key = cv.waitKey(30)
  if key == 27:
    break
cap.release()
cv.distroyAllWindows()