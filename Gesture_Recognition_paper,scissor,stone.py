#We recognize gestures by measuring the distances from the wrist to the fingertips and to the finger PIPs, 
#allowing gesture detection even when the hand is facing backward.
#以 0點(手腕) 與 指尖 和 Finger Pip 的距離來辨識手勢，在手掌不是正面的情況下也能辨識出手勢。

import cv2
import math
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

def normalized_to_pixel_coordinates(normalized_x, normalized_y, image_width, image_height):
    x_px = min(int(normalized_x * image_width), image_width - 1)
    y_px = min(int(normalized_y * image_height), image_height - 1)
    return (x_px, y_px)

def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
with mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue
        
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)

    ## Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            pixelCoordinatesLandmarks = []
            for point in mp_hands.HandLandmark:
                    normalizedLandmark = hand_landmarks.landmark[point]
                    pixelCoordinatesLandmark = normalized_to_pixel_coordinates(
                        normalizedLandmark.x, 
                        normalizedLandmark.y, 
                        frameWidth, 
                        frameHeight)
                    if pixelCoordinatesLandmark:
                        pixelCoordinatesLandmarks.append(pixelCoordinatesLandmark)
                        
        wrist = pixelCoordinatesLandmarks[0]
        tip_ids = [4, 8, 12, 16, 20]
        pip_ids = [2, 6, 10, 14, 18]
        distances_tip = [distance(wrist, pixelCoordinatesLandmarks[i]) for i in tip_ids]
        distances_pip = [distance(wrist, pixelCoordinatesLandmarks[i]) for i in pip_ids]
        far = []
        
        # thumb
        if distances_tip[0] < distances_pip[0]:
            far.append(0)
        else:
            far.append(1)
            
        # other four fingers
        for id in range(1, 5):
            if distances_tip[id] < distances_pip[id]:
                far.append(0)
            else:
                far.append(1)
            
        if far == [0, 0, 0, 0, 0] or far == [1, 0, 0, 0, 0]:
            cv2.putText(image, 'Rock', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
        elif far == [1, 1, 1, 1, 1]:
            cv2.putText(image, 'Paper', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
        elif far == [0, 1, 1, 0, 0] or [1, 1, 1, 0, 0]:
             cv2.putText(image, 'Scissor', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
        else:
            cv2.putText(image, 'Unknowned', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
    
    cv2.imshow('MediaPipe Hands', image)
    c = cv2.waitKey(1)
    if c == 27:
        break
cap.release()
cv2.destroyAllWindows()