# Gesture recognition and display on the screen (Thumbs-up, Thumbs-down, OK, Victory)
# 辨識手勢並顯示在螢幕上

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

## Use distance from top of the finger to wrist, so that it won't missed when rotating
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
    thumb_flag = 0

    ## Get the hand landmarks and draw.
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
                    cv2.circle(image, pixelCoordinatesLandmark, 2, (255, 0, 0), -1)
                    if pixelCoordinatesLandmark:
                        pixelCoordinatesLandmarks.append(pixelCoordinatesLandmark)

        ## Compare whether distance of tip is further than pip's distance
        wrist = pixelCoordinatesLandmarks[0]
        tip_ids = [4, 8, 12, 16, 20]
        pip_ids = [1, 6, 10, 14, 18]
        distances_tip = [distance(wrist, pixelCoordinatesLandmarks[i]) for i in tip_ids]
        distances_pip = [distance(wrist, pixelCoordinatesLandmarks[i]) for i in pip_ids]
        far = []

        ## Thumb, if tip is further than cmc from wrist = 1
        if distances_tip[0] > distances_pip[0]:
            far.append(1)
            if distance(wrist, pixelCoordinatesLandmarks[4]) > 100 and distance(wrist, pixelCoordinatesLandmarks[4])\
            - distance(wrist, pixelCoordinatesLandmarks[6]) > 12:
                thumb_flag = 1
        else:
            far.append(0)
            
        #print(0, ' distances_tip[0] = ',  distances_tip[0], ' distances_pip[0] = ',  distances_pip[0])
        
        ## Another four fingers, if tip is further than pip from wrist = 1
        for id in range(1, 5):
            if distances_tip[id] > distances_pip[id]:
                far.append(1)
                #print(id, ' distances_tip[id] = ',  distances_tip[id], ' distances_pip[id] = ',  distances_pip[id])
            else:
                far.append(0)
                #print(id, ' distances_tip[id] = ',  distances_tip[id], ' distances_pip[id] = ',  distances_pip[id])

        ## Find out the gesture after comparing pip's distance and tip's distance
        if far == [1, 0, 0, 0, 0] and thumb_flag == 1:
            ## Compare wrist's y-axis with top of the thumb's y-axis
            if(wrist[1] >  pixelCoordinatesLandmarks[4][1]):
                cv2.putText(image, 'Thumbs-up', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
            else:
                cv2.putText(image, 'Thumbs-up(down)', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)             
        elif far == [1, 0, 1, 1, 1]:
            cv2.putText(image, 'OK', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
        elif far == [1, 1, 1, 0, 0]:
            cv2.putText(image, 'Victory', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
        else:
            cv2.putText(image, 'Unknowned', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
    
    cv2.imshow('MediaPipe Hands', image)
    c = cv2.waitKey(1)
    if c == 27:
        break
cap.release()
cv2.destroyAllWindows()