import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True,
                       max_num_hands=2, min_detection_confidence=0.5,
                       min_tracking_confidence=0.5, model_complexity=0)

mp_drawing = mp.solutions.drawing_utils

Cap = cv2.VideoCapture(0)

while True:
    Ret, Frames = Cap.read()
    Frames = cv2.flip(Frames, 1)
    Frames_rgb = cv2.cvtColor(Frames, cv2.COLOR_BGR2RGB)
    
    Results = hands.process(Frames_rgb)
    
    Circles_color=mp_drawing.DrawingSpec(color=(255,0,0),thickness=4, circle_radius=2)
    Lines_color=mp_drawing.DrawingSpec(color=(0,0,255),thickness=3)
    
    if Results.multi_hand_landmarks is not None:
        for Landmarks in Results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(Frames,Landmarks, 
                                      mp_hands.HAND_CONNECTIONS,
                                      Circles_color,Lines_color)
            
    cv2.imshow("LANDMARKS",Frames)

    t = cv2.waitKey(1)  
    if t == ord('q') or t == ord("Q"):
        break
    
Cap.release()
cv2.destroyAllWindows()
