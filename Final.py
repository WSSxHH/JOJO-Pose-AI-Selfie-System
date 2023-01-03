import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
    if int(angle) == 0:
        return 1
    else:
        return int(angle) 
def calculate_point(angle1,angle2):
    angle1 = angle1 - angle2
    angle1 = angle1 / angle2
    point = (1.0 - np.abs(np.sum(angle1)/10))*100
    return int(point)
def standard_frame(sample):
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        image = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            #LEFT
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            foot = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
            #RIGHT
            Rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            Relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            Rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            Rhip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            Rknee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            Rankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            Rfoot = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
            
            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)
            angle2 = calculate_angle(hip, knee, ankle)
            angle3 = calculate_angle(elbow, shoulder, hip)
            angle4 = calculate_angle(shoulder, hip, knee)
            angle5 = calculate_angle(knee, ankle, foot)
            
            angle6 = calculate_angle(Rshoulder, Relbow, Rwrist)
            angle7 = calculate_angle(Rhip, Rknee, Rankle)
            angle8 = calculate_angle(Relbow, Rshoulder, Rhip)
            angle9 = calculate_angle(Rshoulder, Rhip, Rknee)
            angle10 = calculate_angle(Rknee, Rankle, Rfoot)

            # Visualize angle
            cv2.putText(image, str(angle), 
                           tuple(np.multiply(elbow, frame_size).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, str(angle2), 
                           tuple(np.multiply(knee, frame_size).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, str(angle3), 
                           tuple(np.multiply(shoulder, frame_size).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, str(angle4), 
                           tuple(np.multiply(hip, frame_size).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, str(angle5), 
                           tuple(np.multiply(ankle, frame_size).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            
            cv2.putText(image, str(angle6), 
                           tuple(np.multiply(Relbow, frame_size).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 5, cv2.LINE_AA
                                )
            cv2.putText(image, str(angle7), 
                           tuple(np.multiply(Rknee,frame_size).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 5, cv2.LINE_AA
                                )
            cv2.putText(image, str(angle8), 
                           tuple(np.multiply(Rshoulder, frame_size).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 5, cv2.LINE_AA
                                )
            cv2.putText(image, str(angle9), 
                           tuple(np.multiply(Rhip, frame_size).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 5, cv2.LINE_AA
                                )
            cv2.putText(image, str(angle10), 
                           tuple(np.multiply(Rankle, frame_size).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 5, cv2.LINE_AA
                                )
    
                       
        except:
            pass
        
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
#         cv2.namedWindow('Mediapipe Feed',cv2.WINDOW_NORMAL)
        cv2.imshow('Mediapipe standard', image)
        return np.array([angle,angle2,angle3,angle4,angle5,angle6,angle7,angle8,angle9,angle10])

cap = cv2.VideoCapture(0)
frame_size=np.array([526,720])
frame2_size=np.array([640,480])
sample = cv2.imread('sample2.jpg')    
standard_angle=standard_frame(sample)
maxpoint=90
i=1
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    photo_angle = np.array([1,1,1,1,1,1,1,1,1,1])
    while cap.isOpened():
        # Recolor image to RGB
            
        
        ret, frame = cap.read()
        ret, frame_cap = cap.read()
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            #LEFT
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            foot = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
            #RIGHT
            Rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            Relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            Rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            Rhip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            Rknee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            Rankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            Rfoot = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
            
            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)
            angle2 = calculate_angle(hip, knee, ankle)
            angle3 = calculate_angle(elbow, shoulder, hip)
            angle4 = calculate_angle(shoulder, hip, knee)
            angle5 = calculate_angle(knee, ankle, foot)
            
            angle6 = calculate_angle(Rshoulder, Relbow, Rwrist)
            angle7 = calculate_angle(Rhip, Rknee, Rankle)
            angle8 = calculate_angle(Relbow, Rshoulder, Rhip)
            angle9 = calculate_angle(Rshoulder, Rhip, Rknee)
            angle10 = calculate_angle(Rknee, Rankle, Rfoot)   
            photo_angle = np.array([angle,angle2,angle3,angle4,angle5,angle6,angle7,angle8,angle9,angle10])
            # Visualize angle
            cv2.putText(image, str(angle), 
                           tuple(np.multiply(elbow, frame2_size).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, str(angle2), 
                           tuple(np.multiply(knee, frame2_size).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, str(angle3), 
                           tuple(np.multiply(shoulder, frame2_size).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, str(angle4), 
                           tuple(np.multiply(hip, frame2_size).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, str(angle5), 
                           tuple(np.multiply(ankle, frame2_size).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            
            cv2.putText(image, str(angle6), 
                           tuple(np.multiply(Relbow, frame2_size).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 5, cv2.LINE_AA
                                )
            cv2.putText(image, str(angle7), 
                           tuple(np.multiply(Rknee,frame2_size).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 5, cv2.LINE_AA
                                )
            cv2.putText(image, str(angle8), 
                           tuple(np.multiply(Rshoulder, frame2_size).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 5, cv2.LINE_AA
                                )
            cv2.putText(image, str(angle9), 
                           tuple(np.multiply(Rhip, frame2_size).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 5, cv2.LINE_AA
                                )
            cv2.putText(image, str(angle10), 
                           tuple(np.multiply(Rankle, frame2_size).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 5, cv2.LINE_AA
                                )
            
                       
        except:
            pass
        
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               

        cv2.putText(image,str(calculate_point(photo_angle,standard_angle)) ,(30,150),cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 10, cv2.LINE_AA)
        cv2.imshow('Mediapipe photo line', image)       
        cv2.imshow('Mediapipe photo', frame_cap) 
        
        if calculate_point(photo_angle,standard_angle)>=maxpoint:
            savename = "nice-jojo" + str(i) + ".jpg"
            print(savename)
            cv2.imwrite(savename,frame_cap)
            maxpoint=calculate_point(photo_angle,standard_angle)
            i = i+1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        if i>=50:
            break
    cap.release()
    cv2.destroyAllWindows()
    
    
    
    
    
    
