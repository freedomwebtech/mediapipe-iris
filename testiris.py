import cv2
import mediapipe as mp
import numpy as np
LEFT_EYE=[362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
RIGHT_EYE=[33,7,163,144,145,153,155,133,173,157,158,159,160,161,246]
LEFT_IRIS=[474,475,476,477]
RIGHT_IRIS=[469,470,471,472]
mp_mesh_face=mp.solutions.face_mesh
face_mesh=mp_mesh_face.FaceMesh(max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.5,min_tracking_confidence=0.5)
#cap=cv2.VideoCapture("videofilepaTH")
cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    frame=cv2.resize(frame,(640,480))
    rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results=face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        meshpoints=np.array([np.multiply([p1.x,p1.y],[640,480]).astype(int)for p1 in results.multi_face_landmarks[0].landmark])
        (x1,y1),l_radius=cv2.minEnclosingCircle(meshpoints[LEFT_IRIS])
        center_of_left_iris=np.array([x1,y1],dtype=np.int32)
        x1,y1=center_of_left_iris
#        print(x1,y1)
        (x2,y2),r_radius=cv2.minEnclosingCircle(meshpoints[RIGHT_IRIS])
        center_of_right_iris=np.array([x2,y2],dtype=np.int32)
        x2,y2=center_of_right_iris
        cv2.circle(frame,(x1,y1),3,(0,0,255),3)
        cv2.circle(frame,(x1,y1),int(l_radius),(0,255,0),3)
        cv2.circle(frame,(x2,y2),3,(0,0,255),3)
        cv2.circle(frame,(x2,y2),int(r_radius),(0,255,0),3)
    
    
    
        
    cv2.imshow("FRAME",frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()