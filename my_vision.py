import cv2
import face_recognition
import numpy as np


print("Loading reference face...")
my_image = face_recognition.load_image_file("me.jpg")
my_encoding = face_recognition.face_encodings(my_image)[0]


video_capture = cv2.VideoCapture(0)

print("Vision System Active. Looking for YOU...")

while True:
    #  GET FRAME
    ret, frame = video_capture.read()
    if not ret: break
    
    # Scale down for speed 
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

   
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    target_found = False
    
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        
        
        matches = face_recognition.compare_faces([my_encoding], face_encoding, tolerance=0.6)
        face_distance = face_recognition.face_distance([my_encoding], face_encoding)
        
        
        top *= 4; right *= 4; bottom *= 4; left *= 4

        if matches[0]:
           
            target_found = True
            
            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"TARGET (Dist: {face_distance[0]:.2f})", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            
            # Calculate center of the face
            center_x = (left + right) / 2
            center_y = (top + bottom) / 2
            
            # Calculate Error (0.0 is center, -1.0 is left, 1.0 is right)
            height, width, _ = frame.shape
            error_x = (center_x - (width / 2)) / (width / 2)
            error_y = (center_y - (height / 2)) / (height / 2)
            
            # Calculate Area (Distance)
            face_area = (bottom - top) * (right - left)
            area_norm = face_area / (width * height)
            
            print(f"SENDING TO DRONE -> ErrorX: {error_x:.2f}, ErrorY: {error_y:.2f}, Area: {area_norm:.2f}")
            
            
            
        else:
            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    #
    if not target_found:
        print("Target Lost - Hovering...")

    
    cv2.imshow('Drone Vision', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
