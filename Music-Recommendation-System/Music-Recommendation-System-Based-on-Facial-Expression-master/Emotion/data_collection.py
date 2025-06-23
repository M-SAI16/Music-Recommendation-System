import mediapipe as mp 
import numpy as np 
import cv2 
import sys
import time

def initialize_webcam():
    """Try to initialize the webcam with multiple attempts"""
    max_attempts = 3
    for i in range(max_attempts):
        cap = cv2.VideoCapture(0)
        if cap is None or not cap.isOpened():
            print(f"Warning: Unable to open webcam. Attempt {i+1}/{max_attempts}")
            time.sleep(1)
        else:
            print("Webcam initialized successfully!")
            return cap
    
    raise ValueError("Failed to initialize webcam after multiple attempts")

def main():
    try:
       
        holistic = mp.solutions.holistic
        hands = mp.solutions.hands
        holis = holistic.Holistic()
        drawing = mp.solutions.drawing_utils

        
        cap = initialize_webcam()
        if cap is None:
            raise ValueError("Failed to initialize webcam")

        
        name = input("Enter the name of the data (e.g., happy, sad, etc.): ")
        print("\nPress ESC to stop recording early")
        print("Recording will automatically stop after 100 frames")
        print("\nPreparing to record in 3 seconds...")
        time.sleep(3)

        X = []
        data_size = 0

        while True:
            lst = []
            ret, frm = cap.read()

            if not ret:
                print("Failed to grab frame")
                break

            
            frm = cv2.flip(frm, 1)

            
            rgb_frm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
            res = holis.process(rgb_frm)

            if res.face_landmarks:
                
                for i in res.face_landmarks.landmark:
                    lst.append(i.x - res.face_landmarks.landmark[1].x)
                    lst.append(i.y - res.face_landmarks.landmark[1].y)

               
                if res.left_hand_landmarks:
                    for i in res.left_hand_landmarks.landmark:
                        lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                        lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
                else:
                    lst.extend([0.0] * 42)

                
                if res.right_hand_landmarks:
                    for i in res.right_hand_landmarks.landmark:
                        lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                        lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
                else:
                    lst.extend([0.0] * 42)

                X.append(lst)
                data_size += 1

                
                drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
                drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
                drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

            
            status_text = f"Recording: {data_size}/100 frames"
            cv2.putText(frm, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Data Collection", frm)

            
            if cv2.waitKey(1) == 27 or data_size >= 100:
                break

        
        cv2.destroyAllWindows()
        cap.release()

    
        if data_size > 0:
            np.save(f"{name}.npy", np.array(X))
            print(f"\nData collection complete!")
            print(f"Saved {data_size} frames to {name}.npy")
            print(f"Data shape: {np.array(X).shape}")
        else:
            print("\nNo data was collected!")

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure your webcam is properly connected")
        print("2. Try closing other applications that might be using the webcam")
        print("3. Try restarting your computer if the problem persists")
        print("4. Check if you have the correct permissions to access the webcam")
        sys.exit(1)

if __name__ == "__main__":
    main()
    