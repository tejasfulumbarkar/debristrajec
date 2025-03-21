import cv2
import numpy as np

def main():
    # Change this to your video file or camera source
    video_source = "DEBRIS VIDEO.mp4"
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Background subtractor for detecting moving objects
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
    
    # Initialize Kalman Filter
    # State vector: [x, y, dx, dy]
    # Measurement vector: [x, y]
    kalman = cv2.KalmanFilter(4, 2)
    
    # The measurement matrix maps the state to measurement space.
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
    
    # The transition matrix tells the filter how our state evolves from one timestamp to the next.
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    
    # Process noise covariance: tweak these values based on expected motion dynamics.
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    
    # For storing the predicted trajectory points
    trajectory_pts = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Step 1. Apply background subtraction to obtain foreground mask
        fgMask = backSub.apply(frame)
        
        # Clean up the mask using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
        
        # Step 2. Find contours in the mask
        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        measurement_available = False
        
        if contours:
            # Pick the largest contour by area (assuming debris is the dominant moving object)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            if area > 25:  # filter out very small contours as noise
                x, y, w, h = cv2.boundingRect(largest_contour)
                cx = x + w / 2
                cy = y + h / 2
                measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
                measurement_available = True
                
                # Correct the Kalman filter with the measurement
                kalman.correct(measurement)
                
                # Visualize the detection (green rectangle and red center point)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)
        
        # Step 3. Predict the next state with the Kalman filter
        prediction = kalman.predict()
        pred_x, pred_y = int(prediction[0]), int(prediction[1])
        trajectory_pts.append((pred_x, pred_y))
        
        # Visualize the Kalman prediction (blue circle)
        cv2.circle(frame, (pred_x, pred_y), 4, (255, 0, 0), -1)
        cv2.putText(frame, "Prediction", (pred_x + 5, pred_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw the predicted trajectory as a trail (blue line)
        for i in range(1, len(trajectory_pts)):
            cv2.line(frame, trajectory_pts[i - 1], trajectory_pts[i], (255, 0, 0), 2)
        
        # Display the results
        cv2.imshow("Frame", frame)
       # cv2.imshow("Foreground Mask", fgMask)
        
        # Press 'q' to quit
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()