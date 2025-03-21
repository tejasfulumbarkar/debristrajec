import cv2
import numpy as np
import time

class DebrisTracker:
    def __init__(self, id, initial_pos):
        self.id = id
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)
        self.kalman.statePost = np.array([[initial_pos[0]],
                                          [initial_pos[1]],
                                          [0],
                                          [0]], np.float32)
        self.last_position = initial_pos
        self.skipped_frames = 0
        self.trajectory = []  # List to store trajectory points
        self.color = (238, 130, 238)  # Default color (pink)

    def predict(self):
        prediction = self.kalman.predict()
        pred_pos = (int(prediction[0][0]), int(prediction[1][0]))  # Extract single elements
        self.trajectory.append(pred_pos)  # Add prediction to trajectory
        return pred_pos

    def update(self, measurement):
        msr = np.array([[np.float32(measurement[0])],
                        [np.float32(measurement[1])]])
        self.kalman.correct(msr)
        self.last_position = measurement
        self.skipped_frames = 0

    def get_state(self):
        state = self.kalman.statePost.flatten()
        return state

    def compute_priority(self, removal_position):
        state = self.get_state()
        pos = np.array([state[0], state[1]])
        vel = np.array([state[2], state[3]])
        d = pos - removal_position
        v_norm_sq = np.dot(vel, vel)
        if v_norm_sq == 0:
            return float('inf'), pos, np.linalg.norm(d)
        t_closest = - np.dot(d, vel) / v_norm_sq
        t_closest = max(t_closest, 0)
        predicted_pos = pos + vel * t_closest
        min_distance = np.linalg.norm(predicted_pos - removal_position)
        priority = t_closest
        return priority, predicted_pos, min_distance

def main():
    cap = cv2.VideoCapture("DEBRIS VIDEO.mp4")
    if not cap.isOpened():
        print("Error: Cannot open video source.")
        return

    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
    trackers = {}
    next_track_id = 1
    max_skipped_frames = 10
    association_threshold = 50
    removal_position = np.array([100, 100])

    start_time = time.time()  # Start timer

    while True:
        ret, frame = cap.read()
        if not ret or (time.time() - start_time) > 5:  # Stop after 5 seconds
            break

        fgMask = backSub.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            cx = x + w // 2
            cy = y + h // 2

            if area < 25:
                continue

            detections.append((cx, cy, area))

        assigned_det = set()
        for tid, tracker in trackers.items():
            pred = tracker.predict()
            min_dist = float('inf')
            matched_detection = None
            for i, detection in enumerate(detections):
                if i in assigned_det:
                    continue
                cx, cy, area = detection
                distance = np.linalg.norm(np.array(pred) - np.array([cx, cy]))
                if distance < min_dist:
                    min_dist = distance
                    matched_detection = (i, detection)
            if matched_detection is not None and min_dist < association_threshold:
                det_index, det = matched_detection  # Correctly unpack two values
                tracker.update(det[:2])
                assigned_det.add(det_index)
                # Update color based on area
                if det[2] < 50:
                    tracker.color = (238, 130, 238)  # Pink for small debris
                elif 50 <= det[2] < 200:
                    tracker.color = (0, 165, 255)  # Orange for medium debris
                else:
                    tracker.color = (0, 255, 0)  # Green for large debris
            else:
                tracker.skipped_frames += 1

        for i, detection in enumerate(detections):
            if i not in assigned_det:
                cx, cy, area = detection
                trackers[next_track_id] = DebrisTracker(next_track_id, (cx, cy))
                # Set color based on area
                if area < 50:
                    trackers[next_track_id].color = (238, 130, 238)  # Pink for small debris
                elif 50 <= area < 200:
                    trackers[next_track_id].color = (0, 165, 255)  # Orange for medium debris
                else:
                    trackers[next_track_id].color = (0, 255, 0)  # Green for large debris
                next_track_id += 1

        del_ids = [tid for tid, trk in trackers.items() if trk.skipped_frames > max_skipped_frames]
        for tid in del_ids:
            del trackers[tid]

        removal_info = []
        for tid, tracker in trackers.items():
            priority, pred_pos, min_distance = tracker.compute_priority(removal_position)
            removal_info.append((tid, priority, pred_pos, min_distance))
            state = tracker.get_state()
            current_pos = (int(state[0]), int(state[1]))
            contour = np.array([[current_pos]], dtype=np.int32)  # Create a contour from the single point
            area = cv2.contourArea(contour)

            # Draw debris rectangle and center.
            cv2.circle(frame, current_pos, 5, tracker.color, -1)
            cv2.putText(frame, f"ID:{tid}", (current_pos[0] + 5, current_pos[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, tracker.color, 2)

            # Draw trajectory points for this tracker.
            for j in range(1, len(tracker.trajectory)):
                cv2.line(frame, tracker.trajectory[j - 1], tracker.trajectory[j], tracker.color, 2)

        # Save removal order to a text file
        with open("removal_order.txt", "w") as f:
            removal_info_sorted = sorted(removal_info, key=lambda x: x[1])
            f.write("Removal Order:\n")
            for i, info in enumerate(removal_info_sorted):
                tid, priority, pred_int, min_distance = info
                text = f"{i+1}: ID {tid}, t_closest: {priority:.2f}, d: {min_distance:.1f}\n"
                f.write(text)

        removal_asset_pt = (int(removal_position[0]), int(removal_position[1]))
        cv2.circle(frame, removal_asset_pt, 8, (0, 0, 255), -1)
        cv2.putText(frame, "Removal Asset", (removal_asset_pt[0] + 10, removal_asset_pt[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        #cv2.imshow("Foreground Mask", fgMask)
        key = cv2.waitKey(30) & 0xFF  # Adjusted delay to 30 milliseconds
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()