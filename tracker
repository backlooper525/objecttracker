import cv2
import numpy as np

def detect_and_track_fastest_largest_object(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video was successfully opened
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Create a background subtractor
    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    # Initialize the tracker
    tracker = cv2.TrackerCSRT_create()

    # Variable to check if tracking is initialized
    initialized = False

    # Frame counter for skipping frames
    frame_count = 0
    skip_frames = 5  # Process every 3rd frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Increment the frame counter
        frame_count += 1

        # Resize the frame to speed up processing
        frame = cv2.resize(frame, (640, 480))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Skip frames to reduce processing load
        if frame_count % skip_frames != 0:
            if initialized:
                # Update the tracker
                success, box = tracker.update(frame)
                if success:
                    (x, y, w, h) = [int(v) for v in box]

                    # Draw the bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Display the frame
            cv2.imshow('Object Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        if not initialized:
            # Apply background subtraction
            fg_mask = back_sub.apply(frame)

            # Find contours in the foreground mask
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Find the largest moving object
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                (x, y, w, h) = cv2.boundingRect(largest_contour)

                # Initialize the tracker with the largest object
                tracker.init(frame, (x, y, w, h))
                initialized = True

        if initialized:
            # Update the tracker
            success, box = tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in box]

                # Draw the bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Object Tracking', frame)

        # Exit if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_path = 'PATH TO YOUR VIDEO FILE'
detect_and_track_fastest_largest_object(video_path)
