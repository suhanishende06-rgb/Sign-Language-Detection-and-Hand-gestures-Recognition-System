
import cv2
from hand_detection import HandDetector

detector = HandDetector()

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Webcam not found!")
    exit()

print("Webcam Started Successfully!")
print("Press Q to close camera.")

while True:
    success, frame = camera.read()

    if not success:
        print("Camera Frame Error!")
        break

    frame = cv2.flip(frame, 1)

    frame, results = detector.detect(frame)

    if results.multi_hand_landmarks:
        hand_count = len(results.multi_hand_landmarks)
        cv2.putText(
            frame,
            f"Hands Detected: {hand_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow("ISL - Live Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
detector.close()
cv2.destroyAllWindows()
