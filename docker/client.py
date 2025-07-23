# read_remote.py
import cv2

# Reading from a remote camera stream
stream_url = "http://localhost:8080"  

cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("it was not possible to open the stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("it was not possible to read the frame.")
        break

    cv2.imshow("Remote Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
