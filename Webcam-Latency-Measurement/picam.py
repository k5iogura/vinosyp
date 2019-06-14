from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
 
camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(320, 240))
 
# allow the camera to warmup
time.sleep(0.1)
 
count = 0
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    t1 = time.time()
    image = frame.array
 
    # show the frame
    cv2.imshow("Frame", image)
    key = cv2.waitKey(100) & 0xFF
 
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
 
    if key == ord("q"):
        break
    count += 1
    t2 = time.time()
    if count >= 33:
        fps = count / ( t2 - t1 )
        print("%.3f %d / %.6f"%(fps,count,(t2-t1)))
        count = 0
