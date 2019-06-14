import sys, os
import threading
from time import time
import cv2
 
 
class UVC:
    def __init__(self,deviceNo=0):
        assert os.path.exists('/dev/video'+str(deviceNo))
        self.cap = cv2.VideoCapture(deviceNo)
        assert self.cap is not None
        r,self.frame = self.cap.read()
        assert r is True
        self.cont  = True
        self.thread= None
    def _read_task(self):
        while True:
            if not self.cont:break
            r,self.frame = self.cap.read()
            assert r is True
        self.cap.release()
    def start(self):
        self.thread = threading.Thread(target=self._read_task,args=())
        self.thread.start()
        return self
    def stop(self):
        self.cont=False
        self.thread.join()
    def read(self):
        return self.frame

count = 0
uvc = UVC().start()
start = time()
while True:
    image = uvc.read()
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1)
    if key != -1:
        uvc.stop()
        break
    count+=1
    sys.stdout.write('\b'*40)
    sys.stdout.write('%.3f FPS'%(count/(time()-start)))
    sys.stdout.flush()
print("\nfin")
