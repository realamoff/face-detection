
from pydoc import cram
from types import CoroutineType
import cv2

capture = cv2.VideoCapture(0)

pretrained_model = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
i = 0
while True:
    boolean, frame = capture.read()
    if boolean == True:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        coordinate_list = pretrained_model.detectMultiScale(gray, scaleFactor = 1.1,minNeighbors=3)
        for (x,y,w,h) in coordinate_list:
            cv2.rectangle(frame,(x,y),(x+w,y+h), (200,200,0),7)
            crop_img = frame[y:y+h,x:x+w]
            
            cv2.imwrite("kirpilmis"+str(i)+".jpg",crop_img)
            
            i+=1
            
            




                
                    
                    



        cv2.imshow("live face detection",frame)
        if cv2.waitKey(10) == ord("q"):
            break
capture.release()
cv2.destroyAllWindows()


