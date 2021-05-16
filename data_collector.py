import cv2

video=cv2.VideoCapture(0)
faces=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

count = 0

while True:
    ret,frame=video.read()
    facedet=faces.detectMultiScale(frame,1.3,5)
    for x,y,w,h in facedet:
        count=count+2
        name='./helmet_self_dataset/with_helmet/with_'+str(count+100-1) + '.jpg'
        print("Creating Images........." +str(name))
        cv2.imwrite(name, frame[y:y+h,x-15:x+w])
        frame=cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)
    if k==ord('c') or count>=100:
        break

video.release()
cv2.destroyAllWindows()