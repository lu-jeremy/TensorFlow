import cv2

cap = cv2.VideoCapture(0)

i = 441

while True:
    ret, frame = cap.read()

    cv2.imwrite('./images/gesture_' + str(i) + '.jpg', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    i += 1
 
    cv2.imshow('frame', frame)
                
cap.release()
cv2.destroyAllWindows()
