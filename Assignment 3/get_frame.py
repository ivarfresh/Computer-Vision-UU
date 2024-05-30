import numpy as np
import cv2


cap = cv2.VideoCapture('data/video/cam2.avi')

# to get online frame
cap2 = cv2.VideoCapture('data/video/cam2.avi')

ret_list = []
while(cap.isOpened()):
    ret, frame = cap.read()
    #ret_list.append(ret)
    #print(f'len ret_list:{len(ret_list)}')

    #if len(ret_list) > 100:
    #    cv2.imwrite('frame2.jpg', frame)

    #    break

    if ret:
        cv2.imwrite('frame2.jpg', frame)

    break


cap.release()
cv2.destroyAllWindows()
