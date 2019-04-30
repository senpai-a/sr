import cv2
import numpy as np
a=np.random.random((100,100))
cv2.imshow("aaa",a)
k=cv2.waitKey(0)
if k==ord('0'):
    print('0')
    cv2.destroyAllWindows()
else:
    cv2.imshow('',np.random.random((100,100)))
    cv2.waitKey(0)
print(k)
cv2.destroyAllWindows()