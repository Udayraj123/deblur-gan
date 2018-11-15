import cv2
import glob
for x in glob.glob('*.jpg'):
	two = cv2.imread(x,cv2.IMREAD_COLOR);
	h,w = two.shape[:2]
	x_t,y_t = two[:,:int(w/2)],two[:,int(w/2):]
	cv2.imwrite('A/'+x,x_t)
	cv2.imwrite('B/'+x,y_t)