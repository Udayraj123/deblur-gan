import cv2
import glob
for x in glob.glob('*.jpg'):
	two = cv2.imread(x,cv2.IMREAD_COLOR);
	h,w = two.shape[:2]
	x_t,y_t = two[:,:w/2],two[:,w/2:]
	cv2.imwrite('A/'+x,x_t)
	cv2.imwrite('B/'+y,y_t)