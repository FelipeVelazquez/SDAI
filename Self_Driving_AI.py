from pydarknet import Detector, Image
import numpy as np
import cv2

net = Detector(bytes("cfg/yolov3_tiny.cfg", encoding="utf-8"), bytes("weights/yolov3-tiny.weights", encoding="utf-8"), 0, bytes("cfg/coco.data",encoding="utf-8"))
cap = cv2.VideoCapture('t1.mp4')
theta = 0 

while(cap.isOpened()):
	ret, frame = cap.read()
	img_g = frame

	###########################################################################################
	#hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	lower_red = np.array([0,200,200])
	upper_red = np.array([255,255,255])

	mask = cv2.inRange(frame, lower_red, upper_red)
	res = cv2.bitwise_and(frame,frame, mask= mask)
	image_g = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

	#Canny Filter 
	height, width = image_g.shape[:2]
	ROI = np.array([[(170, 612),(500, 480), (600, 490), (500, 640)]], dtype=np.int32)
	blank = np.zeros_like(image_g)
	maskB = cv2.fillPoly(blank, ROI, 255)
	m_image = cv2.bitwise_and(image_g, maskB)
	image_canny = cv2.Canny(m_image, 50, 200, apertureSize = 3)

	############################################################################################

	#Hough Line Detection
	lines = cv2.HoughLines(image_canny, 1, np.pi/180, 125)
	if lines is not None:
		theta = lines[0][0][1]
	else:
		theta = theta
	a = 1
	B = 0.5
	G = 0
	frame[image_canny>0] = (0,0,255)
	mask_res = cv2.fillPoly(res, ROI, 255)
	frame2 = cv2.addWeighted(img_g, a, mask_res, B, G)
	frame = cv2.bitwise_and(img_g,frame2)

	cv2.putText(frame2,"Angulo: " + str(int(np.degrees(theta))),(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)
	if(int(np.degrees(theta)) >= 65):
		cv2.putText(frame2,"Girar Derecha",(10,45),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)
	elif(int(np.degrees(theta)) <= 55):
		cv2.putText(frame2,"Girar Izquierda",(10,45),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)
	else:
		cv2.putText(frame2,"Avanzar",(10,45),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)


	##############################################################################################
	#AI Detection
	img_darknet = Image(frame2)
	results = net.detect(img_darknet)
	cantidad = 0    

	for cat, score, bounds in results:
		print("Detecting...")
		x, y, w, h = bounds
		cv2.rectangle(frame2, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), thickness=2)
		cv2.putText(frame2,str(cat.decode("utf-8")),(int(x),int(y)),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0))
		"""if (cat.decode("utf-8") == 'car'):
									cantidad = cantidad + 1"""

	#cv2.imshow('frame',frame)
	cv2.imshow('res',frame2)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
