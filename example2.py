from pydarknet import Detector, Image
import cv2


def loadModel():
	net = Detector(bytes("cfg/yolov3_tiny.cfg", encoding="utf-8"), bytes("weights/yolov3-tiny.weights", encoding="utf-8"), 0, bytes("cfg/coco.data",encoding="utf-8"))
	return net



def videoShow(net):
	cap = cv2.VideoCapture('london.mp4')
	#img = cv2.imread('car.jpg')
	while (True):
		ret, frame2 = cap.read()
		frame = cv2.resize(frame2, (720,560), interpolation = cv2.INTER_CUBIC) 
		img_darknet = Image(frame)

		results = net.detect(img_darknet)
		    
		for cat, score, bounds in results:
			if (cat.decode("utf-8") == 'person'):
			    x, y, w, h = bounds
			    cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), thickness=2)
			    cv2.putText(frame,str(cat.decode("utf-8")),(int(x- (w/2)),int(y- (h/2))),cv2.FONT_HERSHEY_COMPLEX,0.4,(255,255,0))

			if (cat.decode("utf-8") == 'car'):
			    x, y, w, h = bounds
			    cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 0, 255), thickness=2)
			    cv2.putText(frame,str(cat.decode("utf-8")),(int(x- (w/2)),int(y- (h/2))),cv2.FONT_HERSHEY_COMPLEX,0.4,(255,255,0))

		cv2.imshow("output", frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		#cv2.waitKey(0)



if __name__ == '__main__':
	netM = loadModel()
	videoShow(netM)