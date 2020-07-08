from pydarknet import Detector, Image
import cv2

net = Detector(bytes("cfg/yolov3_tiny.cfg", encoding="utf-8"), bytes("weights/yolov3-tiny.weights", encoding="utf-8"), 0, bytes("cfg/coco.data",encoding="utf-8"))

img = cv2.imread('car2.jpg')
img_darknet = Image(img)

results = net.detect(img_darknet)

cantidad = 0 
    
for cat, score, bounds in results:
    x, y, w, h = bounds
    cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), thickness=2)
    cv2.putText(img,str(cat.decode("utf-8")),(int(x),int(y)),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0))
    if (cat.decode("utf-8") == 'car'):
    	cantidad = cantidad + 1

print(cantidad)
cv2.imshow("output", img)
cv2.waitKey(0)