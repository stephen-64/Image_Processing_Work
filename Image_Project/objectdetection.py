import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='object detection using YOLOV3')
parser.add_argument('-v','--verbose', help="verbosity", action='store_true', default='store_false')
parser.add_argument('-a','--architecture',help='YOLO architecture')
parser.add_argument('-w','--weights', help='weights from trained data')
parser.add_argument('-c','--classes', help='data classes')
parser.add_argument('-i','--image',help='object detection on image')
parser.add_argument('-m','--movie',help='object detection on video')
args = parser.parse_args()

#crtl-C ctrl-v
def object_detection():
    blob = cv2.dnn.blobFromImage(image, 1.0/255.0, (320,320), [0,0,0], True, crop=False)
    Width = image.shape[1]
    Height = image.shape[0]
    net.setInput(blob)
    
    outs = net.forward(getOutputsNames(net))
    
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    
    for out in outs: 
        #print(out.shape)
        for detection in out:
            
        #each detection  has the form like this [center_x center_y width height obj_score class_1_score class_2_score ..]
            scores = detection[5:]#classes scores starts from index 5
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    
    # apply  non-maximum suppression algorithm on the bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_pred(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
   
    # Put efficiency information.
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(image, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
    
    cv2.imshow(window_title, image)
    
    usleep = lambda x: time.sleep(x/1000000.0)
    usleep(100)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

RED = '#fc0505'
SILVER = '#b3b3b3'
BLUE = '#054ffc'
ORANGE = '#fc8505'
DARKRED = '#a10000'
LIGHTBLUE = '#ooccff'
PINK = 'f700ff'
BROWN = '#654321'
WHITE = '#ffffff'

#defined color for each class
COLORS = [RED,SILVER,BLUE,ORANGE,DARKRED,LIGHTBLUE,PINK,BROWN,WHITE]
if args.verbose:
    print(COLORS)

#put classes into a list
with open(args.classes, 'r') as classes:
    CLASSES = classes.readlines()
CLASSES = [x.strip() for x in CLASSES]
if args.verbose:
    print(CLASSES)

#configure network with weights
TRAINED_NET = cv2.dnn.readNet(args.weights,args.architecture)


#select imput type
if args.image:
    if args.verbose:
        print("loading image: "+args.image)
    image = cv2.imread(args.image)
    object_detection()
    
elif args.movie:
    if args.verbose:
        print("loading video: "+args.movie)
    cap = cv2.VideoCapture(args.movie)
    while(cap.isOpened()):
        hasframe, image = cap.read()
        object_detection()
    
else:
    if args.verbose:
        print("testing network")