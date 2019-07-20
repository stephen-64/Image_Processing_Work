import cv2
import numpy as np
import argparse
import sys
import time

def vid_runner(config,weights,classes,vid):
    parser = argparse.ArgumentParser(description='object detection using YOLOV3')
    parser.add_argument('-v','--verbose', help="verbosity", action='store_true', 
        default='store_false')
    parser.add_argument('-a','--config',help='YOLO config',
        default=config)
    parser.add_argument('-w','--weights', help='weights from trained data', 
        default=weights)
    parser.add_argument('-c','--classes', help='data classes',
        default=classes)
    parser.add_argument('-i','--input',help='object detection on image or video',
        default=vid)
    parser.add_argument('-t','--confidence',help='threshold',default=0.1)
    parser.add_argument('-n','--nms',help='non-maximal noise suppression',default=0.05)
    parser.add_argument('-m','--matrix',help='size of initial image',default=320)
    args = parser.parse_args()

    RED = (5,5,252)
    SILVER = (180,180,180)
    BLUE = (252,79,5)
    ORANGE = (5,133,252)
    DARKRED = (0,0,161)
    YELLOW = (52,252,252)
    LIGHTBLUE = (255,204,0)
    PINK = (247,0,255)
    BROWN = (33,67,101)
    WHITE = (255,255,255)

    #defined color for each class
    COLORS = [RED,SILVER,BLUE,ORANGE,DARKRED,YELLOW,LIGHTBLUE,PINK,BROWN,WHITE]
    if args.verbose:
        print(COLORS)

    #put classes into a list
    with open(args.classes, 'r') as classes:
        CLASSES = classes.readlines()
    CLASSES = [x.strip() for x in CLASSES]
    if args.verbose:
        print(CLASSES)

    #configure network with weights
    net = cv2.dnn.readNet(args.weights,args.config)
    #net.setPreferableBackend(args.backend)
    #net.setPreferableTarget(args.target)

    WINDOW_NAME = 'Vehicle-Object-Detection'
    cv2.namedWindow(WINDOW_NAME,cv2.WINDOW_NORMAL)

    def callback_conf(pos):
        global threshold_confidence
        threshold_confidence = pos / 100.0
        
    def callback_nms(pos):
        global threshold_nms
        threshold_nms = pos/100.0

    threshold_confidence = args.confidence
    threshold_nms = args.nms

    cv2.createTrackbar('Confidence threshold, %', WINDOW_NAME, int(threshold_confidence * 100), 99, callback_conf)
    cv2.createTrackbar('Non-Maxima noise supression, %', WINDOW_NAME, int(threshold_nms * 100), 99, callback_nms)

    cap = cv2.VideoCapture(args.input)

    def getOutputsNames(net):
        layersNames = net.getLayerNames()
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    while(cap.isOpened()):
        hasframe, image = cap.read()
        overlay = image.copy()
        net.setInput(cv2.dnn.blobFromImage(image, 1.0/255.0, (args.matrix,args.matrix), [0,0,0], True))
        Width, Height = image.shape[1], image.shape[0]

        outs = net.forward(getOutputsNames(net))

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
                for detection in out:
                    scores = detection[5:]
                    classId = np.argmax(scores)
                    confidence = scores[classId]
                    if confidence > threshold_confidence:
                        center_x = int(detection[0] * Width)
                        center_y = int(detection[1] * Height)
                        width = int(detection[2] * Width)
                        height = int(detection[3] * Height)
                        left = int(center_x - width / 2)
                        top = int(center_y - height / 2)
                        class_ids.append(classId)
                        confidences.append(float(confidence))
                        boxes.append([left, top, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold_confidence, threshold_nms)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            
            label = str(CLASSES[class_ids[i]]+': {:.2f}%'.format(100*confidences[i]))
            color = COLORS[class_ids[i]]

            cv2.rectangle(image, (int(left),int(top)), (int(left + width),int(top + height)), color, 2)
            cv2.putText(image, label, (int(left),int(top-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            alpha = (confidences[i]-threshold_confidence)*(1/(1-threshold_confidence))
            cv2.addWeighted(image,alpha,overlay,1-alpha,0,image)

        # Put efficiency information.
        if args.verbose:
            t, _ = net.getPerfProfile()
            label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
            cv2.putText(image, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        
        cv2.imshow(WINDOW_NAME, image)
        #if args.verbose:
            #cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty(WINDOW_NAME,0) == -1:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    vid_runner()