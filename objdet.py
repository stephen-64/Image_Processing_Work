import cv2
import numpy as np
import argparse
import sys
import time

def vid_runner(config,weights,classes,vid):
    parser = argparse.ArgumentParser(description='object detection using YOLOV3')
    parser.add_argument('-a','--config',help='YOLO config',
        default='YOLO_Tiny/yolov3-tiny-F1-10C.cfg')
    parser.add_argument('-w','--weights', help='weights from trained data', 
        default='YOLO_Tiny/yolov3-tiny-F1-10C_60000.weights')
    parser.add_argument('-c','--classes', help='data classes',
        default='YOLO_Tiny/F1-obj-10C.names')
    parser.add_argument('-i','--input',help='object detection on image or video',
        default='F1_Austria_Race_Trimmed30.mkv')
    parser.add_argument('-t','--confidence',help='threshold',default=0.1)
    parser.add_argument('-n','--nms',help='non-maximal noise suppression',default=0.7)
    parser.add_argument('-m','--matrix',help='size of initial image',default=320)
    args = parser.parse_args()

    print(cv2.__version__)

    matrix_values = [320,416,608]
    i = 0

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

    verbose = True

    #defined color for each class
    COLORS = [RED,SILVER,BLUE,ORANGE,DARKRED,YELLOW,LIGHTBLUE,PINK,BROWN,WHITE]
    if verbose:
        print(COLORS)

    #put classes into a list
    with open(args.classes, 'r') as classes:
        CLASSES = classes.readlines()
    CLASSES = [x.strip() for x in CLASSES]
    if verbose:
        print(CLASSES)

    #configure network with weights
    net = cv2.dnn.readNetFromDarknet(args.config,args.weights)

    WINDOW_NAME = 'Vehicle-Object-Detection using YOLO'
    cv2.namedWindow(WINDOW_NAME)

    def callback_conf(pos):
        global threshold_confidence
        threshold_confidence = pos / 100.0
        
    def callback_nms(pos):
        global threshold_nms
        threshold_nms = pos/100.0

    threshold_confidence = args.confidence
    threshold_nms = args.nms
    matrix = args.matrix

    cv2.createTrackbar('Confidence threshold, %', WINDOW_NAME, int(threshold_confidence * 100), 99, callback_conf)
    cv2.createTrackbar('Non-Maxima noise supression, %', WINDOW_NAME, int(threshold_nms * 100), 99, callback_nms)

    cap = cv2.VideoCapture(args.input)
    frames = 0

    while(cap.isOpened()):
        hasframe, image = cap.read()
        overlay = image.copy()
        scale_factor = 1.0/255.0
        net.setInput(cv2.dnn.blobFromImage(image, scale_factor, (matrix,matrix), swapRB=True))
        Height, Width = image.shape[:2]

        #only works with yolo_tiny and 320?
        outs = net.forward(['yolo_16', 'yolo_23'])

        class_ids = []
        confidences = []
        boxes = []
        cars_on_track = np.zeros(10,dtype=np.int)
    #-------
        for out in outs:
            for detection in out:
                #ignore first five values scores is the conficence of of the box
                scores = detection[5:]
                # find index of max confidence
                classId = np.argmax(scores)
                #get max confidence
                confidence = scores[classId]
                if confidence > threshold_confidence:
                    
                    cars_on_track[classId] += 1

                    #size and centre of box
                    CX, CY, width, height = detection[0:4] * [Width, Height, Width, Height]

                    #reformat to find corners
                    X = int(CX - width / 2)
                    Y = int(CY - height / 2)
                    width = int(width)
                    height = int(height)

                    #corners of the box
                    boxes.append([X,Y,width,height])
                    class_ids.append(classId)
                    confidences.append(float(confidence))

        #performs non maxima suppression
        remaining_boxes = cv2.dnn.NMSBoxes(boxes, confidences, threshold_confidence, threshold_nms)

        #remaining boxes after NMS
        for i in remaining_boxes:
            i = i[0]
            left, top, width, height = boxes[i]
            
            label_car = str(CLASSES[class_ids[i]]+': {:.2f}%'.format(100*confidences[i]))
            color = COLORS[class_ids[i]]
            label_numberof_cars = str(cars_on_track[class_ids[i]])
    #-------
            cv2.rectangle(image, (int(left-10),int(top-10)), (int(left + width + 10),int(top + height + 10)), color, 2)
            cv2.putText(image, label_car, (int(left-10),int(top-15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(image,label_numberof_cars,(int(left+width-5),int(top+height+5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            alpha = confidences[i]+0.5
            if alpha > 1:
                alpha = 1
            cv2.addWeighted(image,alpha,overlay,1-alpha,0,image)

        # debug info
        if verbose:
            time, _ = net.getPerfProfile()
            label_frames = 'time: %.2f s' % (float(frames)*0.02)
            frames += 1
            label_inference = 'Inference time: %.2f ms' % (time * 1000.0 / cv2.getTickFrequency())
            label_config =    'Config: %s' % args.config
            label_weight =    'Weight: %s' % args.weights
            label_input =     'Input: %s' % args.input
            label_mat_size =  'Mat Size: %s' % matrix

            cv2.putText(image, label_frames, (Width - 150, 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
            cv2.putText(image, label_inference, (0, 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
            cv2.putText(image, label_config, (0, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
            cv2.putText(image, label_weight, (0, 45), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
            cv2.putText(image, label_input, (0, 60), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
            cv2.putText(image, label_mat_size, (0, 75), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
            print(cars_on_track)
        
        cv2.imshow(WINDOW_NAME, image)
        #if verbose:
            #cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord('v'):
            verbose = not verbose
            sleep(1)
        if cv2.getWindowProperty(WINDOW_NAME,0) == -1:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    vid_runner()