import cv2
import numpy as np
import time
net = cv2.dnn.readNet("yolov3_training_2500.weights","yolov3_testing.cfg")
layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture(1)  # 0 for 1st webcam
font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
while True:
    _, frame = cap.read()  #
    frame=cv2.flip(frame,1)
    frame_id += 1

    height2, width2, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (640, 640), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(outputlayers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # onject detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(
                    float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str("VR")
            confidence = confidences[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255,255,255), 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 1, (255, 255, 255), 2)

    elapsed_time = time.time() - starting_time
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)

    if key == 27:
        break;

cap.release()
cv2.destroyAllWindows()
def show_webcam(mirror=False):
    cam = cv2.VideoCapture(1)
    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()

