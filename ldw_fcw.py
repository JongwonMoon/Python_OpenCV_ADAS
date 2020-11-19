import sys
import numpy as np
import cv2
import logging
import time
import winsound



# Gray -> filter -> edge
def detect_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)


    edges = cv2.Canny(blur, 50, 150)
    return edges

# ROI Set
def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)
    
    polygon = np.array([[
    (0, height),          
    (width, height),
    (320, 170),
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    return cropped_edges


# Hough Transform
def detect_line_segments(cropped_edges):
    rho = 1  
    angle = np.pi / 180 
    min_threshold = 20 
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, 
    np.array([]), minLineLength=30, maxLineGap=50)
    return line_segments


# x1,y1,x2,y2축으로 기울기,y절편을 구함 .. 그리고 라인의 평균을 구함
def average_slope_intercept(frame, line_segments):
    lane_lines = []

    if line_segments is None:
        
        logging.info('No line_segment segments detected')
        return lane_lines
    
    height, width = frame.shape[:2]
    
    left_fit = []
    right_fit = []

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                logging.info('skipping vertical line segment (slope=inf): %s' % line_segment)
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]

            if slope < -0.8 and slope > -1.2:
                left_fit.append((slope, intercept))
            elif slope > 0.5 and slope < 1.0:
                right_fit.append((slope, intercept))
            

    left_fit_average = np.average(left_fit, axis=0)
    
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))
       
        if lane_lines[0][0][0] < 30 :
            str = 'Warning'
            cv2.putText(frame, str, (80, 170), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 0, 255), thickness=2)
            winsound.Beep(3500, dur)
    
    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))
        
        if lane_lines[0][0][0] > 630 :
            str = 'Warning'
            cv2.putText(frame, str, (80, 170), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 0, 255), thickness=2)
            winsound.Beep(3500, dur)
    
    logging.debug('lane lines: %s' % lane_lines) # [[[316, 720, 484, 432]], [[1009,720, 718, 432]]]
    return lane_lines  


def make_points(frame, line):
    #print(line)
    height, width, _ = frame.shape 
    slope, intercept = line
    y1 = height
    y2 = int(y1 * 2 / 3) 
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]


def detect_lane(frame):
    edges = detect_edges(frame)
    cropped_edges = region_of_interest(edges)
    cv2.imshow("cedges", cropped_edges)
    line_segments = detect_line_segments(cropped_edges)
    lane_lines = average_slope_intercept(frame, line_segments)  
    return lane_lines


def display_lines(frame, lines):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2),(0, 255, 0), 4)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image

def car_detect(frame,lane_lines):
    car_mask = np.zeros_like(frame)
    
    match_mask_color = (0,255,255)

    cv2.fillPoly(car_mask, [np.array([(320, 50), 
                                    (30,360),(620,360)],np.int32)], match_mask_color)
    car_masked_image = cv2.bitwise_and(frame, car_mask)  

    cv2.imshow("car_masked_image", car_masked_image)                              
    h, w = frame.shape[:2]

    class_ids = []
    confidences = []
    boxes = []

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.
    getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(car_masked_image, 1/255., (320, 320), swapRB=True)
    net.setInput(blob)
    outs = net.forward(output_layers)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confThreshold:
                cx = int(detection[0] * w)
                cy = int(detection[1] * h)
                bw = int(detection[2] * w)
                bh = int(detection[3] * h)
                sx = int(cx - bw / 2)
                sy = int(cy - bh / 2)
                boxes.append([sx, sy, bw, bh])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))
                
                distance = int((500 * (h - (sy + bh))) / bw)
                if len(lane_lines) >= 1 :
                    #if lane_lines[0][0][0] < 60 or lane_lines[0][0][0] > 550:
                    if distance < 250:
                        
                        print(distance)
                        str = 'distance'
                        cv2.putText(frame, str, (350, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, 
                            (0, 255, 255), thickness=2)
                        winsound.Beep(freq, dur)    

                        
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    
    for i in indices:
        i = i[0]
        sx, sy, bw, bh = boxes[i]
        label = f'{classes[class_ids[i]]}: {confidences[i]:.2}'
        color = colors[class_ids[i]]
        cv2.rectangle(frame, (sx, sy, bw, bh), color, 2)
        cv2.putText(frame, label, (sx, sy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 1, cv2.LINE_AA)



model = 'yolo_v3/yolov3-tiny.weights'
config = 'yolo_v3/yolov3-tiny.cfg'
class_labels = 'yolo_v3/coco.names'
confThreshold = 0.5

nmsThreshold = 0.4

cap = cv2.VideoCapture(1)

classes = []
with open(class_labels, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

colors = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNet(model, config)

if net.empty():
    print('Net open failed!')
    sys.exit()
    
prevTime = 0
# w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))            
# h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'DIVX') # *'DIVX' == 'D', 'I', 'V', 'X'     
delay = round(1000 / fps)

out = cv2.VideoWriter('output3.avi', fourcc, fps, (w, h))

delay = round(1000 / fps)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print('original size: %d, %d' % (width, height))
 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
 
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print('changed size: %d, %d' % (width, height))

freq = 2500 # Set frequency To 2500 Hertz
dur = 120 # Set duration To 1000 ms == 1 second

if not cap.isOpened()  :
    print("video open failed !")
    cap.release()
    sys.exit() 


while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:  #real time
         break      #real time
    
    
    curTime = time.time()
    sec = curTime - prevTime
    prevTime = curTime
    fps = 1/(sec)
    str = "FPS : %0.1f" % fps
    cv2.putText(frame, str, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    
    lane_lines = detect_lane(frame)

    car_detect(frame, lane_lines)
    lane_lines_image = display_lines(frame, lane_lines)
    cv2.imshow("lane lines", lane_lines_image)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
