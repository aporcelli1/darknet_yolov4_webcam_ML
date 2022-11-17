import darknet
import argparse
import cv2

from collections import deque
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'      #select the available cuda gpu


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)   #open the webcam with default option
    parser.add_argument("--width", help='cap width', type=int, default=960)     #default size of the webcam yolo defaul 608x608
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument("--weights", default="yolov4.weights", help="yolo weights path")    #weights
    parser.add_argument("--dont_show", action='store_true', help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true', help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg", help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data", help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25, help="remove detections with confidence below this value")

    args = parser.parse_args()
    print(args)

    return args


#class for the fps calculation

class CvFpsCalc(object):
    def __init__(self, buffer_len=1):
        self._start_tick = cv2.getTickCount()
        self._freq = 1000.0 / cv2.getTickFrequency()
        self._difftimes = deque(maxlen=buffer_len)

    def get(self):
        current_tick = cv2.getTickCount()
        different_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick

        self._difftimes.append(different_time)

        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        fps_rounded = round(fps, 2)

        return fps_rounded

def draw_fps(image,fps):
    '''
    Draw the current frame rate in the image
    '''
    cv2.putText(image, "FPS:" + str(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, "FPS:" + str(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 1, cv2.LINE_AA)

def load_image(frame,network):
    '''
    Initialize a image workspace with size (608,608,3)
    (darknet_width, darknet_height,3)
    Inside that workspace it gonna be load the data:
        Current image acquired by the webcam
        resulting detection with confidence, label and bbox
        Current frame rate
    '''

    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)

    img_for_detection = darknet.make_image(darknet_width, darknet_height, 3)  # create an empty image space with size (608,608,3)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (darknet_width, darknet_height), interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(img_for_detection, img_resized.tobytes())  #load the current image acquired /
    # from the webcam into the image workspace

    return img_for_detection



def draw_bbox(detections, image, colors,network):
    '''
    Alternative function to the one provided by the darknet->darknet.draw_boxes
    darknet.draw_boxes(detections,frame,class_colors)
    Advantages, could draw the bbox with different webcam width and height

    return the image with the bbox, the label and the confidence
    '''

    # get image ratios to convert bounding boxes to proper size
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)
    img_height, img_width, _ = image.shape
    width_ratio = img_width / darknet_width
    height_ratio = img_height / darknet_height

    for label, confidence, bbox in detections:
        left, top, right, bottom = darknet.bbox2points(bbox)
        left, top, right, bottom = int(left * width_ratio), int(top * height_ratio), int(right * width_ratio), int(
            bottom * height_ratio)
        cv2.rectangle(image, (left, top), (right, bottom-10), colors[label], 2)
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                    (left, top - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.7,
                    colors[label], 2)
    return image



def main():

    #initiation and creation args object with attribute #device; #width; #height
    args = get_args()

    #Load Network
    network, class_names, class_colors = darknet.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=1
        )


    cvFpsCalc = CvFpsCalc(buffer_len=10)  # buffer for the fps management

    # we create the video capture object cap
    cap = cv2.VideoCapture(args.device) #args.device=0->webcam input
    if not cap.isOpened():
        raise IOError("We cannot open webcam")


    cap_width = 2040 #args.width
    cap_height = 2040 #args.height

    # cast and normalization
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    while True:

        ret, frame=cap.read()
        if not ret:
            break

        fps = cvFpsCalc.get()  # counting the fps

        img_for_detection=load_image(frame,network)    #create the image for the detections

        detections = darknet.detect_image(network, class_names, img_for_detection)   #return detections list [('label','confidence',(bbox)),...,...]
        darknet.free_image(img_for_detection)     #release the space for the new img_for_detection

        print('FPS',fps,'/n','DETECTIONS',detections)

        #det_image=darknet.draw_boxes(detections,frame,class_colors)  #use the default class of darknet
        det_image = draw_bbox(detections, frame, class_colors,network)
        draw_fps(det_image,fps)


        # show us frame with detection
        cv2.imshow("Web cam input", det_image)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break



if __name__ == '__main__':
    main()

