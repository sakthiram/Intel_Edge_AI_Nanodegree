"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import numpy as np

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network = Network()
    # Load the network to IE plugin to get shape of input layer
    n, c, h, w = infer_network.load_model(args.model, args.device, 1, 1, 2, args.cpu_extension)[1]

    ### TODO: Handle the input stream ###
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print("\nUnable to open video file... Exiting...\n")
        sys.exit(0)

    fps = cap.get(cv2.CAP_PROP_FPS)
#     if args.flag == "async":
#         is_async_mode = True
#         print('Application running in async mode')
#     else:
#         is_async_mode = False
#         print('Application running in sync mode')

    is_async_mode = True

    print("To stop the execution press Esc button")
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_count = 1
    accumulated_image = np.zeros((initial_h, initial_w), np.uint8)
    mog = cv2.createBackgroundSubtractorMOG2()
    ret, frame = cap.read()
    cur_request_id = 0
    next_request_id = 1
    total_count = 0
    people_present = False
    no_people_frames = 0
    inference_time = 0
    ### TODO: Loop until stream is over ###
    while cap.isOpened():

        ### TODO: Read from the video capture ###
        ret, next_frame = cap.read()
        start_time = time.time()

        if not ret:
            break
        frame_count = frame_count + 1
        ### TODO: Pre-process the image as needed ###
        in_frame = cv2.resize(next_frame, (w, h))
        # Change data layout from HWC to CHW
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape((n, c, h, w))

        ### TODO: Start asynchronous inference for specified request ###
        inf_start = time.time()
        if is_async_mode:
            infer_network.exec_net(next_request_id, in_frame)
        else:
            infer_network.exec_net(cur_request_id, in_frame)

        ### TODO: Wait for the result ###
        if infer_network.wait(cur_request_id) == 0:
            det_time = time.time() - inf_start
            people_count = 0

            # Converting to Grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Remove the background
            fgbgmask = mog.apply(gray)
            # Thresholding the image
            thresh = 2
            max_value = 2
            threshold_image = cv2.threshold(fgbgmask, thresh, max_value,
                                              cv2.THRESH_BINARY)[1]
            # Adding to the accumulated image
            accumulated_image = cv2.add(threshold_image, accumulated_image)
            colormap_image = cv2.applyColorMap(accumulated_image, cv2.COLORMAP_HOT)

            ### TODO: Get the results of the inference request ###
            res = infer_network.get_output(cur_request_id)

            ### TODO: Extract any desired stats from the results ###
            for obj in res[0][0]:
                # Draw only objects when probability more than specified threshold
                if obj[2] > args.prob_threshold:
                    xmin = int(obj[3] * initial_w)
                    ymin = int(obj[4] * initial_h)
                    xmax = int(obj[5] * initial_w)
                    ymax = int(obj[6] * initial_h)
                    class_id = int(obj[1])
                    # Draw bounding box
                    color = (min(class_id * 12.5, 255), min(class_id * 7, 255),
                    min(class_id * 5, 255))
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    people_count = people_count + 1
                    people_present = True
                    no_people_frames = 0
                else:
                    no_people_frames += 1
                    if people_present and no_people_frames > 1000:
                        total_count = total_count + 1
                        people_present = False
                        no_people_frames = 0

            people_count_message = "People Count : " + str(people_count)
            cv2.putText(frame, people_count_message, (15, 65), cv2.FONT_HERSHEY_COMPLEX, 1,
                     (0, 0, 0), 2)

#             cv2.imshow("Detection Results", frame)

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            client.publish("person", json.dumps({"count": people_count, "total": total_count}))
            client.publish("person/duration", json.dumps({"duration": "00:01"}))

        ### TODO: Send the frame to the FFMPEG server ###
#         sys.stdout.buffer.write(frame)  
#         sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###

        frame = next_frame
        if is_async_mode:
            cur_request_id, next_request_id = next_request_id, cur_request_id
#         print("FPS : {}".format(1/(time.time() - start_time)))
        inference_time = inference_time + time.time() - start_time
        # Frames are read at an interval of 1 millisecond
        key = cv2.waitKey(1)
        if key == 27:
            break

    print("Avg inference time: ", inference_time/frame_count)
    cap.release()
    cv2.destroyAllWindows()
    infer_network.clean()
    client.disconnect()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
