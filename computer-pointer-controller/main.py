import cv2
import os
import logging
import numpy as np
import time
from argparse import ArgumentParser
from input_feeder import InputFeeder
from face_detection import Model_Face_Detection
from facial_landmarks_detection import Model_Facial_Landmarks_Detection
from gaze_estimation import Model_Gaze_Estimation
from head_pose_estimation import Model_Head_Pose_Estimation
from mouse_controller import MouseController


def build_argparser():
    parser = ArgumentParser()

    parser.add_argument("-f", "--facedetectionmodel", required=True, type=str,
                        help=" Path to .xml file of Face Detection model.")
    parser.add_argument("-fl", "--faciallandmarkmodel", required=True, type=str,
                        help=" Path to .xml file of Facial Landmark Detection model.")
    parser.add_argument("-hp", "--headposemodel", required=True, type=str,
                        help=" Path to .xml file of Head Pose Estimation model.")
    parser.add_argument("-g", "--gazeestimationmodel", required=True, type=str,
                        help=" Path to .xml file of Gaze Estimation model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help=" Path to video file or enter cam for webcam")
    parser.add_argument("-flags", "--previewFlags", required=False, nargs='+',
                        default=[],
                        help="Specify the flags from fd, fld, hp, ge like --flags fd hp fld (Seperated by space)"
                             "for see the visualization of different model outputs of each frame," 
                             "fd for Face Detection, fld for Facial Landmark Detection"
                             "hp for Head Pose Estimation, ge for Gaze Estimation." )
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="path of extensions if any layers is incompatible with  hardware")
    parser.add_argument("-prob", "--prob_threshold", required=False, type=float,
                        default=0.6,
                        help="Probability threshold for model to identify the face .")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to run on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "(CPU by default)")
    
    return parser


def main():
    args = build_argparser().parse_args()
    previewFlags = args.previewFlags
    
    logger = logging.getLogger()
    inputFile = args.input
    inputFeeder = None

    if inputFile.lower() == "cam":
        inputFeeder = InputFeeder("cam")
    else:
        if not os.path.isfile(inputFile):
            logger.error("Unable to find input file")
            exit(1)
        
        inputFeeder = InputFeeder("video",inputFile)

    start_loading = time.time()

    mfd = Model_Face_Detection(args.facedetectionmodel, args.device, args.cpu_extension)
    mfld = Model_Facial_Landmarks_Detection(args.faciallandmarkmodel, args.device, args.cpu_extension)
    mge = Model_Gaze_Estimation(args.gazeestimationmodel, args.device, args.cpu_extension)
    mhpe = Model_Head_Pose_Estimation(args.headposemodel, args.device, args.cpu_extension)
    
    mc = MouseController('medium','fast')

    inputFeeder.load_data()

    mfd.load_model()
    mfld.load_model()
    mge.load_model()
    mhpe.load_model()

    model_loading_time = time.time() - start_loading
    
    counter = 0
    frame_count = 0
    inference_time = 0
    start_inf_time = time.time()
    for ret, frame in inputFeeder.next_batch():
        if not ret:
            break;

        if frame is not None:
            frame_count += 1
            if frame_count%5 == 0:
                cv2.imshow('video', cv2.resize(frame, (500, 500)))
        
            key = cv2.waitKey(60)
            start_inference = time.time()

            croppedFace, face_coords = mfd.predict(frame.copy(), args.prob_threshold)
            if type(croppedFace) == int:
                logger.error("No face detected.")
                if key == 27:
                    break

                continue
            
            hp_out = mhpe.predict(croppedFace.copy())
            
            left_eye, right_eye, eye_coords = mfld.predict(croppedFace.copy())
            
            new_mouse_coord, gaze_vector = mge.predict(left_eye, right_eye, hp_out)
            
            stop_inference = time.time()
            inference_time = inference_time + stop_inference - start_inference
            counter = counter + 1
            if (not len(previewFlags) == 0):
                preview_window = frame.copy()
                
                if 'fd' in previewFlags:
                    if len(previewFlags) != 1:
                        preview_window = croppedFace
                    else:
                        cv2.rectangle(preview_window, (face_coords[0], face_coords[1]), (face_coords[2], face_coords[3]), (0, 150, 0), 3)

                if 'fld' in previewFlags:
                    if not 'fd' in previewFlags:
                        preview_window = croppedFace.copy()

                    cv2.rectangle(preview_window, (eye_coords[0][0] - 10, eye_coords[0][1] - 10), (eye_coords[0][2] + 10, eye_coords[0][3] + 10), (0,255,0), 3)
                    cv2.rectangle(preview_window, (eye_coords[1][0] - 10, eye_coords[1][1] - 10), (eye_coords[1][2] + 10, eye_coords[1][3] + 10), (0,255,0), 3)
                    
                if 'hp' in previewFlags:
                    cv2.putText(
                        preview_window, 
                        "Pose Angles: yaw:{:.2f} | pitch:{:.2f} | roll:{:.2f}".format(hp_out[0], hp_out[1], hp_out[2]), (50, 50), 
                        cv2.FONT_HERSHEY_COMPLEX, 
                        1, 
                        (0, 255, 0), 
                        1, 
                        cv2.LINE_AA
                    )

                if 'ge' in previewFlags:
                    if not 'fd' in previewFlags:
                        preview_window = croppedFace.copy()

                    x, y, w = int(gaze_vector[0] * 12), int(gaze_vector[1] * 12), 160
                    
                    le = cv2.line(left_eye.copy(), (x - w, y - w), (x + w, y + w), (255, 0, 255), 2)
                    cv2.line(le, (x - w, y + w), (x + w, y - w), (255, 0, 255), 2)
                    
                    re = cv2.line(right_eye.copy(), (x - w, y - w), (x + w, y + w), (255, 0, 255), 2)
                    cv2.line(re, (x - w, y + w), (x + w, y - w), (255, 0, 255), 2)
                    
                    preview_window[eye_coords[0][1]:eye_coords[0][3], eye_coords[0][0]:eye_coords[0][2]] = le
                    preview_window[eye_coords[1][1]:eye_coords[1][3], eye_coords[1][0]:eye_coords[1][2]] = re
            
            if len(previewFlags) != 0:
                img_hor = np.hstack((cv2.resize(frame, (500, 500)), cv2.resize(preview_window, (500, 500))))
            else:
                img_hor = cv2.resize(frame, (500, 500))

            cv2.imshow('Visualization', img_hor)

            if frame_count%5 == 0:
                mc.move(new_mouse_coord[0], new_mouse_coord[1])    
            
            if key == 27:
                break

    fps = frame_count / inference_time

    logger.error("video ended...")
    logger.error("Total loading time of the models: " + str(model_loading_time) + " s")
    logger.error("total inference time {} seconds".format(inference_time))
    logger.error("Average inference time: " + str(inference_time/frame_count) + " s")
    logger.error("fps {} frame/second".format(fps/5))

    cv2.destroyAllWindows()
    inputFeeder.close()
     
    
if __name__ == '__main__':
    main()
