import cv2
from input_feeder import InputFeeder
from fd import Model_FD
from fl import Model_FL
from ge import Model_GE
from hp import Model_HP
from mouse_controller import MouseController
import  numpy as np
from argparse import ArgumentParser
import time
import logging

parser = ArgumentParser()
logger = logging.getLogger()

parser.add_argument("-fd", "--fd_model", required=False, type=str,
                    default="model/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml",
                    help=" Path to .xml file of Face Detection model.")
parser.add_argument("-fl", "--fl_model", required=False, type=str,
                    default="model/intel/landmarks-regression-retail-0009/FP16-INT8/landmarks-regression-retail-0009.xml",
                    help=" Path to .xml file of Facial Landmark Detection model.")
parser.add_argument("-hp", "--hp_model", required=False, type=str,
                    default="model/intel/head-pose-estimation-adas-0001/FP16-INT8/head-pose-estimation-adas-0001.xml",
                    help=" Path to .xml file of Head Pose Estimation model.")
parser.add_argument("-ge", "--ge_model", required=False, type=str,
                    default="model/intel/gaze-estimation-adas-0002/FP16-INT8/gaze-estimation-adas-0002.xml",
                    help=" Path to .xml file of Gaze Estimation model.")
parser.add_argument("-i", "--input", required=True, type=str,
                    help=" Path to video file or enter cam for webcam")
parser.add_argument("-flags", "--displayFlags", required=False, nargs='+',
                    default=[],
                    help="Specify the flags from fd, fl, hp, ge like -flags fd hp fl (Seperated by space)")
parser.add_argument("-d", "--device", type=str, default="CPU",
                    help="Specify the target device to run on: "
                         "CPU, GPU, FPGA or MYRIAD.")

args = parser.parse_args()

# fd_model = "model/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml"
# fl_model = "model/intel/landmarks-regression-retail-0009/FP16-INT8/landmarks-regression-retail-0009.xml"
# hp_model = "model/intel/head-pose-estimation-adas-0001/FP16-INT8/head-pose-estimation-adas-0001.xml"
# ge_model = "model/intel/gaze-estimation-adas-0002/FP16-INT8/gaze-estimation-adas-0002.xml"
fd_model = args.fd_model
fl_model = args.fl_model
hp_model = args.hp_model
ge_model = args.ge_model

# device = "CPU"
device = args.device
cpu_extension = None
if args.input.lower() == "cam":
    inputFeeder = InputFeeder("cam")
else:
    inputFeeder = InputFeeder("video", args.input)

mfd = Model_FD(fd_model, device, cpu_extension)
mfl = Model_FL(fl_model, device, cpu_extension)
mge = Model_GE(ge_model, device, cpu_extension)
mhp = Model_HP(hp_model, device, cpu_extension)

mc = MouseController('medium','fast')

inputFeeder.load_data()

start = time.time()

mfd.load_model()
mfl.load_model()
mge.load_model()
mhp.load_model()

model_loading_time = time.time() - start
logger.debug("Loading done.")

prob_threshold = 0.6
device = "CPU"
frame_count = 0
inference_time = 0

for ret, frame in inputFeeder.next_batch():
    if not ret:
        break

    if frame is not None:
        key = cv2.waitKey(3)
        frame_count += 1

        start = time.time()

        ##################
        # Face detection #
        ##################

        # Preprocess Input API
        img_processed = mfd.preprocess_input(frame.copy())
        # Predict API
        outputs = mfd.predict(img_processed)
        # Preprocess Output API
        coords = mfd.preprocess_output(outputs, prob_threshold)

        h = frame.shape[0]
        w = frame.shape[1]
        face_coords = coords[0]* np.array([w, h, w, h])
        face_coords = face_coords.astype(np.int32)

        croppedFace = []

        if len(coords) != 0:
            # Get cropped face API
            cropped_face = mfd.get_cropped_face(coords, frame.copy())
        else:
            logger.error("No face detected")
            continue

        # Next the cropped face (output from face detection model) is fed into next stages (head pose estimation model).
        ########################
        # Head pose estimation #
        ########################

        outputs = mhp.predict(cropped_face.copy())
        hp_out = mhp.preprocess_output(outputs)

        ##################
        # Face landmarks #
        ##################
        outputs = mfl.predict(cropped_face.copy())
        coords = mfl.preprocess_output(outputs)

        h = cropped_face.shape[0]
        w = cropped_face.shape[1]
        coords = coords* np.array([w, h, w, h])
        coords = coords.astype(np.int32)

        # Fetch Eye coordinates from cropped face using the landmarks coordinates
        eye_coords, le, re = mfl.fetch_eye_coords(cropped_face.copy(), coords)

        # Next use the head pose output (hp_out) & eye coords (output from face landmarks stage) to determine gaze estimation
        ###################
        # Gaze estimation #
        ###################

        outputs = mge.predict(le, re, hp_out)
        mouse_coords, gaze_vec = mge.preprocess_output(outputs, hp_out)

        inference_time = inference_time + time.time() - start

        # Now that we have the mouse coords based on new eye location and head pose, we can use the pyautogui to move the mouse pointer
        if frame_count%5 == 0:
            mc.move(mouse_coords[0], mouse_coords[1])


        # Visualization windows based on flags
        if (not len(args.displayFlags) == 0):

            if 'fd' in args.displayFlags:
                preview_window1 = frame.copy()
                cv2.rectangle(preview_window1, (face_coords[0], face_coords[1]), (face_coords[2], face_coords[3]), (0, 255, 0), 3)
                cv2.imshow('FD', preview_window1)

            if 'fl' in args.displayFlags:
                preview_window2 = cropped_face.copy()
                cv2.rectangle(preview_window2, (eye_coords[0][0] - 10, eye_coords[0][1] - 10), (eye_coords[0][2] + 10, eye_coords[0][3] + 10), (0,255,0), 3)
                cv2.rectangle(preview_window2, (eye_coords[1][0] - 10, eye_coords[1][1] - 10), (eye_coords[1][2] + 10, eye_coords[1][3] + 10), (0,255,0), 3)
                cv2.imshow('FL', preview_window2)

            if 'hp' in args.displayFlags:
                preview_window3 = frame.copy()
                cv2.putText(
                    preview_window3,
                    "Pose Angles: yaw:{:.2f} | pitch:{:.2f} | roll:{:.2f}".format(hp_out[0], hp_out[1], hp_out[2]), (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA
                )
                cv2.imshow('HP', preview_window3)

            if 'ge' in args.displayFlags:
                preview_window4 = cropped_face.copy()

                x, y, w = int(gaze_vec[0] * 12), int(gaze_vec[1] * 12), 160

                left_c = cv2.line(le.copy(), (x - w, y - w), (x + w, y + w), (255, 0, 255), 2)
                cv2.line(le, (x - w, y + w), (x + w, y - w), (255, 0, 255), 2)

                right_c = cv2.line(re.copy(), (x - w, y - w), (x + w, y + w), (255, 0, 255), 2)
                cv2.line(re, (x - w, y + w), (x + w, y - w), (255, 0, 255), 2)

                preview_window4[eye_coords[0][1]:eye_coords[0][3], eye_coords[0][0]:eye_coords[0][2]] = left_c
                preview_window4[eye_coords[1][1]:eye_coords[1][3], eye_coords[1][0]:eye_coords[1][2]] = right_c
                cv2.imshow('GE', preview_window4)

fps = frame_count / inference_time
logger.debug("Video ended.")
print("Loading time: " + str(model_loading_time) + " s")
print("Average inference time: " + str(inference_time/frame_count) + " s")
print("FPS : ", format(fps/5))


cv2.destroyAllWindows()
inputFeeder.close()

