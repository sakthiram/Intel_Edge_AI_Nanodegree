# Computer Pointer Controller

## Project Set Up and Installation
*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.
1) Unzip the package, install pip packages in requirements.txt
2) Run `python src/computer_pointer_controller.py -i cam` to run using webcam
3) Run `python src/computer_pointer_controller.py -i demo.avi` to run using the demo video

## Description
The project involves using head pose & gaze to move the mouse pointer using pyautogui.
Using input feeder, we will fetch video from cam or video file based on user input.
- First step is detect a face from the frame, and crop it.
- Next sending the cropped face to a head pose & face landmark models will help us with the head pose estimates and eye coordinates respectively.
- Now using the output of gaze estimation model and head pose angle, we can find the x & y movements for the mouse pointer.  
Based on the display flags the intermediate outputs can be seen on the visualization windows.
Once the video has ended, the input feeder and the visulization windows are closed.

## Directory Structure

|
|--demo.avi
|--model
    |--intel
        |--face-detection-adas-binary-0001
        |--gaze-estimation-adas-0002
        |--head-pose-estimation-adas-0001
        |--landmarks-regression-retail-0009
|--src
    |--fd.py
    |--fl.py
    |--ge.py
    |--input_feeder.py
    |--computer_pointer_controller.py
    |--mouse_controller.py
|--README.md
|--requirements.txt

The model files are under the model directory and all the source files are inside src directory.
fd.py => APIs for preprocessing inputs/outputs, load model & run model APIs for face detection model, and API to fetch cropped face.
fl.py => APIs for preprocessing inputs/outputs, load model & run model APIs for face landmarks model, and API to fetch eye coordinates.
hp.py => APIs for preprocessing inputs/outputs, load model & run model APIs for head pose model.
ge.py => APIs for preprocessing inputs/outputs, load model & run model APIs for gaze estimation model.

## Command line 
usage: computer_pointer_controller.py [-h] [-fd FD_MODEL] [-fl FL_MODEL]
                                      [-hp HP_MODEL] [-ge GE_MODEL] -i INPUT
                                      [-flags DISPLAYFLAGS [DISPLAYFLAGS ...]]
                                      [-d DEVICE]

optional arguments:
  -h, --help            show this help message and exit
  -fd FD_MODEL, --fd_model FD_MODEL
                        Path to .xml file of Face Detection model.
  -fl FL_MODEL, --fl_model FL_MODEL
                        Path to .xml file of Facial Landmark Detection model.
  -hp HP_MODEL, --hp_model HP_MODEL
                        Path to .xml file of Head Pose Estimation model.
  -ge GE_MODEL, --ge_model GE_MODEL
                        Path to .xml file of Gaze Estimation model.
  -i INPUT, --input INPUT
                        Path to video file or enter cam for webcam
  -flags DISPLAYFLAGS [DISPLAYFLAGS ...], --displayFlags DISPLAYFLAGS [DISPLAYFLAGS ...]
                        Specify the flags from fd, fl, hp, ge like -flags fd
                        hp fl (Seperated by space)
  -d DEVICE, --device DEVICE
                        Specify the target device to run on: CPU, GPU, FPGA or
                        MYRIAD.


## Benchmarks

FP32
```
Loading time: 0.9460844993591309 s
Average inference time: 0.0314764006663177 s
FPS :  6.353966646955801
```
FP16
```
Loading time: 0.9600083827972412 s
Average inference time: 0.030927011522196107 s
FPS :  6.46683886208861
```
FP16-INT8
```
Loading time: 1.1910083293914795 s
Average inference time: 0.029303655786029364 s
FPS :  6.825086994618291
```

## Results
As can be seen from the benchmark, FP16-INT8 is the fastest, and FP32 is slowest. The reason for faster inference time for INT8 & FP16 is because of less number of bits per weight (less precision), so the operations are faster.

## Stand Out Suggestions
- Using click when closing eye lids.

### Edge Cases
- In some poses, face is not detected, so skipping those frames from feeding to other models, since we need to send the cropped part to others.
