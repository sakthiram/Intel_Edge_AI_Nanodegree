# Computer Pointer Controller
This project is an application that controls the computer pointer with the use of human eye gaze direction. It supports input from video file and camera video stream. 

## Demo
To run the application use the following command
```bash
$ python3 main.py -f model/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -fl model/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml -hp model/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml -g model/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml -i demo.avi -flags fd, fld, hp, ge
```