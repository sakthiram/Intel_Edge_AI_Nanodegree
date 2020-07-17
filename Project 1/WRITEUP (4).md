# Project Write-Up

## Explaining Custom Layers
I used the SSD MobileNet V2 model from TensorFlow as such. No updates to the layers was necessary.
Link: http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz.
The IR models (frozen_inference_graph.bin & frozen_inference_graph.xml files) generated can be found inside the intel/object-det-tf folder.

## Comparing Model Performance

- Without openvino: with GPU Acceleration enabled, the average inference time per frame was 50 ms.
- With openvino, using the IR converted ssd model (ssd coco mobilenet v2), the average inference time per frame comes to 85 ms.
- With openvino, using exsiting IR model (person-detection-retail-0013: FP16), the average inference time per frame comes to 63 ms.
- With openvino, using exsiting IR model (person-detection-retail-0013: FP32), the average inference time per frame comes to 63 ms.


## Steps for demo

With ssd coco v2 model after conversion to IR: python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m /home/workspace/intel/object-det-tf/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

With existing IR model from openvino: python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m /home/workspace/intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

Mosca server, ffmpeg server, and the webservice were run as mentioned in the problem statement.

Note: The total count value is correctly published on MQTT as seen from Mosca server, somehow web ui not showing correctly.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are maintaining social distance in a specified location, warning when more number of people are present per sq meter area, in retail queue management, etc.

Each of these use cases would be useful because the server can notify the respective subjects based on the stats obtained from the prediction outputs.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows:
- Lighting in real world scenario needs to be considered, especially if the edge device is supposed to work in night mode as well. Need to consider IR flash, which means the model we consider need to be able to work with grayscale images.
- Model accuracy depends on the usecase,depending on the criticality of it and also it impacts the decision on the hardware we choose and hence affecting the power/size factors.
- Camera focal length (mostly fixed for edge applicaitons) and field of view need to be considered again based on the usecase we choose, whether the objects pointed to the camera are close or far.  And the image size affects the memory available in the hardware we choose, and also the inference speed requirements. We could also rescale the image using an ISP before feeding into the processor to limit the memeory used.

## Model Research

- Dowloaded the ssd mobilenet model for object detection (label: human). 
- Converted to IR bin + xml.
- Used the IR model for inference, and tested using the mosca & ffmpef server.

Mobilenet with its faster inference speed, it can be used in real time.

I obtained the model from http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz.
I used this command for converting to IR `python intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json`
The IR conversion succeeded, and also I was able to fetch the outputs using the predict function as defined in the inference.py. 