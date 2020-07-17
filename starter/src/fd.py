'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import numpy as np
from openvino.inference_engine import IECore

class Model_FD:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_name = model_name
        self.model_weights = self.model_name.split('.')[0]+'.bin'
        self.device = device
        self.extensions = extensions
        self.network = None
        self.exec_net = None
        self.input_name = None
        self.input_shape = None
        self.output_name = None
        self.output_shape = None

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        plugin = IECore()
        self.network = plugin.read_network(model = self.model_name, weights = self.model_weights)
        self.exec_net = plugin.load_network(network = self.network, device_name = self.device, num_requests = 1)
        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        self.output_name = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_name].shape

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        outputs = self.exec_net.infer({self.input_name : image})
        return outputs

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        image_resized = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        img_processed = np.transpose(np.expand_dims(image_resized, axis = 0), (0, 3, 1, 2))

        return img_processed

    def preprocess_output(self, outputs, prob_threshold):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        coords = []
        outputs = outputs[self.output_name][0][0]
        for output in outputs:
            confidence = output[2]

            if confidence >= prob_threshold:
                xmin = output[3]
                ymin = output[4]
                xmax = output[5]
                ymax = output[6]

                coords.append([xmin, ymin, xmax, ymax])

        return coords

    def get_cropped_face(self, coords, frame):
        coords = coords[0]
        h = frame.shape[0]
        w = frame.shape[1]
        coords = coords* np.array([w, h, w, h])
        coords = coords.astype(np.int32)

        cropped_face = frame[coords[1]:coords[3], coords[0]:coords[2]]
        return cropped_face
