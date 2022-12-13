#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='model/keypoint_classifier/keypoint_classifier.tflite',
        num_threads=1,
    ):  # model_path is Location of tflite model file

        # Load TFLite model and allocate tensors.    
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)
        # Allocate tensors
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(
        self,
        pair_comb,
    ):
        input_details_tensor_index_1 = self.input_details[0]['index']
        input_details_tensor_index_2 = self.input_details[1]['index']

        # Create input tensor out of landmark
        self.interpreter.set_tensor(input_details_tensor_index_1, pair_comb[0])
        self.interpreter.set_tensor(input_details_tensor_index_2, pair_comb[1])

        

        # Run inference
        self.interpreter.invoke()


        # output_details[0]['index'] = the index which provides the input
        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)
        result_index = np.squeeze(result)

        #result_index = np.argmax(np.squeeze(result))

        return result_index
