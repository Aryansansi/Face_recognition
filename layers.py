import tensorflow as tf
from tensorflow.keras.layers import Layer  # type: ignore

# Custom L1 Distance Layer
class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
       
    # Magic happens here - similarity calculation
    def call(self, inputs):
        input_embedding, validation_embedding = inputs
        return tf.math.abs(input_embedding - validation_embedding)
