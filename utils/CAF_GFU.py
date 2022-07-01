import tensorflow as tf

class GLU(tf.keras.layers.Layer):
    """
        Custom activation function GLU for tensorflow 
    """
    def __init__(self, bias=True, dim=-1, **kwargs):
        super(GLU, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim
        self.dense = tf.keras.layers.Dense(2, use_bias=bias)

    def call(self, x):
        out, gate = tf.split(x, 2, self.dim)
        gate = tf.sigmoid(gate)
        x = tf.multiply(out, gate)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "bias": self.bias,
            "dim": self.dim,
            "dense": self.dense,
        })
        return config
