import tensorflow as tf


class EyeInitializer(tf.keras.initializers.Initializer):
    def __init__(self, mean=0, stddev=0.1, eye_scaling=1):
        """
        Identity matrix + random normal matrix identified by mean and stddev. Using very low stddev and mean=0 will
        lead to identity matrix

        :param mean: Mean of the random normal matrix
        :param stddev: Standard deviation of the random normal matrix
        """
        self.mean = mean
        self.stddev = stddev
        self.eye_scaling=eye_scaling

    def __call__(self, shape, dtype=None, **kwargs):
        nrows = shape[-2]
        ncols = shape[-1]
        if len(shape) != 2:
            raise RuntimeError("Error in initializing eye initializer. Weights have to be 2 dimensional tensor, that is, a matrix.")
        x = tf.random.normal(
            shape, mean=self.mean, stddev=self.stddev, dtype=dtype)
        y = tf.eye(num_rows=nrows, num_columns=ncols, batch_shape=None, dtype=dtype) * self.eye_scaling

        return x + y

    def get_config(self):  # To support serialization
        return {"mean": self.mean, "stddev": self.stddev, "eye_scaling":self.eye_scaling}

