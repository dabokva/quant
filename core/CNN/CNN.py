import tensorflow as tf
from tensorflow.keras import layers


class CNN_Predecoder(tf.keras.Model):

    def __init__(self, mask_alpha=1.0):
        super().__init__(name='cnn_predecoder')

        self.mask_alpha = mask_alpha

    def build(self, input_shape):
        self.spatial_conv = layers.Conv3D(filters=32, kernel_size=(1, 3, 3), padding='same')
        self.activation1 = layers.ReLU()
        self.temporal_conv = layers.Conv3D(filters=32, kernel_size=(3, 1, 1), padding='same')
        self.activation2 = layers.ReLU()
        self.downsample_conv = layers.Conv3D(filters=48, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')
        self.activation3 = layers.ReLU()
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(units=32, activation='relu')
        self.dense2 = layers.Dense(units=16, activation='relu')
        self.output_layer = layers.Dense(units=1, activation='sigmoid', dtype=tf.float32, name='logical_flip_probability')
        super().build(input_shape)

    def call(self, inputs, training=False):
        events = inputs[..., 0:1]

        mask = tf.cast(inputs[..., 1:2], events.dtype)

        events = events * mask
        mask_feature = self.mask_alpha * mask

        x = tf.concat([events, mask_feature], axis=-1)

        x = self.spatial_conv(x)
        x = self.activation1(x)

        x = self.temporal_conv(x)
        x = self.activation2(x)

        x = self.downsample_conv(x)
        x = self.activation3(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return self.output_layer(x)

    def train_step(self, data):
        inputs, labels, sample_weight = (tf.keras.utils.unpack_x_y_sample_weight(data))

        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)

            loss = self.compute_loss(
                x=inputs,
                y=labels,
                y_pred=predictions,
                sample_weight=sample_weight,
            )

        gradients = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(loss)
            else:
                metric.update_state(
                    labels,
                    predictions,
                    sample_weight=sample_weight,
                )

        return self._metric_results()

    def test_step(self, data):
        inputs, labels, sample_weight = (tf.keras.utils.unpack_x_y_sample_weight(data))

        predictions = self(inputs, training=False)

        loss = self.compute_loss(
            x=inputs,
            y=labels,
            y_pred=predictions,
            sample_weight=sample_weight,
        )

        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(loss)
            else:
                metric.update_state(
                    labels,
                    predictions,
                    sample_weight=sample_weight,
                )

        return self._metric_results()

    def _metric_results(self):
        results = {}

        for metric in self.metrics:
            result = metric.result()

            if isinstance(result, dict):
                results.update(result)
            else:
                results[metric.name] = result

        return results