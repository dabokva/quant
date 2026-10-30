import tensorflow as tf
from tensorflow.keras import layers


class GraphConvolution(layers.Layer):

    def __init__(self, units, activation=None, **kwargs):
        super().__init__(autocast=False, **kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        input_dimension = int(input_shape[-1])

        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_dimension, self.units),
            initializer='glorot_uniform',
            trainable=True,
        )

        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
        )

        super().build(input_shape)

    def call(self, node_features, edge_index, edge_weights):
        x = tf.matmul(tf.cast(node_features, self.compute_dtype),
                      tf.cast(self.kernel, self.compute_dtype))
        x = tf.cast(x, tf.float32)

        sources = edge_index[0]
        targets = edge_index[1]

        messages = tf.gather(x, sources, axis=1)
        messages *= tf.cast(edge_weights[None, :, None], tf.float32)
        messages = tf.transpose(messages, (1, 0, 2))

        x = tf.math.unsorted_segment_sum(
            data=messages,
            segment_ids=targets,
            num_segments=tf.shape(node_features)[1],
        )

        x = tf.transpose(x, (1, 0, 2))
        x += tf.cast(self.bias, tf.float32)

        if self.activation is not None:
            x = self.activation(x)

        return x


class GCNPredecoder(tf.keras.Model):

    def __init__(
        self,
        edge_index,
        edge_weights,
        hidden_units=96,
        number_of_gcn_layers=3,
        dense_units=128,
        dropout_rate=0.1,
        dense2_units=None,
        precision='float32',
    ):
        super().__init__(name='gcn_predecoder', dtype='float32')
        if precision not in ('float32', 'mixed_float16'):
            raise ValueError('Unsupported computation precision')
        self.precision = precision
        self.architecture = dict(hidden_units=hidden_units,
                                 number_of_gcn_layers=number_of_gcn_layers,
                                 dense_units=dense_units,
                                 dense2_units=dense_units // 2 if dense2_units is None else dense2_units,
                                 dropout_rate=dropout_rate)
        transform_policy = tf.keras.mixed_precision.Policy(precision)

        self.edge_index = tf.convert_to_tensor(edge_index, tf.int32)
        self.edge_weights = tf.convert_to_tensor(edge_weights, tf.float32)

        self.input_projection = layers.Dense(
            hidden_units,
            activation='relu',
            name='input_projection', dtype=transform_policy,
        )

        self.gcn_layers = [
            GraphConvolution(
                hidden_units,
                activation='relu',
                name=f'gcn_{index + 1}', dtype=transform_policy,
            )
            for index in range(number_of_gcn_layers)
        ]

        self.normalization_layers = [
            layers.LayerNormalization(name=f'norm_{index + 1}', dtype='float32')
            for index in range(number_of_gcn_layers)
        ]

        self.graph_dropout = layers.Dropout(dropout_rate, dtype='float32')
        self.attention_score = layers.Dense(1, name='attention_score', dtype='float32')

        self.dense1 = layers.Dense(
            dense_units,
            activation='relu',
            name='dense_1', dtype=transform_policy,
        )

        self.dense2 = layers.Dense(
            self.architecture['dense2_units'],
            activation='relu',
            name='dense_2', dtype=transform_policy,
        )

        self.output_dropout = layers.Dropout(dropout_rate, dtype='float32')

        self.output_layer = layers.Dense(
            1,
            activation='sigmoid',
            dtype=tf.float32,
            name='logical_flip_probability',
        )

        self.loss_function = tf.keras.losses.BinaryCrossentropy()
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')

        self.accuracy_tracker = tf.keras.metrics.BinaryAccuracy(
            name='accuracy'
        )

        self.auc_tracker = tf.keras.metrics.AUC(name='roc_auc')

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.accuracy_tracker,
            self.auc_tracker,
        ]

    def call(self, node_features, training=False):
        edge_index, edge_weights = self.edge_index, self.edge_weights
        if isinstance(node_features, dict):
            edge_index = tf.cast(node_features['edge_index'], tf.int32)
            edge_weights = tf.cast(node_features['edge_weights'], tf.float32)
            node_features = node_features['features']
        x = tf.cast(self.input_projection(node_features), tf.float32)

        for gcn_layer, normalization in zip(
            self.gcn_layers,
            self.normalization_layers,
        ):
            residual = x
            x = gcn_layer(x, edge_index, edge_weights)
            x = normalization(x + residual)
            x = self.graph_dropout(x, training=training)

        mean_pool = tf.reduce_mean(x, axis=1)
        max_pool = tf.reduce_max(x, axis=1)

        attention_weights = tf.nn.softmax(
            self.attention_score(x),
            axis=1,
        )

        attention_pool = tf.reduce_sum(
            x * attention_weights,
            axis=1,
        )

        x = tf.concat(
            [mean_pool, max_pool, attention_pool],
            axis=-1,
        )

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.output_dropout(x, training=training)

        return self.output_layer(x)

    def compute_loss_value(self, labels, predictions):
        loss = self.loss_function(labels, predictions)

        if self.losses:
            loss += tf.add_n(self.losses)

        return loss

    def update_metrics(self, labels, predictions, loss):
        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(labels, predictions)
        self.auc_tracker.update_state(labels, predictions)

    def metric_results(self):
        return {
            metric.name: metric.result()
            for metric in self.metrics
        }

    def train_step(self, data):
        node_features, labels = data
        labels = tf.cast(tf.reshape(labels, (-1, 1)), tf.float32)

        with tf.GradientTape() as tape:
            predictions = self(node_features, training=True)
            loss = self.compute_loss_value(labels, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)

        gradient_pairs = [
            (gradient, variable)
            for gradient, variable in zip(
                gradients,
                self.trainable_variables,
            )
            if gradient is not None
        ]

        self.optimizer.apply_gradients(gradient_pairs)
        self.update_metrics(labels, predictions, loss)

        return self.metric_results()

    def test_step(self, data):
        node_features, labels = data
        labels = tf.cast(tf.reshape(labels, (-1, 1)), tf.float32)

        predictions = self(node_features, training=False)
        loss = self.compute_loss_value(labels, predictions)

        self.update_metrics(labels, predictions, loss)

        return self.metric_results()
