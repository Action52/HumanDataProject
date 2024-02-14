from collections import Counter

import tensorflow as tf
from keras.layers import Layer, Flatten, Conv1D, Dense, Lambda, Dropout, GRU, SimpleRNN, LSTM
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
from hda.models.base_model import WandbKerasModel


from hda.preprocessor import Preprocessor
from hda.utils import load_config, get_dataset_shape, map_labels_to_classes, get_labels_to_classes
import numpy as np


class WindowsConvolutionLayer(Layer):
    def __init__(self, num_diodes, convolutions_conf, **kwargs):
        super(WindowsConvolutionLayer, self).__init__(**kwargs)
        self.num_diodes = num_diodes  # Number of diodes/channels
        self.conv_blocks = [
            [Conv1D(**conv_conf) for _, conv_conf in convolutions_conf.items()]
            for _ in range(num_diodes)
        ]

    def call(self, inputs):
        conv_outputs = []
        for i in range(self.num_diodes):
            diode_slice = Lambda(lambda x: x[:, :, :, i:i+1])(inputs)  # Shape: (20, 6, 215, 1)
            for conv_layer in self.conv_blocks[i]:
                diode_slice = conv_layer(diode_slice)  # Apply convolution to each diode slice
            # After convolutions, shape is (20, 6, 207, 4) for each diode
            # Reshape to add an extra dimension for diode feature maps
            diode_slice = Lambda(lambda x: tf.expand_dims(x, -2))(diode_slice)  # Shape: (20, 6, 207, 1, 4)
            conv_outputs.append(diode_slice)
        # Stack along the new dimension to keep diodes and their feature maps separate
        combined = Lambda(lambda x: tf.concat(x, axis=-2))(conv_outputs)  # Shape: (20, 6, 207, 22, 4)
        return combined


class DenseLayer(Layer):
    def __init__(self, dense_conf, **kwargs):
        super(DenseLayer, self).__init__(**kwargs)
        self.sequence = [Flatten()]  # Start with a Flatten layer

        # Iterate through the configuration dictionary
        for _, config in dense_conf.items():
            if 'dropout' in config:
                # Interpret this layer as a Dropout layer
                self.sequence.append(Dropout(rate=config['dropout']))
            elif 'dense' or 'output' in config:
                # Interpret this layer as a Dense layer
                activation = config.get('activation', 'relu') if 'activation' in config else 'softmax'
                self.sequence.append(Dense(units=config['units'], activation=activation))

    def call(self, inputs):
        x = inputs
        for layer in self.sequence:
            x = layer(x)
        return x


class TimeSeriesModelSimple(tf.keras.Model):
    def __init__(self, num_versions, time_steps, diodes, num_classes, convolutions_conf, dense_conf, **kwargs):
        super(TimeSeriesModelSimple, self).__init__(**kwargs)
        self.multi_version_conv = WindowsConvolutionLayer(diodes, convolutions_conf)
        self.dense = DenseLayer(dense_conf)

    def call(self, inputs):
        x = self.multi_version_conv(inputs)
        x = self.dense(x)
        return x


class TimeSeriesModelGRU(tf.keras.Model):
    def __init__(self, num_versions, time_steps, diodes, num_classes, convolutions_conf, dense_conf, gru_conf, **kwargs):
        super(TimeSeriesModelGRU, self).__init__(**kwargs)
        self.multi_version_conv = WindowsConvolutionLayer(diodes, convolutions_conf)
        self.gru_layer = GRU(**gru_conf)
        self.dense = DenseLayer(dense_conf)

    def call(self, inputs):
        x = self.multi_version_conv(inputs)
        # We need to merge the versions and diodes dimensions and treat it as the features dimension for the GRU
        x = tf.reshape(x, shape=(-1, x.shape[2], x.shape[1] * x.shape[3] * x.shape[4]))
        x = self.gru_layer(x)
        x = self.dense(x)
        return x


class TimeSeriesModelVanillaRNN(tf.keras.Model):
    def __init__(self, num_versions, time_steps, diodes, num_classes, convolutions_conf, dense_conf, rnn_conf, **kwargs):
        super(TimeSeriesModelVanillaRNN, self).__init__(**kwargs)
        self.multi_version_conv = WindowsConvolutionLayer(diodes, convolutions_conf)
        self.rnn_layer = SimpleRNN(**rnn_conf)
        self.dense = DenseLayer(dense_conf)

    def call(self, inputs):
        x = self.multi_version_conv(inputs)
        # We need to merge the versions and diodes dimensions and treat it as the features dimension for the GRU
        x = tf.reshape(x, shape=(-1, x.shape[2], x.shape[1] * x.shape[3] * x.shape[4]))
        x = self.rnn_layer(x)
        x = self.dense(x)
        return x
    
    
class TimeSeriesModelLSTM(tf.keras.Model):
    def __init__(self, num_versions, time_steps, diodes, num_classes, convolutions_conf, dense_conf, lstm_conf, **kwargs):
        super(TimeSeriesModelLSTM, self).__init__(**kwargs)
        self.multi_version_conv = WindowsConvolutionLayer(diodes, convolutions_conf)
        self.lstm_layer = LSTM(**lstm_conf)
        self.dense = DenseLayer(dense_conf)

    def call(self, inputs):
        x = self.multi_version_conv(inputs)
        x = tf.reshape(x, shape=(-1, x.shape[2], x.shape[1] * x.shape[3] * x.shape[4]))
        x = self.lstm_layer(x)
        x = self.dense(x)
        return x


def main():
    config = load_config("config.yaml")
    batch_size, versions, time_steps, diodes = get_dataset_shape(config)

    preprocessor = Preprocessor("config.yaml")
    train_dataset, val_dataset, test_dataset = preprocessor.run()
    
    y_train = np.concatenate([np.array([label.numpy()]) for _, label in train_dataset.unbatch()], axis=0)

    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))


    # Instantiate the model
    model = TimeSeriesModelLSTM(
        versions, time_steps, diodes, 7,
        config['convolutions_conf'],
        config['dense_conf'],
        lstm_conf=config['lstm']
    )
    model.build(input_shape=(batch_size, versions, time_steps, diodes))

    # Wrap the model with WandbKerasModel
    wandb_model = WandbKerasModel(
        model=model, project_name="hda-sat", config={}, entity="bdma"
    )

    # Model summary to check the architecture
    wandb_model.model.summary()

    # Compile and fit the model
    wandb_model.compile_and_fit(
        compile_args={
            "optimizer": "adam",
            "loss": "sparse_categorical_crossentropy",
            "metrics": ["accuracy"],
        },
        fit_args={
            "x": train_dataset,
            "epochs": 11,
            "validation_data": val_dataset,
            "experiment_name": "LSTM_with_weight_tanh_full_dataset",
            "class_weight": class_weight_dict,
        },
    )

    # Evaluate the model on the test set and print the confusion matrix
    print("Evaluating on test set...")
    y_true = []
    y_pred = []

    # Assuming your test_dataset yields (features, labels)
    for features, labels in test_dataset:
        preds = wandb_model.model.predict(features)
        y_true.extend(labels.numpy().flatten())
        y_pred.extend(np.argmax(preds, axis=-1).flatten())

    y_true = map_labels_to_classes(y_true, config['preprocess']['drop_labels'])
    y_pred = map_labels_to_classes(y_pred, config['preprocess']['drop_labels'])
    labels = get_labels_to_classes(config['preprocess']['drop_labels'])

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion Matrix:")

    # Modified code to plot the confusion matrix with mapped labels
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)  # Optional: Rotate labels if they overlap
    plt.yticks(rotation=45)  # Optional: Rotate labels if they overlap
    plt.show()

    wandb_model.finish_experiment()


if __name__ == '__main__':
    main()
