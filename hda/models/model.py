import tensorflow as tf
from keras.layers import Layer, Flatten, Conv1D, Dense, Lambda, Concatenate
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

from hda.preprocessor import Preprocessor
from hda.utils import load_config, get_dataset_shape, plot_conv1d_filters, \
    plot_feature_maps

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


class TimeSeriesModel(tf.keras.Model):
    def __init__(self, num_versions, time_steps, diodes, num_classes, convolutions_conf, **kwargs):
        super(TimeSeriesModel, self).__init__(**kwargs)
        self.multi_version_conv = WindowsConvolutionLayer(diodes, convolutions_conf)
        self.flatten = Flatten()
        self.dense = Dense(units=num_classes, activation='softmax')

    def call(self, inputs):
        x = self.multi_version_conv(inputs)
        x = self.flatten(x)
        outputs = self.dense(x)
        return outputs


def main():

    config = load_config("config.yaml")
    batch_size, versions, time_steps, diodes = get_dataset_shape(config)

    train_preprocessor = Preprocessor("config.yaml")
    val_preprocessor = Preprocessor("config.yaml", mode='val')
    test_preprocessor = Preprocessor("config.yaml", mode="test")
    train_dataset = train_preprocessor.run()
    test_dataset = test_preprocessor.run()
    val_dataset = val_preprocessor.run()

    y_train = np.concatenate([np.array([label.numpy()]) for _, label in train_dataset.unbatch()], axis=0)

    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))

    # Instantiate the model
    model = TimeSeriesModel(versions, time_steps, diodes, 7, config['convolutions_conf'])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Provide the input shape to build the model (necessary for model.summary())
    model.build(input_shape=(batch_size, versions, time_steps, diodes))

    # Model summary to check the architecture
    model.summary()

    # Training the model with validation
    # Training the model with validation
    history = model.fit(
        train_dataset,
        epochs=3,
        validation_data=val_dataset,
        class_weight=class_weight_dict
    )

    # Evaluate the model on the test set and print the confusion matrix
    print("Evaluating on val set...")
    y_true = []
    y_pred = []

    # Assuming your test_dataset yields (features, labels)
    for features, labels in val_dataset:
        preds = model.predict(features)
        y_true.extend(labels.numpy().flatten())
        y_pred.extend(np.argmax(preds, axis=-1).flatten())

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # Create an intermediate model using the functional API
    inputs = tf.keras.Input(shape=(versions, time_steps, diodes))
    x = model.multi_version_conv(
        inputs)  # Use the WindowsConvolutionLayer as a functional layer
    intermediate_model = tf.keras.Model(inputs=inputs, outputs=x)

    # Prepare a sample input
    sample_input = tf.random.normal([1, versions, time_steps, diodes])

    # Pass the sample input through the model to ensure it's built
    _ = model(sample_input)

    # Get the feature maps for the sample input
    feature_maps = intermediate_model.predict(sample_input)
    print(feature_maps.shape)

    # Assuming the shape of feature_maps is (1, versions, time_steps, diodes, filters)
    # And you want to plot for the first diode
    diode_index = 0  # Index for the first diode

    # Plot the feature maps for the selected diode across all versions
    print(f"Feature Maps for Diode {diode_index + 1}")
    plot_feature_maps(feature_maps, diode_index=diode_index)


if __name__ == '__main__':
    main()
