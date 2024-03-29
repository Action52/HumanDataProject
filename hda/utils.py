import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import yaml
import numpy as np
import seaborn as sns

from matplotlib.colors import Normalize
from hda.models.base_model import WandbKerasModel
from hda.constants import LABELS_WITHOUT_ZERO, LABELS_WITH_ZERO, ZERO
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, \
    precision_score, recall_score


def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config


def get_dataset_shape(config):
    prep = config['preprocess'] if 'preprocess' in config else config
    shape = (
        prep['batch_size'],
        1 + len(prep['smoothing_windows']) + len(prep['downsamples']),
        prep['max_length_ts'],
        prep['channels']
    )
    return shape


def plot_conv1d_filters(layer_weights, num_columns=6):
    filters = layer_weights[0]
    num_filters = filters.shape[2]

    num_rows = num_filters // num_columns+(1 if num_filters % num_columns else 0)
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns*2, num_rows*2))

    for i in range(num_rows*num_columns):
        ax = axes.flat[i]
        if i < num_filters:
            filter = filters[:, :, i].squeeze()
            ax.plot(filter)
            ax.set_title(f'Filter {i}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_feature_maps(feature_maps, diode_index, num_versions=6, num_filters=4):
    """
    Plots the feature maps for a specific diode with one row per version and one column per filter, with colors indicating signal intensity.

    Parameters:
    - feature_maps: The output from the convolutional layer for a given input, expected shape (1, versions, time_steps, diodes, filters).
    - diode_index: The index of the diode for which to plot the feature maps.
    - num_versions: Number of versions, default to 6 for the 6 versions.
    - num_filters: Number of filters per version, default to 4 for the 4 filters.
    """
    # Extract the feature maps for the selected diode
    diode_feature_maps = feature_maps[0, :, :, diode_index,
                         :]  # Shape: (versions, time_steps, filters)

    fig, axes = plt.subplots(num_versions, num_filters,
                             figsize=(18, num_versions))

    # Define a colormap and a normalization instance
    cmap = plt.get_cmap('inferno')
    norm = Normalize(vmin=np.min(diode_feature_maps),
                     vmax=np.max(diode_feature_maps))

    for version in range(num_versions):
        for filter_index in range(num_filters):
            ax = axes[version, filter_index]
            feature_map = diode_feature_maps[version, :, filter_index]

            # Create a color array using the colormap
            colors = cmap(norm(feature_map))

            # Plot each feature map with color intensity
            ax.scatter(range(len(feature_map)), feature_map, color=colors,
                       marker='.')
            ax.plot(feature_map, color='lightgrey',
                    alpha=0.5)  # Optional: plot a light line to guide the eye

            if version == 0:
                ax.set_title(f'Filter {filter_index + 1}')
            if filter_index == 0:
                ax.set_ylabel(f'Version {version + 1}', rotation=0,
                              size='large', labelpad=30)

    plt.tight_layout()
    plt.show()


def map_labels_to_classes(labels, drop_labels):
    """
    Maps the labels to their corresponding class index using a dictionary.

    Parameters:
    - labels: The labels to map to classes.
    - class_mapper: A dictionary that maps labels to their corresponding class index.

    Returns:
    - The class indices for the input labels.
    """
    class_mapper = LABELS_WITHOUT_ZERO if ZERO in drop_labels else LABELS_WITH_ZERO
    return np.array([class_mapper[label] for label in labels])

def get_labels_to_classes(drop_labels):
    """
    Returns the correct class mapper based on the presence of the zero label in the dataset.

    Parameters:
    - drop_labels: The labels to drop from the dataset.

    Returns:
    - The correct labels
    """
    return list(LABELS_WITHOUT_ZERO.values()) if ZERO in drop_labels else list(LABELS_WITH_ZERO.values())

def predict_and_plot(wandb_model, test_dataset, config, title=""):
    y_true = []
    y_pred = []
    total_time = 0
    num_samples = 0

    for features, labels in test_dataset:
        start_time = time.time()
        preds = wandb_model.model.predict(features)
        end_time = time.time()
        total_time += (end_time - start_time)
        num_samples += features.shape[0]

        y_true.extend(labels.numpy().flatten())
        y_pred.extend(np.argmax(preds, axis=-1).flatten())

    avg_prediction_time = total_time / num_samples

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')

    # Calculate the number of parameters
    num_params = wandb_model.model.count_params()

    # Calculate model size in MB
    model_path = f"{config['results_path']}{title}_weights.h5"
    wandb_model.model.save_weights(model_path)
    model_size = os.path.getsize(model_path) / 1e6  # Convert from bytes to MB
    os.remove(model_path)  # Clean up

    # Save metrics to CSV
    metrics = {
        'Metric': ['Average Prediction Time (s/sample)', 'Test Accuracy', 'Test F1-score', 'Test Precision', 'Test Recall', 'Number of Parameters', 'Model Weights (MB)'],
        'Value': [round(avg_prediction_time, 3), round(accuracy, 3), round(f1, 3), round(precision, 3), round(recall, 3), round(num_params, 3), round(model_size, 3)]
    }

    test_metrics = {k: v for k, v in zip(metrics['Metric'], metrics['Value'])}

    wandb_model.log_metrics(test_metrics)

    df_metrics = pd.DataFrame(metrics)
    print(df_metrics)

    df_metrics.to_csv(f"{config['results_path']}{title}_metrics.csv", index=False)
    print(f"Metrics saved to {config['results_path']}{title}_metrics.csv")

    y_true = map_labels_to_classes(y_true, config['preprocess']['drop_labels'])
    y_pred = map_labels_to_classes(y_pred, config['preprocess']['drop_labels'])
    labels = get_labels_to_classes(config['preprocess']['drop_labels'])

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(cm)

    # Modified code to plot the confusion matrix with mapped labels
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{title}:Confusion Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.show()
