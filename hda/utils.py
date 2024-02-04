import matplotlib.pyplot as plt
import yaml
import numpy as np
from matplotlib.colors import Normalize


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
