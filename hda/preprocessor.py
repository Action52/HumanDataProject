import os

import numpy as np
import scipy.io
import tensorflow as tf
import yaml
from tqdm import tqdm

import matplotlib.pyplot as plt

from hda.utils import get_dataset_shape
from hda.constants import BIG_SEGMENTS, CLASS_MAPPER, SEGMENT_PADDING, ZERO


class Preprocessor:
    def __init__(self, config_path, mode='train'):
        """
        Initializes the Preprocessor with configuration parameters specified in a YAML file.

        This method loads the configuration from the given YAML file, focusing specifically on the 'preprocess'
        section of the configuration, which is expected to contain preprocessing-related parameters.

        Args:
            config_path (str): The file path to the YAML configuration file.
        """
        with open(config_path, "r") as config_file:
            self.config = yaml.safe_load(config_file)
            self.config = self.config["preprocess"]

        self.mode = mode

        if mode == 'val':
            self.config['data_directory'] = self.config['val_directory']
        if mode == 'test':
            self.config['data_directory'] = self.config['test_directory']

        self.total_segments = 0

    def get_file_names(self, data_directory=None):
        """
        Retrieves the names of .mat files containing "HaLT" in their names from the specified data directory.

        This method filters all files in the configured data directory, selecting only those that are MATLAB files
        (.mat extension) and contain "HaLT" in their filenames, which is indicative of a specific dataset or data type.

        Returns:
            list: A list of matching .mat file names.
        """
        if not data_directory:
            data_directory = self.config['data_directory']

        return [
            file
            for file in os.listdir(data_directory)
            if file.endswith(".mat") and "HaLT" in file
        ]

    def _smooth_eeg_data(self, eeg_data, window_size):
        """
        Applies a moving average filter to EEG data for smoothing, using edge padding to handle boundary effects.

        This method smooths each channel of the EEG data independently by applying a moving average filter. To avoid
        artifacts at the edges of the data, it uses edge padding where the signal is extended by replicating the edge values.

        Args:
            eeg_data (np.ndarray): The raw EEG data to be smoothed, typically with shape (timepoints, channels).
            window_size (int, optional): The size of the moving average window.

        Returns:
            np.ndarray: The smoothed EEG data, maintaining the original shape of the input.
        """
        pad_size = window_size // 2
        smoothed_eeg_data = np.empty_like(eeg_data)

        for i in range(eeg_data.shape[1]):  # tqdm(range(eeg_data.shape[1]), desc="smoothen eeg"):
            padded_channel = np.pad(eeg_data[:, i], pad_size, mode="edge")
            smoothed_channel = np.convolve(
                padded_channel, np.ones(window_size) / window_size, mode="same"
            )
            smoothed_eeg_data[:, i] = (
                smoothed_channel[pad_size:-pad_size]
                if pad_size > 0
                else smoothed_channel
            )

        return smoothed_eeg_data

    def _normalize_data(self, data):
        """
        Normalizes EEG data by scaling based on the 5th and 99th percentile values across each channel.

        This method enhances the robustness of the normalization by using percentile values, which are less
        sensitive to extreme outliers compared to min-max normalization. The data is scaled such that the
        range between the 5th and 99th percentiles is normalized to a standard range, improving the consistency
        of EEG signal amplitudes across channels.

        Args:
            data (np.ndarray): The EEG data to be normalized, typically with shape (timepoints, channels).

        Returns:
            np.ndarray: The normalized EEG data, maintaining the original shape of the input.
        """
        norm_factor = np.percentile(data, 99, axis=0) - np.percentile(
            data, 5, axis=0
        )
        return data / norm_factor

    def _downsample_data(self, segment, k):
        """
        Downsamples the input data by selecting every k-th sample, effectively reducing the sampling rate.

        This method is used to reduce the data size and computational load for subsequent processing steps
        by decreasing the temporal resolution of the data. Downsampled data retains the overall shape and
        trends of the original signal but with fewer data points.

        Args:
            segment (np.ndarray): The input data segment to be downsampled, typically with shape (timepoints, features).
            k (int): The downsampling factor, specifying how many samples to skip. For example, k=2 means keep every second sample.

        Returns:
            np.ndarray: The downsampled data segment, with a reduced number of timepoints.
        """
        k = max(1, k)
        downsampled_matrix = segment[::k, :]
        num_rows_to_pad = segment.shape[0] - downsampled_matrix.shape[0]
        padded_matrix = np.pad(
            downsampled_matrix,
            ((0, num_rows_to_pad), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        return padded_matrix

    def _create_segment_indexes_from_dataset(self, information):
        """
        Identifies segment boundaries and their corresponding labels within the dataset based on markers.

        This method processes an array of labels/markers indicating different segments within the dataset. It identifies
        the start and end points of each segment and associates them with their corresponding labels. This is useful for
        segmenting the dataset into meaningful parts based on labeled events or conditions.

        Args:
            information (dict): A dictionary containing dataset information, including markers that indicate different segments.

        Returns:
            list: A list of tuples, each containing a pair of indexes (start, end) representing the boundaries of a segment,
                  and the label associated with that segment. For example, `[(0, 5, label1), (6, 10, label2)]` indicates
                  that the first segment from index 0 to 5 is associated with `label1`, and the second segment from index
                  6 to 10 is associated with `label2`.
        """
        marker = [el[0] for el in information["marker"][0][0]]
        segments_by_labels = []
        start = end = 0

        for i in range(len(marker)):  # tqdm(range(len(marker)), desc="process the marker"):
            if i < len(marker) - 1 and marker[i] == marker[i + 1]:
                end += 1
                continue

            if i == len(marker) - 1:
                end = i

            segment = self.__create_segments(start, end, marker[i], max_length=self.config["max_length_ts"])
            
            segments_by_labels.extend(segment)
            start = i + 1
            end = start

        return segments_by_labels
    
    def __create_segments(self, start, end, marker, max_length):
        """
            This method to handle consecutives 0s in the beginning of the data,
            this will break down big 0s segment to several segments. 
            This will only apply for segments more in the beginning of the data
        """
        if end-start < BIG_SEGMENTS or marker != 0:
            return [[(start, end), marker]]
        
        new_segments = []
        new_start = start
        for i in range(start, end+1):
            if (i-new_start) == max_length and i < end:
                new_segments.append([(new_start, i), marker])
                new_start = i
            
            # Append the leftover segments
            if i == end:
                 new_segments.append([(new_start, i), marker])
        
        return new_segments
            
    def _segment_ts(
        self,
        eeg_data_norm,
        eeg_data_smooths,
        label,
        segment_indexes,
        downsamples,
    ):
        """
        Segments the time series EEG data, applies smoothing and downsampling, and pads each segment to a uniform length.

        This method processes a segment of EEG data by applying normalization, smoothing with different window sizes, and
        downsampling by various factors. It ensures that each processed segment has the same length by padding shorter
        segments. This uniform length is specified in the configuration.

        Args:
            eeg_data_norm (np.ndarray): The normalized EEG data.
            eeg_data_smooths (list): A list of EEG data arrays, each smoothed with a different window size.
            label (int): The label associated with the current segment, indicating the condition or event.
            segment_indexes (tuple): A pair indicating the start and end indexes of the segment within the EEG data.
            downsamples (list): A list of integers representing the downsampling factors to be applied to the segment.

        Returns:
            tuple: A tuple where the first element is a list containing the padded, normalized, smoothed, and downsampled
                   segment arrays, and the second element is the label associated with the segment.
        """
        segment_start, segment_end = segment_indexes
        max_length = self.config["max_length_ts"]
        
        # Add segment padding to include the 0s in the segment
        segment_start -= SEGMENT_PADDING
        segment_end += SEGMENT_PADDING
        
        segment_norm = eeg_data_norm[segment_start:segment_end, :]
        segment_smooth = [
            eeg_data_smooth[segment_start:segment_end, :]
            for eeg_data_smooth in eeg_data_smooths
        ]
        
        segment_norm_padded = self._pad_sequences(
            [segment_norm], maxlen=max_length, value=0.0
        )[0]
        segment_smooth_padded = [
            self._pad_sequences([segment], maxlen=max_length, value=0.0)[0]
            for segment in segment_smooth
        ]
        
        segment_downsamples = [
            self._downsample_data(segment_norm_padded, k) for k in downsamples
        ]

        output = (
            [segment_norm_padded]
            + segment_smooth_padded
            + segment_downsamples,
            label,
        )

        return output

    def _get_data_information(self, mat_data):
        """
        Extracts and returns the metadata from the given .mat file, excluding the raw data itself.

        This method iterates through the fields in the loaded .mat file data structure, collecting all metadata
        information except for the actual data. This metadata might include details such as sampling rates, subject
        information, recording conditions, and more, depending on the dataset.

        Args:
            mat_data (dict): A dictionary representing the loaded .mat file, typically obtained using scipy.io.loadmat.

        Returns:
            dict: A dictionary containing all metadata fields found in the .mat file, excluding the 'data' field.
        """
        info = {}

        for name in mat_data["o"].dtype.names:
            if name != "data":
                info[name] = mat_data["o"][name]

        return info

    def _load_dataset(self, mat_data, is_normalized, smoothing_windows):
        """
        Loads and processes the dataset from the provided .mat data structure.

        This method extracts the raw data from the .mat structure, applies normalization if specified, and then
        smooths the data using the provided window sizes. The result is the original data alongside its smoothed
        versions.

        Args:
            mat_data (dict): The loaded .mat file data structure, typically obtained using scipy.io.loadmat.
            is_normalized (bool): A flag indicating whether the data should be normalized.
            smoothing_windows (list): A list of integers representing the window sizes to be used for smoothing the data.

        Returns:
            tuple: A tuple containing the original (and possibly normalized) data and a list of its smoothed versions.
        """
        data = mat_data["o"]["data"][0][0]
        smoothened_data = []

        if is_normalized:
            data = self._normalize_data(data)

        for window in smoothing_windows:
            smoothened_data.append(self._smooth_eeg_data(data, window))

        return data, smoothened_data

    def _preload_raw_dataset(self, filename, directory=None):
        """
        Preloads the dataset from a specified .mat file.

        This method constructs the full path to the .mat file based on the provided filename and the data directory
        specified in the configuration. It then loads the .mat file into memory, making it ready for further processing.

        Args:
            filename (str): The name of the .mat file to be loaded.

        Returns:
            dict: The data structure loaded from the .mat file, typically containing both the raw data and its associated metadata.
        """
        if not directory:
            directory = self.config["data_directory"]
        file_path = os.path.join(directory, filename)
        mat_data = scipy.io.loadmat(file_path)

        return mat_data

    def load_and_preprocess(self, filename):
        """
        Loads a .mat file specified by the filename, preprocesses its contents, and returns processed segments along with their labels.

        This method performs a series of preprocessing steps on EEG data contained in a .mat file. These steps include loading the raw
        data from the file, extracting relevant segments based on predefined markers or conditions, applying normalization and smoothing
        operations to these segments, and optionally downsampling the data for each segment. Each processed segment is then paired with
        its corresponding label, which is derived from the file's metadata or predefined conditions.

        Args:
            filename (str): The name of the .mat file to load and preprocess. This file should be located within the data directory
                            specified in the preprocessor's configuration.

        Returns:
            list: A list of tuples, where each tuple contains a processed data segment (as a NumPy array) and its associated label.
                  The shape and content of each data segment depend on the preprocessing operations applied.
        """
        if isinstance(filename, bytes):
            filename = filename.decode("utf-8")


        mat_data = self._preload_raw_dataset(filename, )

        dataset_information = self._get_data_information(mat_data)
        segment_informations = self._create_segment_indexes_from_dataset(
            dataset_information
        )

        normalized_data, smoothed_data = self._load_dataset(
            mat_data,
            self.config["is_normalized"],
            self.config["smoothing_windows"],
        )

        all_segments = []
        
        for segment_indexes, label in segment_informations:
            if label in self.config["drop_labels"]:
                continue
            
            if ZERO in self.config["drop_labels"]:
                label = CLASS_MAPPER[label]
            
            segment, _ = self._segment_ts(
                normalized_data,
                smoothed_data,
                label,
                segment_indexes,
                self.config["downsamples"],
            )
            stacked_segment = np.stack(segment, axis=0)  # Shape: (6, 200, 22)
            all_segments.append((stacked_segment, label))
            self.total_segments += 1
        return all_segments

    def _pad_sequences(
        self,
        sequences,
        maxlen=None,
        dtype="float32",
        padding="post",
        truncating="post",
        value=0.0,
    ):
        """
        Pads 2D sequences to the same length.

        Arguments:
            sequences: List of 2D sequences (list of 2D arrays) to be padded.
            maxlen: Maximum length of all sequences. If not provided, it's calculated as the maximum length of all sequences.
            dtype: Type of the output sequences.
            padding: 'pre' or 'post', indicates whether to add padding before or after each sequence.
            truncating: 'pre' or 'post', indicates whether to truncate sequences before or after maxlen.
            value: Float, value to fill the padding.

        Returns:
            Padded sequences of shape (num_sequences, maxlen, num_features).
        """
        if not maxlen:
            maxlen = max(seq.shape[0] for seq in sequences)

        num_features = sequences[0].shape[
            1
        ]  # Assuming all sequences have the same number of features
        padded_sequences = np.full(
            (len(sequences), maxlen, num_features), value, dtype=dtype
        )

        for i, seq in enumerate(sequences):
            if seq.shape[0] > maxlen:  # Truncate
                if truncating == "pre":
                    trunc = seq[-maxlen:]
                else:
                    trunc = seq[:maxlen]
            else:
                trunc = seq

            if padding == "post":  # Pad
                padded_sequences[i, : trunc.shape[0], :] = trunc
            else:
                padded_sequences[i, -trunc.shape[0] :, :] = trunc

        return padded_sequences

    def _set_shapes(self, features, labels):
        _, versions, time_steps, channels = get_dataset_shape(self.config)
        features.set_shape([None, versions, time_steps,
                            channels])
        labels.set_shape([None])
        return features, labels
    
    def _get_approximate_dataset_size(self, filenames):
        """
        Get the approximate size of the dataset to be loaded.

        This method calculates the approximate size of the dataset to be loaded, based on the number of files and the
        number of segments in each file. It does so by counting the number of segments in each file and summing them up.

        Args:
            filenames (list): A list of file names to be loaded.

        Returns:
            int: The approximate size of the dataset to be loaded.
        """
        dataset_size = 0

        for filename in tqdm(filenames, desc="get approximate dataset size"):
            mat_data = self._preload_raw_dataset(filename)
            dataset_information = self._get_data_information(mat_data)
            segment_informations = self._create_segment_indexes_from_dataset(
                dataset_information
            )
            
            for _ , label in segment_informations:
                if label in self.config["drop_labels"]:
                    continue
                dataset_size += 1

        return dataset_size

    def run(self, data_directory=None):
        """
        Constructs and returns a TensorFlow dataset from preprocessed EEG data files.

        This method orchestrates the preprocessing pipeline by first retrieving the names of EEG data files,
        then applying a series of preprocessing steps to each file, and finally aggregating the preprocessed
        data into a TensorFlow dataset suitable for training or evaluation in machine learning models.

        The preprocessing steps include decoding filenames, loading and preprocessing data from each file,
        converting the preprocessed data into TensorFlow tensors, and wrapping these tensors in a TensorFlow
        dataset. The resulting dataset is further configured with options for caching, shuffling, batching,
        and prefetching to optimize the data pipeline for performance.

        Returns:
            tf.data.Dataset: A TensorFlow dataset containing batches of preprocessed EEG data segments and their
                             associated labels. This dataset is ready for use in training or evaluating machine
                             learning models.
        """
        if not data_directory:
            data_directory = self.config['data_directory']

        filenames = self.get_file_names(data_directory)
        approximate_dataset_size = self._get_approximate_dataset_size(filenames)

        def preprocess_func(file_name):
            file_name_str = file_name.numpy().decode("utf-8")
            segments, labels = zip(*self.load_and_preprocess(file_name_str))
            segments_tensor = tf.convert_to_tensor(segments, dtype=tf.float32)
            labels_tensor = tf.convert_to_tensor(labels, dtype=tf.int32)
            return segments_tensor, labels_tensor

        def wrapped_preprocess_func(file_name):
            segments_tensor, labels_tensor = tf.py_function(
                preprocess_func, inp=[file_name], Tout=[tf.float32, tf.int32]
            )
            return tf.data.Dataset.from_tensor_slices(
                (segments_tensor, labels_tensor)
            )

        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = dataset.interleave(
            wrapped_preprocess_func,
            cycle_length=tf.data.AUTOTUNE,
            block_length=1,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        if self.config["cache_file"]:
            dataset = dataset.cache(f'{self.mode}_dataset_cache')

        if self.config["shuffle"]:
            dataset = dataset.shuffle(buffer_size=approximate_dataset_size)

        # Set sizes for train, validation, and test datasets
        train_size = int(float(self.config["train_size"]) * approximate_dataset_size)
        val_size = int(float(self.config["val_size"]) * train_size)

        train_dataset = dataset.take(train_size)
        remaining = dataset.skip(train_size)
        val_dataset = remaining.take(val_size)
        remaining = remaining.skip(val_size)
        test_dataset = remaining

        train_dataset = train_dataset.batch(self.config["batch_size"]).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(self.config["batch_size"]).prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(self.config["batch_size"]).prefetch(tf.data.AUTOTUNE)

        train_dataset = train_dataset.map(self._set_shapes)
        val_dataset = val_dataset.map(self._set_shapes)
        test_dataset = test_dataset.map(self._set_shapes)

        return train_dataset, val_dataset, test_dataset


def main():
    # Test code
    preprocessor = Preprocessor("config.yaml")
    train_dataset, val_dataset, test_dataset = preprocessor.run()
    train_count = val_count = test_count = 0
    
    for _ in train_dataset.as_numpy_iterator():
        train_count += 1
    for _ in val_dataset.as_numpy_iterator():
        val_count += 1
    for _ in test_dataset.as_numpy_iterator():
        test_count += 1
    print(train_count, val_count, test_count)

    for data, label in train_dataset.as_numpy_iterator():
        print(f"Shape: {data.shape}, Label: {label}")

        # Determine the number of time series to plot
        num_ts = data[0].shape[
            0]  # Number of time series in the first element of the batch

        # Create a figure and a set of subplots
        # fig, axs = plt.subplots(num_ts, 1, figsize=(16, 8))  # Adjust the figure size as needed

        # for i, ts in enumerate(data[0]):
        #     print(ts.shape)

        #     # If there's only one time series, 'axs' won't be an array, so we need to handle that case
        #     ax = axs[i] if num_ts > 1 else axs

        #     ax.plot(ts)
        #     ax.set_title(f"Label: {label}")
        #     ax.set_xlabel("Time")
        #     ax.set_ylabel("Amplitude")

        # # Adjust layout to prevent overlap
        # plt.tight_layout()

        # # Display the plot
        # plt.show()

        # # Clear the current figure before the next batch
        # plt.clf()


if __name__ == "__main__":
    main()
