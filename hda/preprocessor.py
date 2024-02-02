import os
import scipy.io
import mne
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from itertools import islice


from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DATA_DIRECTORY = 'datasets/'

class Preprocessor:
    NUM_OF_FREQUENCY = 200
    path = None

    def __init__(self, path):
        self.path = path
        
    def get_file_names(self):
        return  [file for file in os.listdir('datasets/') if file.endswith('.mat') and "HaLT" in file]

    def _smooth_eeg_data(self, eeg_data, window_size=200):
        """
        Smooth EEG data using a moving average filter with edge padding.
        """
        # Calculate pad size
        pad_size = window_size // 2

        # Initialize an array for the smoothed data
        smoothed_eeg_data = np.empty_like(eeg_data)

        # Apply moving average for each channel
        for i in tqdm(range(eeg_data.shape[1]), desc="smoothen eeg"):
            # Pad the data for the current channel
            padded_channel = np.pad(eeg_data[:, i], pad_size, mode='edge')

            # Apply convolution with a uniform filter
            smoothed_channel = np.convolve(padded_channel, np.ones(window_size) / window_size, mode='same')

            # Assign the smoothed data (trimming the extra padded edges)
            smoothed_eeg_data[:, i] = smoothed_channel[pad_size:-pad_size] if pad_size > 0 else smoothed_channel

        return smoothed_eeg_data

    def _normalize_data(self, data):
        # Amplitude estimateIterate through the datasets
        norm_factor = np.percentile(data, 99, axis=0) - np.percentile(data, 5, axis=0)
        return (data / norm_factor)

    def _downsample_data(self, segment, k):
        """
        Downsample the input matrix along axis 0 by a factor of k.
        Parameters:
        - segment (np.ndarray): The input data matrix with shape (n, m) where n is the number of time points
          and m is the number of diodes (or features).
        - k (int): The downsampling rate, indicating we keep every kth data point.
        Returns:
        - np.ndarray: The downsampled matrix.
        """
        # Ensure k is at least 1 to avoid division by zero or empty selection
        k = max(1, k)
        # Select every kth row from the matrix starting from the first row
        downsampled_matrix = segment[::k, :]
        num_rows_to_pad = segment.shape[0] - downsampled_matrix.shape[0]
        # noinspection PyTypeChecker
        padded_matrix = np.pad(downsampled_matrix,
                               ((0, num_rows_to_pad), (0, 0)),
                               mode='constant', constant_values=0)

        return padded_matrix
    
    def _create_segment_indexes_from_dataset(self, information):
            # Get the label for the data
            marker = [el[0] for el in information['marker'][0][0]]

            segments_by_labels = []
            start = 0
            end = 0

            '''
                Given an array of labels, i.e = [0,0,0,0,0,1,1,1,1,2,2,2]
                it will produce dictionary containing label position based on given index
                the output will be {0: [(0,5)], 1: [(5, 9)], 2: [(9,12)]}
            '''
            for i in tqdm(range(len(marker)), desc='process the marker'):

                if i < len(marker)-1 and marker[i] ==  marker[i+1]:
                    end += 1
                    continue

                if i == len(marker)-1:
                    end = i

                start = max(start, 0)
                end = min(end+1, len(marker))

                segments_by_labels.append([(start, end), marker[i]])

                start = i+1
                end = start

            return segments_by_labels

            
    def _segment_ts(self, eeg_data_norm, eeg_data_smooths, label, segment_indexes, downsamples):
        # eeg data norm, eeg data smooth, segments=(start, end), label=1 (key from dict)
        # output: ([segmented_eeg_norm, segmente_eeg_smooth, segmented_eeg_downsampled], label)
        segment_start, segment_end = segment_indexes
        
        segment_norm = [eeg_data_norm[segment_start:segment_end, :]]
        segment_smooth = [eeg_data_smooth[segment_start:segment_end, :] for eeg_data_smooth in eeg_data_smooths]
        segment_downsamples = [self._downsample_data(segment_norm[0], k) for k in downsamples]
        
        output = ([segment_norm, segment_smooth, segment_downsamples], label)
        
        return output
    
    def _get_data_information(self, mat_data):
        # Extract data - adjust 'data_key' based on your file structure
        info = {}

        for name in mat_data['o'].dtype.names:
            if name != 'data':
                info[name] = mat_data['o'][name]

        return info
    
    def _load_dataset(self, mat_data, is_normalized =True, smoothing_windows=[20,100,200]):

        # Extract data - adjust 'data_key' based on your file structure
        data = mat_data['o']['data'][0][0]
        smoothened_data = []

        # Store data in the dictionary
        if is_normalized:
            data = self._normalize_data(data)
        
        for window in smoothing_windows:
            smoothened_data.append(self._smooth_eeg_data(data, window))    

        return (data, smoothened_data)
    
    def _preload_raw_dataset(self, filename):
        # Construct full file path
        file_path = os.path.join(self.path, filename)

        # Load the .mat file
        mat_data = scipy.io.loadmat(file_path)
        
        return mat_data
    
    def load_and_preprocess(self, filename, is_normalized=True, smoothing_windows=[20,100,200], downsamples=[2, 4], drop_labels={91, 92, 99}):
        # Decode filename from bytes to string if necessary
        if isinstance(filename, bytes):
            filename = filename.decode('utf-8')
            
        mat_data = self._preload_raw_dataset(filename)

        # Dataset information and segment indexes with label
        dataset_information = self._get_data_information(mat_data)
        segment_informations = self._create_segment_indexes_from_dataset(dataset_information)
        
        # dataset
        normalized_data, smoothed_data = self._load_dataset(mat_data, is_normalized, smoothing_windows)
        
        segments = []
        for segment_indexes , label in segment_informations:
            if label in drop_labels:
                continue
            
            segment = self._segment_ts(normalized_data, smoothed_data, label, segment_indexes, downsamples)
            segments.append(segment)
        
        print(segment[0])  
        return segment
    
def main():
    
    
    preprocessor = Preprocessor(DATA_DIRECTORY)
    file_names = preprocessor.get_file_names()
    
    dataset = tf.data.Dataset.from_tensor_slices(file_names)
    
    py_func = lambda file_name: (tf.numpy_function(preprocessor.load_and_preprocess, [file_name], tf.float32))
    dataset = dataset.map(py_func, num_parallel_calls=os.cpu_count())
    
    # # Add a map step to deserialize the segments
    # dataset = dataset.map(preprocessor.deserialize_tensor)
    
    for data in dataset:
        print(data.numpy())
    
main()
