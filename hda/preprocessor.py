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

class Preprocessor:
    NUM_OF_FREQUENCY = 200
    path = None

    def __init__(self, path):
        self.path = path

    def smooth_eeg_data(self, eeg_data, window_size=200):
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

    def normalize_data(self, data):
        # Amplitude estimateIterate through the datasets
        norm_factor = np.percentile(data, 99, axis=0) - np.percentile(data, 5, axis=0)
        return (data / norm_factor)

    def read_as_dictionary(self):
        # Collect the label from the dataset
        data_by_filename = {}
        info_by_filename = {}

        mat_files = [file for file in os.listdir('datasets/') if file.endswith('.mat')]

        # Loop through each file in the directory with a tqdm progress bar
        for file in tqdm(mat_files, desc="Reading .mat files"):
            # Construct full file path
            file_path = os.path.join(self.path, file)

            # Load the .mat file
            mat_data = scipy.io.loadmat(file_path)

            # Extract data - adjust 'data_key' based on your file structure
            data = mat_data['o']['data']
            info = {}

            for name in mat_data['o'].dtype.names:
                if name != 'data':
                    info[name] = mat_data['o'][name]

            # Extract label from file name
            label = file.replace('.mat', '')

            data_by_filename[label] = {}
            # Store data in the dictionary
            normalized_data = self.normalize_data(data[0][0])
            smoothened_data = self.smooth_eeg_data(normalized_data)
            data_by_filename[label]['normalized'] = normalized_data
            data_by_filename[label]['smoothened'] = smoothened_data

            info_by_filename[label] = info

        return data_by_filename, info_by_filename

    def create_segments_by_labels(self, info_by_filename):
        result = {}

        for key in info_by_filename.keys():
            # Get the label for the data
            marker = [el[0] for el in info_by_filename[key]['marker'][0][0]]

            segments_by_labels = {}
            start = 0
            end = 0

            length_arr = []

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

                segments_by_labels.setdefault(marker[i], [])
                segments_by_labels[marker[i]].append((start, end))

                if marker[i] > 90:
                    length_arr.append(end-start)

                start = i+1
                end = start

            result[key] = segments_by_labels

        return result

    def slice_segments(self, segments_by_labels, slices = 3):
        pass

    def segment_ts(self, eeg_data_norm, eeg_data_smooths: list, segment_start, segment_end, label,
                   downsamples):
        # eeg data norm, eeg data smooth, segments=(start, end), label=1 (key from dict)
        # output: ([segmented_eeg_norm, segmente_eeg_smooth, segmented_eeg_downsampled], label)
        segment_norm = [eeg_data_norm[segment_start:segment_end, :]]
        segment_smooth = [eeg_data_smooth[segment_start:segment_end, :] for eeg_data_smooth in eeg_data_smooths]
        segment_downsamples = [self.downsample_data(segment_norm[0], k) for k in downsamples]
        output = ([segment_norm, segment_smooth, segment_downsamples], label)
        return output

    def downsample_data(self, segment, k):
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

    def preprocess_file(self, data_by_filename, info_by_filename, filename, downsamples):
        index_segments_by_labels = self.create_segments_by_labels(info_by_filename)
        eeg_data_norm = data_by_filename[filename]['normalized']
        eeg_data_smooths = data_by_filename[filename]['smoothened']

        # Shuffle the indexes
        index_segments_by_labels.shuffle()

        for (start_index, end_index), label in index_segments_by_labels:
            segments_label = self.segment_ts(
                eeg_data_norm,
                eeg_data_smooths,
                start_index,
                end_index,
                label,
                downsamples
            )
            yield segments_label

    def batch_file(self, filename, data_by_filename, info_by_filename, batch_size=100, downsamples=(2, 4)):
        generator = self.preprocess_file(data_by_filename, info_by_filename,
                                         filename, downsamples)

        while True:
            batch = list(islice(generator, batch_size))
            if not batch:
                break
            yield batch

def main():
    preprocessor = Preprocessor('datasets/')
    data_by_filename, info_by_filename = preprocessor.read_as_dictionary()
    segments_by_labels = preprocessor.create_segments_by_labels(info_by_filename)

    labels = info_by_filename['HaLT-SubjectJ-161121-6St-LRHandLegTongue']['marker'][0][0]
    eeg_data = data_by_filename['HaLT-SubjectJ-161121-6St-LRHandLegTongue']['smoothened']

    print(segments_by_labels['HaLT-SubjectJ-161121-6St-LRHandLegTongue'][1][:10])
    print(eeg_data[353:35514, :])

    #segments_label_1 = preprocessor.segment_ts(eeg_data, segments_by_labels['HaLT-SubjectJ-161121-6St-LRHandLegTongue'][1], 1)
    #print(segments_label_1)

    #downsampled = preprocessor.downsample_data(segments_label_1[0][0], 2)
    #print(segments_label_1[0][0].shape)
    #print(downsampled.shape)

    flattened_segments = preprocessor.preprocess_file(
        data_by_filename,
        info_by_filename,
        'HaLT-SubjectJ-161121-6St-LRHandLegTongue',
        downsamples=(2,4))

    print(len(flattened_segments))
    print(len(flattened_segments[0]))
    print(len(flattened_segments[0][0]))



    # # Define the row range and the channel index to plot
    # row_start = 0
    # row_end = eeg_data.shape[0]
    # channel_index = 0  # Change this to plot a different channel (0 to 21)

    # # Extract the relevant data for the specified channel
    # channel_data = eeg_data[row_start:row_end, channel_index]

    # # Plot the data
    # plt.figure(figsize=(12, 4))
    # plt.plot(channel_data, label=f'Channel {channel_index + 1}')

    # plt.title(f'EEG Data from Rows 36000 to 40000 (Channel {channel_index + 1})')
    # plt.xlabel('Sample')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.show()

    # class_mapping = {91: 7, 92: 8, 99: 9}
    # for original, new in class_mapping.items():
    #     labels[labels == original] = new

    # print(labels.shape, eeg_data.shape)
    # # One-hot encode the labels
    # # One-hot encode the labels
    # labels_one_hot = to_categorical(labels, num_classes=10)

    # # Reshape data to have shape (samples, timesteps, features)
    # # As each sample is independent, timesteps=1
    # data_reshaped = eeg_data.reshape(eeg_data.shape[0], 1, eeg_data.shape[1])

    # # Split the data into train and test sets
    # batch_size = 64

    # data_train, data_test, labels_train, labels_test = train_test_split(
    #     data_reshaped, labels_one_hot, test_size=0.2, random_state=42)  # 20% data as test set

    # # Convert to TensorFlow datasets
    # train_dataset = tf.data.Dataset.from_tensor_slices((data_train, labels_train))
    # train_dataset = train_dataset.batch(batch_size).shuffle(buffer_size=10000)

    # test_dataset = tf.data.Dataset.from_tensor_slices((data_test, labels_test))
    # test_dataset = test_dataset.batch(batch_size)

    # # Define the LSTM model
    # model = Sequential()
    # model.add(LSTM(128, input_shape=(1, 22), return_sequences=True))
    # model.add(LSTM(64, return_sequences=False))
    # model.add(Dense(10, activation='softmax'))

    # # Compile the model
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # # Train the model with training data
    # model.fit(train_dataset, epochs=10)

    # # Evaluate the model on the test set
    # test_loss, test_accuracy = model.evaluate(test_dataset)
    # print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")


main()
