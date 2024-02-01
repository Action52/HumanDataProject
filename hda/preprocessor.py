import os
import scipy.io
import mne
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf

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


def main():
    preprocessor = Preprocessor('datasets/')
    data_by_filename, info_by_filename = preprocessor.read_as_dictionary()
    segments_by_labels = preprocessor.create_segments_by_labels(info_by_filename)
    
    labels = info_by_filename['HaLT-SubjectJ-161121-6St-LRHandLegTongue']['marker'][0][0]
    eeg_data = data_by_filename['HaLT-SubjectJ-161121-6St-LRHandLegTongue']['smoothened']
    
    print(segments_by_labels['HaLT-SubjectJ-161121-6St-LRHandLegTongue'][1][:10])
    print(eeg_data[353:35514, :])
    
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