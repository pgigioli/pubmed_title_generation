import os
import random

class PubMed_Dataset:
    def __init__(self, abstract_title_dir, val_split=0.1, test_split=0.1, max_data_size=None):
        self.train_data = None
        self.val_data = None
        self.trainval_data = None
        self.test_data = None
        
        data = self.__get_filenames(abstract_title_dir, max_data_size=max_data_size)
        random.shuffle(data)
        
        self.trainval_data = data[:int((1-test_split)*len(data))]
        self.train_data = self.trainval_data[:int((1-val_split-test_split)*len(data))]
        self.val_data = self.trainval_data[int((1-val_split-test_split)*len(data)):]
        self.test_data = data[int((1-test_split)*len(data)):]
        
    def __get_filenames(self, data_dir, max_data_size=None):
        filenames = []
        for file in os.listdir(data_dir):
            if max_data_size is not None and i >= max_data_size:
                break
            filenames.append(os.path.join(data_dir, file))
        return filenames

class CNN_Dailymail_Dataset:
    def __init__(self, cnn_folder, dailymail_folder, val_split=0.1, test_split=0.1, max_data_size=None):
        self.train_data = None
        self.val_data = None
        self.trainval_data = None
        self.test_data = None
        
        cnn_data = self.__get_filenames(cnn_folder)
        dailymail_data = self.__get_filenames(dailymail_folder)
        
        combined_data = cnn_data + dailymail_data
        random.shuffle(combined_data)
        
        if max_data_size != None:
            combined_data = combined_data[:max_data_size]
        
        self.trainval_data = combined_data[:int((1-test_split)*len(combined_data))]
        self.train_data = self.trainval_data[:int((1-val_split-test_split)*len(combined_data))]
        self.val_data = self.trainval_data[int((1-val_split-test_split)*len(combined_data)):]
        self.test_data = combined_data[int((1-test_split)*len(combined_data)):]
        
    def __get_filenames(self, data_dir):
        filenames = []
        for file in os.listdir(data_dir):
            filenames.append(os.path.join(data_dir, file))
        return filenames