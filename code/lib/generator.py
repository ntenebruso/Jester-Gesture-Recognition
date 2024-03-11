import pandas as pd
import tensorflow as tf
import numpy as np
from glob import glob
import keras
from sklearn.preprocessing import OneHotEncoder

class DataGenerator(keras.utils.Sequence):
    encoder = OneHotEncoder(sparse_output=False)

    def __init__(self, file_path, base_dir, batch_size = 2, frame_count=36, image_dimensions=(256, 256), n_channels=3, n_classes=10, validation=False):
        self.batch_size = batch_size
        self.base_dir = base_dir
        self.df = pd.read_csv(file_path, sep=";")
        self.frame_count = frame_count
        self.image_dimensions = image_dimensions
        self.n_channels = n_channels
        self.n_classes = n_classes

        if validation is False:
            self.encoder.fit(self.df["label"].values.reshape(-1, 1))
            np.save("encoder_classes.npy", self.encoder.categories_)

        self.on_epoch_end()
    
    def __len__(self):
        return self.df.shape[0] // self.batch_size

    def __getitem__(self, index):
        x = np.empty((self.batch_size, self.frame_count, *self.image_dimensions, self.n_channels))
        y = []

        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        for i, item in enumerate(indexes):
            # print(glob(self.base_dir + str(self.df.loc[item, "id"]) + "/*.jpg"))
            files_list = self.standardize_frame_count(glob(self.base_dir + str(self.df.loc[item, "id"]) + "/*.jpg"))
            for j, filename in enumerate(files_list):
                x[i, j] = keras.preprocessing.image.img_to_array(keras.preprocessing.image.load_img(filename, color_mode="grayscale", target_size=self.image_dimensions))
            y.append(self.df.loc[item, "label"])
        
        y = np.asarray(y)
        encoded = self.encoder.transform(y.reshape(-1, 1))
        return x, encoded

    
    def on_epoch_end(self):
        self.indexes = np.arange(self.df.shape[0])
        np.random.shuffle(self.indexes)
        
    def standardize_frame_count(self, files):
        shape = len(files)

        if shape < self.frame_count:
            num_to_add = self.frame_count - shape
            mid = shape // 2
            dup = [files[mid]] * num_to_add
            return files[:mid] + dup + files[mid + 1:]
        elif shape > self.frame_count:
            num_to_remove = shape - self.frame_count
            return files[num_to_remove:]
        
        return files
