from lib.generator import DataGenerator
from lib.resnet import Resnet3DBuilder
import keras
import tensorflow as tf

gesture_list = ["Swiping Left", "Swiping Right", "Swiping Down", "Swiping Up", "No gesture"]

params = {
    "batch_size": 56,
    "n_channels": 3,
    "image_dimensions": (56, 56),
    "frame_count": 36,
    "n_classes": len(gesture_list)
}

train_gen = DataGenerator(file_path="./annotations/train_formatted.csv", base_dir="../data/20bn-jester/", **params)
validation_gen = DataGenerator(file_path="./annotations/validation_formatted.csv", base_dir="../data/20bn-jester/", validation=True, **params)

model = Resnet3DBuilder.build_resnet_101(input_shape=(36, 56, 56, 3), num_outputs=len(gesture_list))
model.summary()

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_gen, validation_data=validation_gen, epochs=10, verbose=1)

model.save("model_final.keras")