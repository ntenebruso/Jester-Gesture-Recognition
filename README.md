# Gesture Recognition Dataset

Using the Jester Gesture Recognition Dataset.

## How to use

*Before running any code, ensure that all dependencies are installed from `requirements.txt`!*

### Train the dataset

To train Jester using ResNet101, follow the steps below:

1. Download the [Jester Dataset](https://developer.qualcomm.com/software/ai-datasets/jester).
2. Extract all the contents of the dataset to `./data/20bn-jester/`. In the future, this path will be able to be changed in a config file.
3. Configure the labels that should be trained in `./code/format_csv.py` by changing the `gesture_list` array. Copy this array into `./code/main.py`. (I know this is a weird approach, I'm working to change this).
4. Configure batch size, image dimensions, and epochs in `./code/main.py` (or use the defaults provided).
5. Run `format_csv.py` to generate the correct annotations using only the labels you have defined. Then run `main.py` to begin training. The model will be exported to `./code/model_final.keras`.

### Use the dataset

1. Ensure that `model_final.keras` and `encoder_classes.npy` are in the `code` folder.
2. Run `camera_app.py` and test out the different hand gestures!

