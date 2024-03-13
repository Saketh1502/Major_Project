import pandas as pd
import numpy as np
import os, cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence

from path import metadata, directory_output
from joblib import dump


class CustomDataGenerator(Sequence):
    def __init__(self, dataframe, directory, batch_size, img_height, img_width):
        self.dataframe = dataframe
        self.directory = directory
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.indexes = np.arange(len(self.dataframe))

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = self.dataframe.iloc[batch_indexes]

        X = []
        y = []

        for _, row in batch_data.iterrows():
            img_path = self.directory + f'/{row["file"]}'
            for filename in os.listdir(img_path):
                img = cv2.imread(img_path + f'/{filename}')
                # print(img)
                img = cv2.resize(img, (self.img_width, self.img_height))
                img = img.astype('float32') / 255.0  # Normalize pixel values
                X.append(img)
                y.append(row['label'])

        return np.array(X), np.array(y)


def train_model():
    # Load data from CSV
    train_dir = directory_output + '/train'
    val_dir = directory_output + '/val'

    # Load the CSV file
    csv_file = metadata
    data = pd.read_csv(csv_file, dtype={'file': str, 'label': str})
    data['file'] = data['file'].str.replace('.mp4', '')

    # Define image dimensions and batch size
    img_height, img_width = 224, 224
    batch_size = 32

    # Usage:
    train_generator = CustomDataGenerator(data, train_dir, batch_size, img_height, img_width)

    # Load pre-trained ResNet50 model
    resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    # Freeze pre-trained layers
    for layer in resnet_model.layers:
        layer.trainable = False

    # Build new model on top of ResNet50
    model = Sequential([
        resnet_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=10
    )

    return model, history


# Train model
model, history = train_model()

trained_model = "model.joblib"
dump(model, trained_model)
