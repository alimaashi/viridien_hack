import pandas as pd
import tensorflow as tf
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.saving import save_model, load_model

IMG_SIZE = 128  # arbitrary # pixels

# 1) PREPROCESSING
def load_images_from_df(df, base_dir):
    images = []
    labels = []

    for _, row in df.iterrows():
        img_path = os.path.join(base_dir, row["Filename"])

        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        images.append(img)
        labels.append(row["Label"])

    return np.array(images), np.array(labels)


# filter csv for only the relevant land use types
selected_land_use_types = [
    "agricultural",
    "forest",
    #"river",
    "buildings",
    "denseresidential",
    "sparseresidential",
    "mediumresidential",
]

land_use_types = [
    "Rural and Agricultural Zone"
    "Rural and Agricultural Zone"
    #"Waterways"
    "Commercial/Industrial"
    "Low Density Residential Zone"
    "Medium Density Residential Zone"
    "High Density Residential Zone"
]

land_use_colours = [
    [59, 134, 36],
    [59, 134, 36],
    #[69, 117, 246],
    [232, 51, 35],
    [240, 240, 80],
    [240, 157, 75],
    [174, 91, 33]

]

import os

model_file = "model.keras"

if not model_file:
    print("Loading data...")
    
    full_train_df = pd.read_csv("land_use_data/train.csv")
    full_val_df = pd.read_csv("land_use_data/validation.csv")
    full_test_df = pd.read_csv("land_use_data/test.csv")
    
    train_df = full_train_df[full_train_df["ClassName"].isin(selected_land_use_types)]
    val_df = full_val_df[full_val_df["ClassName"].isin(selected_land_use_types)]
    test_df = full_test_df[full_test_df["ClassName"].isin(selected_land_use_types)]
    
    # rerun labelling of the images because we filtered most of the land use types out
    class_mapping = {cls: idx for idx, cls in enumerate(selected_land_use_types)}
    train_df["Label"] = train_df["ClassName"].map(class_mapping)
    val_df["Label"] = val_df["ClassName"].map(class_mapping)
    test_df["Label"] = test_df["ClassName"].map(class_mapping)
    
    IMG_SIZE = 128  # arbitrary # pixels
    BASE_DIR = "land_use_data/images"
    NUM_CLASSES = len(train_df["ClassName"].unique())
    
    # actually loading the data into vars
    X_train, y_train = load_images_from_df(train_df, BASE_DIR)
    X_val, y_val = load_images_from_df(val_df, BASE_DIR)
    X_test, y_test = load_images_from_df(test_df, BASE_DIR)
    
    # turning labels into categorical (all except one col for label will be 0)
    y_train = to_categorical(y_train, NUM_CLASSES)
    y_val = to_categorical(y_val, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)
    
    #print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    #print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    
    # 2. BUILDING THE MODEL
    model = Sequential()
    
    # Adding layers
    model.add(Conv2D(16, (3, 3), 1, activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(MaxPooling2D(2, 2))
    
    model.add(Conv2D(32, (3, 3), 1, activation="relu"))
    model.add(MaxPooling2D(2, 2))
    
    model.add(Conv2D(16, (3, 3), 1, activation="relu"))
    model.add(MaxPooling2D(2, 2))
    
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    
    model.add(Dense(NUM_CLASSES, activation="softmax"))
    
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    model.summary()
    
    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=100,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
    )
    
    save_model(model, "land_use_model.keras")
    
    # 3. EVALUATING THE MODEL
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy}")

model = load_model('land_use_model.keras')

# 4. MAKING PREDICTIONS
area = cv2.imread("edi2.png")
area = cv2.cvtColor(area, cv2.COLOR_BGRA2RGB)
area = cv2.resize(area, (5000, 5000), interpolation=cv2.INTER_LANCZOS4)

MODEL_SIZE = 150

for i in range(0, area.shape[0] - MODEL_SIZE + 1, MODEL_SIZE):  # Avoids incomplete slices
    print(i)
    for j in range(0, area.shape[1] - MODEL_SIZE + 1, MODEL_SIZE):
        section = area[i : i + MODEL_SIZE, j : j + MODEL_SIZE]

        # Ensure section has correct shape
        if section.shape[:2] != (IMG_SIZE, IMG_SIZE):  
            section = cv2.resize(section, (IMG_SIZE, IMG_SIZE))  # Resize if necessary

        section = section / 255.0  # Normalize pixel values
        section = np.expand_dims(section, axis=0)  # Add batch dimension -> (1, 128, 128, 3)
        # Predict
        prediction = model.predict(section)
        if np.max(prediction) > 0.25:
            colour = land_use_colours[np.argmax(prediction)]
            area[i : i + MODEL_SIZE, j : j + MODEL_SIZE] = area[i : i + MODEL_SIZE, j : j + MODEL_SIZE] * 0.3 + np.full((MODEL_SIZE, MODEL_SIZE, 3), colour, dtype=np.uint8) * 0.7

plt.imshow(area)
plt.show()
