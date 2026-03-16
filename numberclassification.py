import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input, BatchNormalization, GlobalAveragePooling2D

import numpy as np
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

np.random.seed(34)
tf.random.set_seed(34)
str_class = ['lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor']

# DATA PREPARATION 

# load training and testing data for "coarse labels"
(coarse_train, coarse_TrLabels), (coarse_test, coarse_TsLabels) = keras.datasets.cifar100.load_data(label_mode='coarse')
print('Coarse class: {}'.format(np.unique(coarse_TrLabels)))

#load training and testing data for "fine labels"
(fine_train, fine_TrLabels), (fine_test, fine_TsLabels) = keras.datasets.cifar100.load_data(label_mode='fine')
print('Fine class for all: {}'.format(np.unique(fine_TrLabels)))

# extract images of a specific coarse class from TRAINING data
idx = []
for i in range(len(coarse_TrLabels)):
    if coarse_TrLabels[i] == 19: #  checks the coarse label of each sample in the training dataset (vehicles 2)
        idx.append(i)

print('Total images with 4 coarse label (Vehicles 2) from TRAINING DATASET: {}'.format(len(idx)))
idx = np.array(idx)

# extract all image and corresponding "fine" label and store in train_images, train_labels variable list.
train_images, train_labels = fine_train[idx], fine_TrLabels[idx]
print("Shape of the image training dataset: {}".format(train_images.shape))
uniq_fineClass = np.unique(train_labels)
print('Fine Class for the extracted training images: {}'.format(uniq_fineClass))

# extract all images of a specific coarse class from the TESTING data 
idx = []
for i in range(len(coarse_TsLabels)):
    if coarse_TsLabels[i] == 19:
        idx.append(i)

print('Total images with 4 coarse label (Vehicles 2) from TESTING DATASET: {}'.format(len(idx)))
idx = np.array(idx)

# extract all image and corresponding "fine" label and store in test_images, test_labels variable list.
test_images, test_labels = fine_test[idx], fine_TsLabels[idx]
print("Shape of the image testing dataset: {}".format(test_images.shape))
uniq_fineClass = np.unique(test_labels)
print('Fine Class for the extracted testing images: {}'.format(uniq_fineClass))

# normalize image pixel values (0–255 to 0–1)
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.1),
    tf.keras.layers.RandomTranslation(0.1,0.1)
])

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-5
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Safe relabeling using a mapping
label_map = {old: new for new, old in enumerate(uniq_fineClass)}

train_labels = np.array([label_map[l[0]] for l in train_labels])
test_labels = np.array([label_map[l[0]] for l in test_labels])
print("New labels:", np.unique(train_labels))

print("Train label distribution:", np.bincount(train_labels))
print("Test label distribution:", np.bincount(test_labels))

# Select one test image per class (fixed)
selected_indices = []

for class_id in range(len(str_class)):
    idx = np.where(test_labels == class_id)[0][0]  # first occurrence
    selected_indices.append(idx)

print("Selected test indices:", selected_indices)

# Plot few samples from images from the TESTING DATASET
plt.figure(figsize=(15,3))

for i, idx in enumerate(selected_indices):
    plt.subplot(1,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[idx])
    plt.xlabel(str_class[test_labels[idx]])

plt.suptitle("Sample images from the testing dataset")
plt.show()

# BUILD THE MODEL 
model = tf.keras.Sequential([
    Input(shape=(32,32,3)),

    data_augmentation,

    Conv2D(32,(3,3),padding='same',activation='relu'),
    BatchNormalization(),
    Conv2D(32,(3,3),padding='same',activation='relu'),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(64,(3,3),padding='same',activation='relu'),
    BatchNormalization(),
    Conv2D(64,(3,3),padding='same',activation='relu'),
    MaxPooling2D((2,2)),
    Dropout(0.3),

    Conv2D(128,(3,3),padding='same',activation='relu'),
    BatchNormalization(),
    Conv2D(128,(3,3),padding='same',activation='relu'),
    MaxPooling2D((2,2)),
    Dropout(0.4),

    GlobalAveragePooling2D(),

    Dense(256,activation='relu'),
    Dropout(0.5),

    Dense(len(uniq_fineClass),activation='softmax')
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
    loss='sparse_categorical_crossentropy',
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

# start training
metricInfo = model.fit(
    train_images,
    train_labels,
    epochs=80,
    batch_size=64,
    validation_split=0.1,
    callbacks=[lr_scheduler, early_stop]
)

loss = metricInfo.history['loss']
val_loss = metricInfo.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.clf()
plt.plot(epochs, loss, 'g-', label='Training loss')
plt.plot(epochs, val_loss, 'b-', label='Validation loss')
plt.title('Training vs validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# TEST THE MODEL
print(test_images.shape)
print("Class in the testing image: {}".format(np.unique(test_labels)))
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy: ', test_acc)

# Predict using the model
classification = model.predict(test_images)
# Convert probabilities to predicted class index
predicted_labels = np.argmax(classification, axis=1)

# Compute confusion matrix
cm = confusion_matrix(test_labels, predicted_labels)

print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=str_class,
    yticklabels=str_class
)

plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Confusion Matrix")

i = 1
while os.path.exists(f"heatmap/confusion_matrix_{i}.png"):
    i += 1

plt.savefig(f"heatmap/confusion_matrix_{i}.png", dpi=300)
plt.show()

# Display predictions for 5 images
plt.figure(figsize=(10,3))

for i, idx in enumerate(selected_indices):

    pred = np.argmax(classification[idx])
    true = test_labels[idx]

    plt.subplot(1,5,i+1)
    plt.imshow(test_images[idx])
    plt.title(f"Prediction:{str_class[pred]}\nTruth:{str_class[true]}")
    plt.axis('off')

j = 1
while os.path.exists(f"prediction/prediction_{j}.png"):
    j += 1

plt.savefig(f"prediction/prediction_{j}.png", dpi=300)
plt.show()

model.save('number_classification_model.keras')