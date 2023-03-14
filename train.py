import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.losses import Huber
from keras.metrics import Mean
from improved_ssd2 import ImprovedSSD_ResNet_50
import numpy as np
import os
import xml.etree.ElementTree as ET
from keras.utils import to_categorical

# Set the input image size and the batch size
IMG_SIZE = (960, 600)
BATCH_SIZE = 8

# Define the path to the data directory
DATA_DIR = './data/'

# Define the number of classes in the dataset
NUM_CLASSES = 2

# Define the number of anchor boxes per grid cell
NUM_ANCHORS = 9

# Define the learning rate and the number of epochs
LR = 0.001
EPOCHS = 50

# Build the ImprovedSSD_ResNet_50 model
model = ImprovedSSD_ResNet_50(num_classes=NUM_CLASSES)

# Define the optimizer and the loss function
optimizer = Adam(lr=LR)
loss_fn = Huber()

# Compile the model
model.compile(optimizer=optimizer, loss=loss_fn)

# Define the data generator function
# Define the data generator function
def data_generator(data_dir, batch_size):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    while True:
        # Generate batches of samples
        batch_paths = np.random.choice(os.listdir(train_dir), size=batch_size)
        batch_input = []
        batch_output = []

        for image_file in batch_paths:
            try:
                # Read the image file
                image_path = os.path.join(train_dir, image_file)
                image = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
                image = tf.keras.preprocessing.image.img_to_array(image)
                batch_input.append(image)

                # Read the corresponding xml file
                annotation_path = os.path.join(train_dir, os.path.splitext(image_file)[0] + '.xml')
                tree = ET.parse(annotation_path)
                root = tree.getroot()

                # Extract the class label from the xml file
                class_name = root.find('object').find('name').text
                if class_name == 'OK':
                    class_index = 1
                else:
                    class_index = 0

                # Convert the class index to a one-hot vector
                class_vector = to_categorical(class_index, NUM_CLASSES)
                batch_output.append(class_vector)
            except:
                # If the image file or annotations cannot be read, skip this iteration
                continue

        # If all images in the batch do not have annotations, skip this batch
        if len(batch_input) == 0:
            continue

        # Preprocess the input batch
        batch_input = tf.keras.applications.resnet.preprocess_input(np.array(batch_input))

        yield np.array(batch_input), np.array(batch_output)

# Create data generators for training and validation
train_generator = data_generator(DATA_DIR, BATCH_SIZE)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    directory=os.path.join(DATA_DIR, 'val'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False)
print('Number of training samples:', len(os.listdir(os.path.join(DATA_DIR, 'train'))))
print('Number of validation samples:', val_generator.n)
print('Batch size:', BATCH_SIZE)

# Define the model checkpoint and tensorboard callbacks
checkpoint_cb = ModelCheckpoint('model.h5', save_best_only=True, monitor='val_loss', mode='min')
tensorboard_cb = TensorBoard(log_dir='logs')

# Train the model
history = model.fit(train_generator,
                    epochs=EPOCHS,
                    validation_data=val_generator,
                    callbacks=[checkpoint_cb, tensorboard_cb])

# Save the final model
model.save('final_model.h5')