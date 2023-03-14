import tensorflow as tf
from keras import layers
from keras.layers import Reshape
from keras.preprocessing.image import ImageDataGenerator

# Load the SSDResNet50 architecture
model = tf.keras.applications.ResNet50(
    include_top=False,
    weights=None,
    input_shape=(224, 224, 3)
)

class CBAM(tf.keras.layers.Layer):
    def __init__(self, reduction_ratio=8, name=None):
        super(CBAM, self).__init__(name=name)
        self.reduction_ratio = reduction_ratio
        self.fc1 = tf.keras.layers.Dense(units=1024, activation='relu')
        self.fc2 = tf.keras.layers.Dense(units=512, activation='relu')
        self.conv2d_1 = tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding='same')
        self.conv2d_2 = tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding='same')
        self.reshape = tf.keras.layers.Reshape((-1, 1, 1))
    
    def call(self, inputs):
        # Global Average Pooling
        gap = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        gap_reshaped = tf.keras.layers.Reshape((1, 1, gap.shape[-1]))(gap)
        print(gap_reshaped.shape)
        mp = tf.keras.layers.Conv2D(gap.shape[-1], (1, 1), padding='same')(inputs)
        print(mp.shape)
        merged = tf.keras.layers.Concatenate(axis=-1)([gap_reshaped, mp])

        # FC layers
        x = self.fc1(merged)
        x = self.fc2(x)

        # Channel Attention
        ca = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        ca = self.conv2d_1(ca)
        ca = tf.nn.sigmoid(ca)
        x = x * ca

        # Spatial Attention
        sa = self.conv2d_2(x)
        sa = self.reshape(sa)
        sa = tf.nn.sigmoid(sa)
        x = x * sa

        return x
    

class CBAMResBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, stride=1, downsample=None):
        super(CBAMResBlock, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=filters,
                                        kernel_size=kernel_size,
                                        strides=stride,
                                        padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.cbam = CBAM()
        self.conv2 = tf.keras.layers.Conv2D(filters=filters,
                                            kernel_size=kernel_size,
                                            strides=1,
                                            padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.downsample = downsample

    def call(self, inputs, training=False):
        identity = inputs

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.cbam(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        if self.downsample is not None:
            identity = self.downsample(inputs)

        x += identity
        x = self.relu(x)

        return x
    
def make_cbam_resnet50(input_shape, num_classes, reduction_ratio=8):
    inputs = tf.keras.Input(shape=input_shape)

    # Initial Convolutional Layers
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # Residual Layers with CBAM
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = CBAM(reduction_ratio=reduction_ratio)(x)
    shortcut = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same')(inputs)
    shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = CBAM(reduction_ratio=reduction_ratio)(x)
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = CBAM(reduction_ratio=reduction_ratio)(x)
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)

    # Final Classification Layers
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(units=num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x, name='cbam_resnet50')
    return model

# Sample training script
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory('./data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory('./data/test',
    target_size=(224, 224),batch_size=32,
class_mode='categorical',
subset='validation'
)

model = make_cbam_resnet50(input_shape=(224, 224, 3), num_classes=10)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, validation_data=validation_generator, epochs=10)