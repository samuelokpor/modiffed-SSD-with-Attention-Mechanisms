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
        self.reshape = Reshape((-1, 1, 1))
    
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

        # First conv block of the ResNet block
        self.conv1 = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()

        # Second conv block of the ResNet block
        self.conv2 = layers.Conv2D(filters, kernel_size, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()

        # Attention mechanism
        self.cbam = CBAM(filters)

        # Downsample convolution to match the shapes when applying shortcut connections
        self.downsample = downsample

        # Shortcut connection
        self.shortcut = tf.keras.Sequential()
        if downsample:
            self.shortcut.add(downsample)
        if stride > 1:
            self.shortcut.add(layers.AveragePooling2D(pool_size=stride, strides=stride, padding='same'))

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = layers.ReLU()(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # Apply attention mechanism
        x = self.cbam(x)

        shortcut = self.shortcut(inputs)
        x += shortcut
        x = layers.ReLU()(x)

        return x
    
def make_cbam_resnet50():
    # Input layer
    inputs = layers.Input(shape=(224, 224, 3))

    # Initial convolutions
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # Residual blocks with attention
    x = CBAMResBlock(filters=64, kernel_size=64, stride=1)(x)
    x = CBAMResBlock(256, 64, stride=2, downsample=layers.Conv2D(256, kernel_size=1, strides=2, use_bias=False))(x)
    x = CBAMResBlock(256, 64, stride=1)(x)
    x = CBAMResBlock(256, 64, stride=1)(x)

    x = CBAMResBlock(512, 128, stride=2, downsample=layers.Conv2D(512, kernel_size=1, strides=2, use_bias=False))(x)
    x = CBAMResBlock(512, 128, stride=1)(x)
    x = CBAMResBlock(512, 128, stride=1)(x)
    x = CBAMResBlock(512, 128, stride=1)(x)

    x =CBAMResBlock(1024, 256, stride=2, downsample=layers.Conv2D(1024, kernel_size=1, strides=2, use_bias=False))(x)
    x = CBAMResBlock(1024, 256, stride=1)(x)
    x = CBAMResBlock(1024, 256, stride=1)(x)
    # SSD model body
    x = CBAMResBlock(512, 128, stride=1)(x)
    x = CBAMResBlock(256, 64, stride=1)(x)

    # Detection layers
    source_layers = [    layers.Conv2D(256, kernel_size=1, strides=1, padding='same', name=f'conv{i}') for i in range(3)]
    ssd_layers = [    layers.Conv2D(256, kernel_size=3, strides=1, padding='same', name=f'ssd{i}') for i in range(6)]

    # Create SSD model with source and ssd layers
    model = SSD(source_layers, ssd_layers)
    return tf.keras.Model(inputs=inputs, outputs=model.call(x))
    

model = make_cbam_resnet50()
model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                                            loss='binary_crossentropy',
                                                                metrics=['accuracy']
                                                                                        )
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory('./data/train/',target_size=(224, 224),batch_size=32,class_mode='binary')
test_generator = test_datagen.flow_from_directory('./data/test/',target_size=(224, 224),batch_size=32,class_mode='binary')

    #Train the model
history = model.fit(train_generator, epochs=10, validation_data=test_generator)
    #Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)