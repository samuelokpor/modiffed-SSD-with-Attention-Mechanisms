import tensorflow as tf
from keras import layers
from ssd_resnet import SSD_ResNet_50
from focal_loss import focal_loss

class CBAM(layers.Layer):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.avg_pool = layers.GlobalAvgPool2D()
        self.max_pool = layers.GlobalMaxPool2D()

        self.fc1 = layers.Conv2D(channels // reduction, kernel_size=1, padding='same')
        self.relu = layers.ReLU()
        self.fc2 = layers.Conv2D(channels, kernel_size=1, padding='same')

        self.sigmoid_channel = layers.Activation('sigmoid')
        self.conv_after_concat = layers.Conv2D(1, kernel_size=7, padding='same')
        self.sigmoid_spatial = layers.Activation('sigmoid')

    def call(self, x):
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        x = tf.concat([avg_pool, max_pool], axis=-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid_channel(x)

        x = tf.concat([x, x], axis=-1)
        x = self.conv_after_concat(x)
        x = self.sigmoid_spatial(x)

        return x


class ImprovedSSD_ResNet_50(tf.keras.Model):
    def __init__(self, num_classes):
        super(ImprovedSSD_ResNet_50, self).__init__()
        self.base = SSD_ResNet_50(num_classes=num_classes)
        self.cbam1 = CBAM(channels=1024)
        self.cbam2 = CBAM(channels=512)
        self.cbam3 = CBAM(channels=512)
        self.cbam4 = CBAM(channels=256)
        self.cbam5 = CBAM(channels=256)
        self.cbam6 = CBAM(channels=256)
        self.conv7 = layers.Conv2D(256, kernel_size=1, padding='same')
        self.conv8 = layers.Conv2D(256, kernel_size=1, padding='same')
        self.conv9 = layers.Conv2D(256, kernel_size=1, padding='same')

    def call(self, x, y_true=None):
        c3, c4, c5 = self.base(x)

        c6 = self.cbam1(c5)
        c6 = tf.math.multiply(c6, c5) # Element-wise multiplication between CBAM output and c5 feature map
        c7 = self.conv7(c6)
        c8 = self.cbam2(c4)
        c8 = tf.image.resize(c8, size=(c7.shape[1], c7.shape[2]))  # Upsampling CBAM output to match c7 feature map size
        c8 = tf.math.multiply(c8, c7)  # Element-wise multiplication between CBAM output and c7 feature map

        c9 = self.conv8(c8)
        c10 = self.cbam3(c3)
        c10 = tf.image.resize(c10, size=(c9.shape[1], c9.shape[2]))  # Upsampling CBAM output to match c9 feature map size
        c10 = tf.math.multiply(c10, c9)  # Element-wise multiplication between CBAM output and c9 feature map

        p6 = self.conv9(c6)
        p7 = tf.nn.max_pool2d(c7, ksize=3, strides=2, padding='SAME')
        p8 = tf.nn.max_pool2d(c9, ksize=3, strides=2, padding='SAME')
        p9 = tf.nn.max_pool2d(c10, ksize=3, strides=2, padding='SAME')

        # Reshape feature maps to be compatible with SSD header
        p6 = tf.transpose(p6, perm=[0, 3, 1, 2])
        p7 = tf.transpose(p7, perm=[0, 3, 1, 2])
        p8 = tf.transpose(p8, perm=[0, 3, 1, 2])
        p9 = tf.transpose(p9, perm=[0, 3, 1, 2])

        predictions = self.base.ssd_header([p6, p7, p8, p9])

        if y_true is not None:
            loss = focal_loss(y_true, predictions)
            return loss

        return predictions