import tensorflow as tf
from keras import layers


class ResidualUnit(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, **kwargs):
        super(ResidualUnit, self).__init__(**kwargs)
        self.filters = filters
        self.strides = strides
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=strides, padding='same', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=strides, padding='same', use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()

    def build(self, input_shape):
        if input_shape[-1] != self.filters:
            self.projection = tf.keras.Sequential([
                tf.keras.layers.Conv2D(self.filters, kernel_size=1, strides=self.strides, use_bias=False),
                tf.keras.layers.BatchNormalization()
            ])
        else:
            self.projection = tf.keras.layers.Lambda(lambda x: x)

    def call(self, inputs):
        identity = self.projection(inputs)
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x += identity
        x = self.relu(x)
        return x

class SSD_ResNet_50(tf.keras.Model):
    def __init__(self, num_classes):
        super(SSD_ResNet_50, self).__init__()
        self.conv1 = layers.Conv2D(64, kernel_size=7, strides=2, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.max_pool = layers.MaxPool2D(pool_size=3, strides=2, padding='same')

        self.layer1 = self._make_layer(64, blocks=3)
        self.layer2 = self._make_layer(128, blocks=4, stride=2)
        self.layer3 = self._make_layer(256, blocks=6, stride=2)
        self.layer4 = self._make_layer(512, blocks=3, stride=2)

        self.fpn = self._make_fpn_layers()

        self.cls_layers = self._make_cls_layers(num_classes=num_classes)
        self.reg_layers = self._make_reg_layers()

    def _make_layer(self, filters, blocks, stride=1):
        downsample = None
        if stride != 1 or filters * 4 != 256:
            downsample = layers.Conv2D(filters * 4, kernel_size=1, strides=stride, padding='same')
            downsample = tf.keras.Sequential([downsample, layers.BatchNormalization()])

        layers_list = [ResidualUnit(filters, stride, downsample)]
        for _ in range(1, blocks):
            layers_list.append(ResidualUnit(filters))

        return tf.keras.Sequential(layers_list)

    def _make_fpn_layers(self):
        P3 = layers.Conv2D(256, kernel_size=1, strides=1, padding='same')
        P4 = layers.Conv2D(256, kernel_size=1, strides=1, padding='same')
        P5 = layers.Conv2D(256, kernel_size=1, strides=1, padding='same')
        P6 = layers.Conv2D(256, kernel_size=3, strides=2, padding='same')
        P7 = layers.Conv2D(256, kernel_size=3, strides=2, padding='same')

        return [P3, P4, P5, P6, P7]
    def _make_cls_layers(self, num_classes):
        cls_layers = []
        for _ in range(5):
            cls_layers.append(layers.Conv2D(num_classes, kernel_size=3, strides=1, padding='same'))
        return cls_layers

    def _make_reg_layers(self):
        reg_layers = []
        for _ in range(5):
            reg_layers.append(layers.Conv2D(4, kernel_size=3, strides=1, padding='same'))
        return reg_layers

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        fpn = [x]
        for i in range(1, 5):
            fpn.append(tf.nn.max_pool(fpn[-1], ksize=1, strides=2, padding='SAME'))

        features = []
        for fpn_feature, conv_layer in zip(fpn, self.fpn):
            features.append(conv_layer(fpn_feature))

        cls_outputs = []
        reg_outputs = []
        for feature in features:
            cls_output = feature
            reg_output = feature

            for cls_layer in self.cls_layers:
                cls_output = cls_layer(cls_output)

            for reg_layer in self.reg_layers:
                reg_output = reg_layer(reg_output)

            cls_outputs.append(cls_output)
            reg_outputs.append(reg_output)

        return [cls_outputs, reg_outputs]
