"""ResNet50 model for Keras.

# Reference:

- [Deep Residual Learning for Image Recognition](
    https://arxiv.org/abs/1512.03385) (CVPR 2016 Best Paper Award)

Adapted from code contributed by BigMoyan.
"""
import keras.layers as layers
import keras.models as models


def residual_stack(X,Filters,Seq,max_pool):
    #1*1 Conv Linear
    X = layers.Conv2D(Filters, (1, 1), padding='same', name=Seq+"_conv1")(X)
    #Residual Unit 1
    X_shortcut = X
    X = layers.Conv2D(Filters, (2, 3), padding='same',activation="relu",name=Seq+"_conv2")(X)
    X = layers.Conv2D(Filters, (2, 3), padding='same', name=Seq+"_conv3")(X)
    X = layers.add([X,X_shortcut])
    X = layers.Activation("relu")(X)
    #Residual Unit 2
    X_shortcut = X
    X = layers.Conv2D(Filters, (2, 3), padding='same',activation="relu",name=Seq+"_conv4")(X)
    X = layers.Conv2D(Filters, (2, 3), padding='same', name=Seq+"_conv5")(X)
    X = layers.Add()([X,X_shortcut])
    X = layers.Activation("relu")(X)
    #MaxPooling
    if max_pool:
        X = layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid')(X)
    return X

def build_model(img_input,
             input_shape=None,
             classes=1000):
    x = layers.Reshape((input_shape[0], input_shape[1], -1))(img_input)

    X = residual_stack(x, 32, "ReStk1", False)  # shape:(1,512,32)
    # X = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 2), padding='valid')(X)

    # Residual Srack 2
    X = residual_stack(X, 32, "ReStk2", True)  # shape:(1,256,32)

    # Residual Srack 3
    X = residual_stack(X, 32, "ReStk3", True)  # shape:(1,128,32)

    # Residual Srack 4
    X = residual_stack(X, 32, "ReStk4", True)  # shape:(1,64,32)

    # Residual Srack 5
    X = residual_stack(X, 32, "ReStk5", True)  # shape:(1,32,32)

    # Residual Srack 6
    X = residual_stack(X, 32, "ReStk6", True)  # shape:(1,16,32)

    # Full Con 1
    X = layers.Flatten()(X)
    X = layers.Dense(128, activation='selu', name="dense1")(X)
    X = layers.AlphaDropout(0.3)(X)
    # Full Con 2
    X = layers.Dense(128, activation='selu', name="dense2")(X)
    X = layers.AlphaDropout(0.3)(X)
    # Full Con 3
    X = layers.Dense(classes, name="dense3")(X)

    outputs = layers.Activation('softmax')(X)

    resNet = models.Model(inputs=[img_input], outputs=[outputs])

    return resNet

# def identity_block(input_tensor, kernel_size, filters, stage, block):
#     """The identity block is the block that has no conv layer at shortcut.
#
#     # Arguments
#         input_tensor: input tensor
#         kernel_size: default 3, the kernel size of
#             middle conv layer at main path
#         filters: list of integers, the filters of 3 conv layer at main path
#         stage: integer, current stage label, used for generating layer names
#         block: 'a','b'..., current block label, used for generating layer names
#
#     # Returns
#         Output tensor for the block.
#     """
#     filters1, filters2, filters3 = filters
#     if backend.image_data_format() == 'channels_last':
#         bn_axis = 3
#     else:
#         bn_axis = 1
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'
#
#     x = layers.Conv2D(filters1, (1, 1),
#                       kernel_initializer='he_normal',
#                       name=conv_name_base + '2a')(input_tensor)
#     x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
#     x = layers.Activation('relu')(x)
#
#     x = layers.Conv2D(filters2, kernel_size,
#                       padding='same',
#                       kernel_initializer='he_normal',
#                       name=conv_name_base + '2b')(x)
#     x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
#     x = layers.Activation('relu')(x)
#
#     x = layers.Conv2D(filters3, (1, 1),
#                       kernel_initializer='he_normal',
#                       name=conv_name_base + '2c')(x)
#     x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
#
#     x = layers.add([x, input_tensor])
#     x = layers.Activation('relu')(x)
#     return x
#
#
# def conv_block(input_tensor,
#                kernel_size,
#                filters,
#                stage,
#                block,
#                strides=(2, 2)):
#     """A block that has a conv layer at shortcut.
#
#     # Arguments
#         input_tensor: input tensor
#         kernel_size: default 3, the kernel size of
#             middle conv layer at main path
#         filters: list of integers, the filters of 3 conv layer at main path
#         stage: integer, current stage label, used for generating layer names
#         block: 'a','b'..., current block label, used for generating layer names
#         strides: Strides for the first conv layer in the block.
#
#     # Returns
#         Output tensor for the block.
#
#     Note that from stage 3,
#     the first conv layer at main path is with strides=(2, 2)
#     And the shortcut should have strides=(2, 2) as well
#     """
#     filters1, filters2, filters3 = filters
#     if backend.image_data_format() == 'channels_last':
#         bn_axis = 3
#     else:
#         bn_axis = 1
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'
#
#     x = layers.Conv2D(filters1, (1, 1), strides=strides,
#                       kernel_initializer='he_normal',
#                       name=conv_name_base + '2a')(input_tensor)
#     x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
#     x = layers.Activation('relu')(x)
#
#     x = layers.Conv2D(filters2, kernel_size, padding='same',
#                       kernel_initializer='he_normal',
#                       name=conv_name_base + '2b')(x)
#     x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
#     x = layers.Activation('relu')(x)
#
#     x = layers.Conv2D(filters3, (1, 1),
#                       kernel_initializer='he_normal',
#                       name=conv_name_base + '2c')(x)
#     x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
#
#     shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
#                              kernel_initializer='he_normal',
#                              name=conv_name_base + '1')(input_tensor)
#     shortcut = layers.BatchNormalization(
#         axis=bn_axis, name=bn_name_base + '1')(shortcut)
#
#     x = layers.add([x, shortcut])
#     x = layers.Activation('relu')(x)
#     return x
#
#
# def ResNet50(
#              input_shape=None,
#              classes=1000
#             ):
#
#     img_input = layers.Input(shape=input_shape)
#
#     inputs = layers.Reshape((input_shape[0], input_shape[1], -1 ))(img_input)
#
#     x = conv_block(inputs, 2, [64, 64, 256], stage=2, block='a', strides=(1, 1))
#     x = identity_block(x, 2, [64, 64, 256], stage=2, block='b')
#     x = identity_block(x, 2, [64, 64, 256], stage=2, block='c')
#
#     x = conv_block(x, 2, [128, 128, 512], stage=3, block='a')
#     x = identity_block(x, 2, [128, 128, 512], stage=3, block='b')
#     x = identity_block(x, 2, [128, 128, 512], stage=3, block='c')
#     x = identity_block(x, 2, [128, 128, 512], stage=3, block='d')
#
#     x = conv_block(x, 2, [256, 256, 1024], stage=4, block='a')
#     x = identity_block(x, 2, [256, 256, 1024], stage=4, block='b')
#     x = identity_block(x, 2, [256, 256, 1024], stage=4, block='c')
#     x = identity_block(x, 2, [256, 256, 1024], stage=4, block='d')
#     x = identity_block(x, 2, [256, 256, 1024], stage=4, block='e')
#     x = identity_block(x, 2, [256, 256, 1024], stage=4, block='f')
#
#     x = conv_block(x, 2, [512, 512, 2048], stage=5, block='a')
#     x = identity_block(x, 2, [512, 512, 2048], stage=5, block='b')
#     x = identity_block(x, 2, [512, 512, 2048], stage=5, block='c')
#
#     x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
#     x = layers.Dense(classes, activation='softmax', name='fc1000')(x)
#
#     model = models.Model(inputs, x, name='resnet50')
#
#     return model
