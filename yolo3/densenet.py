import keras.layers as layers
import keras.backend as backend
import keras.models as models
from keras import backend as K

def dense_block(x, blocks, name):
    for i in range(blocks):
        x = conv_block(x, 15, name=name + '_block' + str(i + 1))
    return x

def transition_block(x, reduction, name):
    bn_axis = 3
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv2D(int(backend.int_shape(x)[bn_axis] * reduction), 1,
                      use_bias=False,
                      name=name + '_conv')(x)
    x = layers.AveragePooling2D((2,2), name=name + '_pool')(x)
    return x

def conv_block(x, growth_rate, name):
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    x1 = layers.BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = layers.Conv2D(4 * growth_rate, 1,
                       use_bias=False,
                       name=name + '_1_conv')(x1)
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = layers.Conv2D(growth_rate, (3,3),
                       padding='same',
                       use_bias=False,
                       name=name + '_2_conv')(x1)
    x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x

def SEN_self_att(x,r,name):
    x_self = x
    chanel = K.int_shape(x)[-1]
    w = K.int_shape(x)[-2]
    # x = layers.Reshape([1,w,-1])(x)
    x = layers.GlobalAveragePooling2D(name=name+'GAP')(x)
    x = layers.Dense(int(chanel/r),activation='relu',name=name+'Dense_down')(x)
    x = layers.Dense(chanel,activation='sigmoid',name=name+'Dense_up')(x)
    x = layers.Multiply(name=name+'Multiply')([x_self,x])
    return x


def DenseNet(img_input,blocks,
             include_top=True,
             input_shape=None,
             pooling=None,
             classes=1000
             ):
    # img_input = layers.Input(shape=input_shape)

    x = layers.Reshape((input_shape[0], input_shape[1], -1))(img_input)

    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = SEN_self_att(x,r=4,name='SEN2')

    x = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = SEN_self_att(x, r=4, name='SEN3')

    x = dense_block(x, blocks[2], name='conv4')
    x = transition_block(x, 0.5, name='pool4')
    x = SEN_self_att(x, r=4, name='SEN4')

    x = dense_block(x, blocks[3], name='conv5')
    x = transition_block(x, 0.5, name='pool5')
    x = SEN_self_att(x, r=4, name='SEN5')

    if include_top:
        # x = layers.Reshape([1,K.int_shape(x)[-2],-1])(x)
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        # x = layers.Flatten()(x)
        x = layers.Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors
    #
    #  of `input_tensor`.
    inputs = img_input

    # Create model.
    if blocks == [6, 12, 24, 16]:
        model = models.Model(inputs, x, name='densenet121')
    elif blocks == [6, 12, 32, 32]:
        model = models.Model(inputs, x, name='densenet169')
    elif blocks == [6, 12, 48, 32]:
        model = models.Model(inputs, x, name='densenet201')
    else:
        model = models.Model(inputs, x, name='densenet')

    return model