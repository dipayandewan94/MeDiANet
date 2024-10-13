from tensorflow.keras import datasets, layers, models
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import Model
from tensorflow.keras import callbacks
from tensorflow.keras.layers import BatchNormalization,Conv2D,DepthwiseConv2D,UpSampling2D,Activation,GlobalAveragePooling2D,MaxPool2D,Add,Multiply,Lambda
#from echoAI.Activation.TF_Keras.custom_activation import ELiSH,HardELiSH
from tensorflow.keras.layers import Input,Dense,AveragePooling2D,Flatten,Dropout,ZeroPadding2D,LayerNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import mish, gelu

#from tensorflow.keras.activations import sparsemax
#import keras_tuner as kt
#from keras_tuner.tuners import RandomSearch, Hyperband, BayesianOptimization
#import wandb
#from wandb.keras import WandbCallback, WandbMetricsLogger, WandbModelCheckpoint

#wandb.login()

# strategy = tf.distribute.MirroredStrategy()
# print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

class GRN(layers.Layer):
    def __init__(self, dim, name='grn', **kwargs):
        super().__init__(**kwargs)
        self.gamma = tf.Variable(
            initial_value=tf.zeros((1, 1, 1, dim), dtype=self.compute_dtype),
            trainable=True,
            dtype=self.compute_dtype,
            name=f'{name}/gamma'
        )
        self.beta = tf.Variable(
            initial_value=tf.zeros((1, 1, 1, dim), dtype=self.compute_dtype),
            trainable=True,
            dtype=self.compute_dtype,
            name=f'{name}/beta'
        )

    def call(self, x):
        Gx = tf.norm(x, ord=2, axis=(1,2), keepdims=True)
        Nx = Gx / (tf.math.reduce_mean(Gx, axis=-1, keepdims=True) + 1e-6)
        return ((self.gamma * (x * Nx)) + self.beta) + x


class Mish(layers.Layer):
    def call(self, x):
        return mish(x)

def dilated_residual_block(input, input_channels=None, output_channels=None, kernel_size=(3, 3), dilation = [1,2,3], drop_prob=0, regularization = None):

    """
    full pre-activation residual block
    https://arxiv.org/pdf/1603.05027.pdf
    """
    regularizer = l2(regularization)
    
    if output_channels is None:
        output_channels = input.shape[-1]
    if input_channels is None:
        input_channels = output_channels // 4
    x = BatchNormalization()(input)
    x = Mish()(x)
    x = Conv2D(input_channels, (1, 1))(x)
    x = BatchNormalization()(x)
    x = Mish()(x)
    x1 = Conv2D(input_channels, kernel_size, padding='same', dilation_rate=(dilation[0]),kernel_regularizer = regularizer)(x)
    x1 = Dropout(0.2)(x1)
    x2 = Conv2D(input_channels, kernel_size, padding='same', dilation_rate=(dilation[1]),kernel_regularizer = regularizer)(x)
    x2 = Dropout(0.2)(x2)
    x3 = Conv2D(input_channels, kernel_size, padding='same', dilation_rate=(dilation[2]),kernel_regularizer = regularizer)(x)
    x3 = Dropout(0.2)(x3)
    x = Add()([x1, x2, x3])
    x = BatchNormalization()(x)
    x = Mish()(x)
    x = Conv2D(output_channels, (1, 1), padding='same')(x)
    input = Conv2D(output_channels, (1, 1), padding='same')(input)
    x = Add()([x, input])
    return x


def residual_block(input, input_channels=None, output_channels=None, kernel_size=(3, 3), stride=1, drop_prob=0):
    """
    full pre-activation residual block
    https://arxiv.org/pdf/1603.05027.pdf
    """
    if output_channels is None:
        output_channels = input.shape[-1]
    if input_channels is None:
        input_channels = output_channels // 4
    strides = (stride, stride)
    x = BatchNormalization()(input)
    x = Mish()(x)
    x = Conv2D(input_channels, (1, 1))(x)
    x = BatchNormalization()(x)
    x = Mish()(x)
    x = Conv2D(input_channels, kernel_size, padding='same', strides=stride)(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Mish()(x)
    x = Conv2D(output_channels, (1, 1), padding='same')(x)
    input = Conv2D(output_channels, (1, 1), padding='same', strides=strides)(input)
    x = Add()([x, input])
    return x



def attention_block2_3(input, input_channels=None, output_channels=None, dilation = [1,2,3], drop_prob = 0):
    """
    attention block
    https://arxiv.org/abs/1704.06904
    """

    if input_channels is None:
        input_channels = input.shape[-1]
    if output_channels is None:
        output_channels = input_channels

    # First Residual Block
    input = residual_block(input, drop_prob = drop_prob)

    # Trunc Branch
    output_trunk = input
    output_trunk = dilated_residual_block(output_trunk, dilation = dilation, drop_prob = drop_prob)
    output_trunk = dilated_residual_block(output_trunk, dilation = dilation, drop_prob = drop_prob)

    # Soft Mask Branch

    ## encoder
    ### first down sampling
    output_soft_mask = MaxPool2D(padding='same')(input)  # 32x32
    output_soft_mask = dilated_residual_block(output_soft_mask, drop_prob = drop_prob)

    skip_connections = []
    ## skip connections
    output_skip_connection = dilated_residual_block(output_soft_mask, drop_prob = drop_prob)
    skip_connections.append(output_skip_connection)
    ## down sampling
    output_soft_mask = MaxPool2D(padding='same')(output_soft_mask)
    output_soft_mask = dilated_residual_block(output_soft_mask, drop_prob = drop_prob)
    output_skip_connection = dilated_residual_block(output_soft_mask, drop_prob = drop_prob)
    skip_connections.append(output_skip_connection)
    ## down sampling
    output_soft_mask = MaxPool2D(padding='same')(output_soft_mask)
    output_soft_mask = dilated_residual_block(output_soft_mask, drop_prob = drop_prob)
    output_skip_connection = dilated_residual_block(output_soft_mask, drop_prob = drop_prob)
    skip_connections.append(output_skip_connection)
    ## down sampling
    #output_soft_mask = MaxPool2D(padding='same')(output_soft_mask)
    output_soft_mask = dilated_residual_block(output_soft_mask, drop_prob = drop_prob)

    ## decoder
    skip_connections = list(reversed(skip_connections))
    ## upsampling
    output_soft_mask = dilated_residual_block(output_soft_mask, drop_prob = drop_prob)
    #output_soft_mask = UpSampling2D()(output_soft_mask)
    ## skip connections
    output_soft_mask = Add()([output_soft_mask, skip_connections[0]])
    output_soft_mask = dilated_residual_block(output_soft_mask, drop_prob = drop_prob)
    output_soft_mask = UpSampling2D()(output_soft_mask)
    ## skip connections
    output_soft_mask = Add()([output_soft_mask, skip_connections[1]])
    output_soft_mask = dilated_residual_block(output_soft_mask, drop_prob = drop_prob)
    output_soft_mask = UpSampling2D()(output_soft_mask)
    ## skip connections
    output_soft_mask = Add()([output_soft_mask, skip_connections[2]])

    ### last upsampling
    output_soft_mask = dilated_residual_block(output_soft_mask, drop_prob = drop_prob)
    output_soft_mask = UpSampling2D()(output_soft_mask)

    ## Output
    output_soft_mask = Conv2D(input_channels, (1, 1))(output_soft_mask)
    output_soft_mask = Conv2D(input_channels, (1, 1))(output_soft_mask)
    output_soft_mask = Activation('sigmoid')(output_soft_mask)

    output = Lambda(lambda x: x + 1)(output_soft_mask)
    output = Multiply()([output, output_trunk])  #

    # Last Residual Block
    output = residual_block(output, drop_prob = drop_prob)


    return output

def attention_block2_2(input, input_channels=None, output_channels=None, dilation = [1,2,3], drop_prob=0):
    """
    attention block
    https://arxiv.org/abs/1704.06904
    """

    if input_channels is None:
        input_channels = input.shape[-1]
    if output_channels is None:
        output_channels = input_channels

    # First Residual Block
    input = residual_block(input, drop_prob = drop_prob)

    # Trunc Branch
    output_trunk = input
    output_trunk = dilated_residual_block(output_trunk, dilation = dilation, drop_prob = drop_prob)
    output_trunk = dilated_residual_block(output_trunk, dilation = dilation, drop_prob = drop_prob)

    # Soft Mask Branch

    ## encoder
    ### first down sampling
    output_soft_mask = MaxPool2D(padding='same')(input)  # 32x32
    output_soft_mask = dilated_residual_block(output_soft_mask, drop_prob = drop_prob)

    skip_connections = []
    ## skip connections
    output_skip_connection = dilated_residual_block(output_soft_mask, drop_prob = drop_prob)
    skip_connections.append(output_skip_connection)
    ## down sampling
    output_soft_mask = MaxPool2D(padding='same')(output_soft_mask)
    output_soft_mask = dilated_residual_block(output_soft_mask, drop_prob = drop_prob)
    output_skip_connection = dilated_residual_block(output_soft_mask, drop_prob = drop_prob)
    skip_connections.append(output_skip_connection)
    ## down sampling
    #output_soft_mask = MaxPool2D(padding='same')(output_soft_mask)
    output_soft_mask = dilated_residual_block(output_soft_mask, drop_prob = drop_prob)

    ## decoder
    skip_connections = list(reversed(skip_connections))
    ## upsampling
    output_soft_mask = dilated_residual_block(output_soft_mask, drop_prob = drop_prob)
    #output_soft_mask = UpSampling2D()(output_soft_mask)
    ## skip connections
    output_soft_mask = Add()([output_soft_mask, skip_connections[0]])
    output_soft_mask = dilated_residual_block(output_soft_mask, drop_prob = drop_prob)
    output_soft_mask = UpSampling2D()(output_soft_mask)
    ## skip connections
    output_soft_mask = Add()([output_soft_mask, skip_connections[1]])
    output_soft_mask = dilated_residual_block(output_soft_mask, drop_prob = drop_prob)
    output_soft_mask = UpSampling2D()(output_soft_mask)
    ## skip connections
    ## Output
    output_soft_mask = Conv2D(input_channels, (1, 1))(output_soft_mask)
    output_soft_mask = Conv2D(input_channels, (1, 1))(output_soft_mask)
    output_soft_mask = Activation('sigmoid')(output_soft_mask)

    output = Lambda(lambda x: x + 1)(output_soft_mask)
    output = Multiply()([output, output_trunk])  #

    # Last Residual Block
    output = residual_block(output, drop_prob = drop_prob)


    return output

def attention_block2_1(input, input_channels=None, output_channels=None, dilation = [1,2,3], drop_prob = 0):
    """
    attention block
    https://arxiv.org/abs/1704.06904
    """

    if input_channels is None:
        input_channels = input.shape[-1]
    if output_channels is None:
        output_channels = input_channels

    # First Residual Block
    input = residual_block(input, drop_prob = drop_prob)

    # Trunc Branch
    output_trunk = input
    output_trunk = dilated_residual_block(output_trunk, dilation = dilation, drop_prob = drop_prob)
    output_trunk = dilated_residual_block(output_trunk, dilation = dilation, drop_prob = drop_prob)

    # Soft Mask Branch

    ## encoder
    ### first down sampling
    output_soft_mask = MaxPool2D(padding='same')(input)  # 32x32
    output_soft_mask = dilated_residual_block(output_soft_mask, drop_prob = drop_prob)

    skip_connections = []
    ## skip connections
    output_skip_connection = dilated_residual_block(output_soft_mask, drop_prob = drop_prob)
    skip_connections.append(output_skip_connection)
    ## down sampling
    #output_soft_mask = MaxPool2D(padding='same')(output_soft_mask)
    output_soft_mask = dilated_residual_block(output_soft_mask, drop_prob = drop_prob)

    ## decoder
    skip_connections = list(reversed(skip_connections))
    #for i in range(encoder_depth - 1):
    ## upsampling
    output_soft_mask = dilated_residual_block(output_soft_mask, drop_prob = drop_prob)
    #output_soft_mask = UpSampling2D()(output_soft_mask)
    ## skip connections
    output_soft_mask = Add()([output_soft_mask, skip_connections[0]])
    output_soft_mask = dilated_residual_block(output_soft_mask, drop_prob = drop_prob)
    output_soft_mask = UpSampling2D()(output_soft_mask)
    ## skip connections
    ## skip connections
    ## Output
    output_soft_mask = Conv2D(input_channels, (1, 1))(output_soft_mask)
    output_soft_mask = Conv2D(input_channels, (1, 1))(output_soft_mask)
    output_soft_mask = Activation('sigmoid')(output_soft_mask)

    output = Lambda(lambda x: x + 1)(output_soft_mask)
    output = Multiply()([output, output_trunk])  #

    # Last Residual Block
    output = residual_block(output, drop_prob = drop_prob)


    return output

def MeDiANet69(shape=(28, 28, 3), n_channels=32, n_classes=100, dropout=0, regularization=0.01, drop_prob=0):
    """
    Attention-56 ResNet for Cifar10 Dataset
    https://arxiv.org/abs/1704.06904
    """
    regularizer = l2(regularization)
    
    input_ = Input(shape=shape)
    #padded_input_data = ZeroPadding2D( (2, 2) )(input_)
    x = Conv2D(n_channels, (7, 7), padding='same', strides=2)(input_)
    x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    x = Mish()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)  # 16x16
    #x = residual_block(x, output_channels=16)
    #x = residual_block(x, output_channels=32, stride=2)

    x = residual_block(x, output_channels=n_channels, drop_prob = drop_prob*(1/8))
    x = attention_block2_3(x, dilation = [4,8,12], drop_prob = drop_prob*(2/8))
    #x = attention_block2_3(x, dilation = [4,8,12])
    #x = attention_block2_3(x, dilation = [4,8,12])

    x = residual_block(x, output_channels=2*n_channels, stride=2, drop_prob = drop_prob*(3/8))  # 8x8
    x = attention_block2_2(x, dilation = [2,4,6], drop_prob = drop_prob*(4/8))
    #x = attention_block2_2(x, dilation = [2,4,6])
    #x = attention_block2_2(x, dilation = [2,4,6])
    #x = attention_block2_2(x, dilation = [2,4,8])
    
    
    x = residual_block(x, output_channels=4*n_channels, stride=2, drop_prob = drop_prob*(5/8))  # 8x8
    x = attention_block2_1(x, dilation = [1,2,3], drop_prob = drop_prob*(6/8))
    #x = attention_block2_1(x, dilation = [1,2,3])
    #x = attention_block2_1(x, dilation = [1,2,3])
    #x = attention_block2_1(x, dilation = [1,2,4])
    #x = attention_block2_1(x, dilation = [1,2,4])
    #x = attention_block2_1(x, dilation = [1,2,4])

    x = residual_block(x, output_channels=8*n_channels, stride=2, drop_prob = drop_prob*(7/8))  # 4x4
    #x = residual_block(x, output_channels=8*n_channels, drop_prob = drop_prob)
    x = residual_block(x, output_channels=256)

    x = AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(x)  # 1x1
    x = Flatten()(x)
    if dropout:
        x = Dropout(dropout)(x)
    output = Dense(2*n_channels, kernel_regularizer=regularizer)(x)
    output = Mish()(output)
    output = Dense(n_classes, kernel_regularizer=l2(0.002))(output)
    output = Activation('softmax', dtype='float32', name='predictions')(output)
    model = Model(input_, output)
    return model

def MeDiANet117(shape=(28, 28, 3), n_channels=32, n_classes=100, dropout=0, regularization=0.01, drop_prob=0):
    """
    Attention-56 ResNet for Cifar10 Dataset
    https://arxiv.org/abs/1704.06904
    """
    regularizer = l2(regularization)
    
    input_ = Input(shape=shape)
    #padded_input_data = ZeroPadding2D( (2, 2) )(input_)
    x = Conv2D(n_channels, (7, 7), padding='same', strides=2)(input_)
    x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    x = Mish()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)  # 16x16
    #x = residual_block(x, output_channels=16)
    #x = residual_block(x, output_channels=32, stride=2)

    x = residual_block(x, output_channels=n_channels, drop_prob = drop_prob*(1/8))
    x = attention_block2_3(x, dilation = [4,8,12], drop_prob = drop_prob*(2/8))
    #x = attention_block2_3(x, dilation = [4,8,12])
    #x = attention_block2_3(x, dilation = [4,8,12])

    x = residual_block(x, output_channels=2*n_channels, stride=2, drop_prob = drop_prob*(3/8))  # 8x8
    x = attention_block2_2(x, dilation = [2,4,6], drop_prob = drop_prob*(4/8))
    x = attention_block2_2(x, dilation = [2,4,6])
    #x = attention_block2_2(x, dilation = [2,4,6])
    #x = attention_block2_2(x, dilation = [2,4,8])
    
    
    x = residual_block(x, output_channels=4*n_channels, stride=2, drop_prob = drop_prob*(5/8))  # 8x8
    x = attention_block2_1(x, dilation = [1,2,3], drop_prob = drop_prob*(6/8))
    x = attention_block2_1(x, dilation = [1,2,3])
    x = attention_block2_1(x, dilation = [1,2,3])
    #x = attention_block2_1(x, dilation = [1,2,4])
    #x = attention_block2_1(x, dilation = [1,2,4])
    #x = attention_block2_1(x, dilation = [1,2,4])

    x = residual_block(x, output_channels=8*n_channels, stride=2, drop_prob = drop_prob*(7/8))  # 4x4
    x = residual_block(x, output_channels=8*n_channels, drop_prob = drop_prob)
    x = residual_block(x, output_channels=256)

    x = AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(x)  # 1x1
    x = Flatten()(x)
    if dropout:
        x = Dropout(dropout)(x)
    output = Dense(2*n_channels, kernel_regularizer=regularizer)(x)
    output = Mish()(output)
    output = Dense(n_classes, kernel_regularizer=l2(0.002))(output)
    output = Activation('softmax', dtype='float32', name='predictions')(output)
    model = Model(input_, output)
    return model