def eval():
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    #os.environ["TF_USE_LEGACY_KERAS"] = "1"

    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    import numpy as np

    #import tf_keras
    import keras
    import numpy as np
    import random

    random.seed(153)
    np.random.seed(153)
    tf.keras.utils.set_random_seed(153)

    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    from dataloader_tensorflow import create_dataset
    from model_tensorflow import MeDiANet69, MeDiANet117

    from tensorflow.keras.models import load_model

    from tensorflow.keras import layers
    from tensorflow.keras.activations import mish, gelu

    model_path = '/saved_models/tensorflow/MeDiANet_base_69.tf'
    test_path = '/newdataset/test/'

    batch_size = 512

    test_data = create_dataset(test_path, batch_size=batch_size, shuffle=False)

    class Mish(layers.Layer):
        def call(self, x):
            return mish(x)
        

    tf.keras.config.enable_unsafe_deserialization()
    # saved_model = load_model('/home/dipayan/MwDiANetLarge2.tf')
    saved_model = load_model(model_path, custom_objects={'Mish': Mish})

    _, test_acc = saved_model.evaluate(test_data, verbose=1)
    print('Test: %.5f' % (test_acc))

if __name__ == "__main__":
    eval()
