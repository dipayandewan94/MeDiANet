def train():
    import os
    import random
    import numpy as np
    from dataloader_tensorflow import create_dataset
    from model_tensorflow import MeDiANet69, MeDiANet117
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint, CSVLogger
    from tensorflow.keras.models import load_model
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="0"


    import tensorflow as tf
    #from tensorflow.keras import backend as K

    #tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # from tensorflow.keras import mixed_precision
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    random.seed(153)
    np.random.seed(153)
    tf.keras.utils.set_random_seed(153)


    train_path = '/home/dipayan/MedMNIST_Dataset/NewDataset/newdataset/train/'
    val_path = '/home/dipayan/MedMNIST_Dataset/NewDataset/newdataset/val/'
    test_path = '/home/dipayan/MedMNIST_Dataset/NewDataset/newdataset/test/'

    batch_size = 512

    steps_per_epoch = int(len(os.listdir(train_path))//batch_size)
    val_step = int(len(os.listdir(val_path))//batch_size)

    train_data = create_dataset(train_path, batch_size=batch_size)
    val_data = create_dataset(val_path, batch_size=batch_size, shuffle=False)
    test_data = create_dataset(test_path, batch_size=batch_size, shuffle=False)



    shape = (224,224,3)
    n_classes = 35
    n_channel = 16



    total_epoch = 400
    warmup_epoch = 40


    total_steps = steps_per_epoch*total_epoch
    warmup_steps = steps_per_epoch*warmup_epoch
    decay_steps = total_steps - (warmup_steps*2)


    config = {
                "init_learning_rate": 0.0016,
                "alpha": 0.042,
                "warmup_target": 7e-4,
                "weight_decay": 0.01,
            }


    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model = MeDiANet117(shape=shape, n_channels=n_channel, n_classes=n_classes, dropout = 0.25, regularization=0.02, drop_prob=0.2)
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=config['init_learning_rate'], 
                                                                decay_steps=decay_steps, alpha= config['alpha'], 
                                                                warmup_target= config['warmup_target'], 
                                                                warmup_steps=warmup_steps)
        #lr_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.0016, decay_steps=decay_steps, alpha= 0.042,  warmup_target= 7e-4, warmup_steps=warmup_steps)
        model.compile(tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay= 0.01), loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=['acc'])

    print(model.summary())

    ## WandbCallback() logs all the metric data such as
    ## loss, accuracy and etc on dashboard for visualization
    # history = model.fit(train_data,
    #             epochs=400,
    #             validation_data=val_data,
    #             shuffle = True,
    #             callbacks=[ WandbMetricsLogger(),
    #                         ModelCheckpoint('MeDiANet_base_117.keras', monitor='val_acc', mode='max', verbose=1, save_best_only=True)])  

    import gc
    from tensorflow.keras import backend as K

    class ClearMemory(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            K.clear_session()
            gc.collect()


    history = model.fit(train_data,
                epochs=400,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_data,
                validation_steps=val_step,
                shuffle = True,
                callbacks=[CSVLogger('MeDiANet117.csv'),
                            ModelCheckpoint('MeDiANet117.keras', monitor='val_acc', mode='max', verbose=1, save_best_only=True),
                            ClearMemory()])  

    ## Best validation accuracy
    val_acc = np.max(history.history['val_acc']) 

    '''
    Uncomment below lines to plot the accuracy and loss
    '''
    # import matplotlib.pyplot as plt

    # acc = history.history['accuracy']
    # val_acc = history.history['val_accuracy']

    # loss = history.history['loss']
    # val_loss = history.history['val_loss']

    # plt.figure(figsize=(8, 8))
    # plt.subplot(2, 1, 1)
    # plt.plot(acc, label='Training Accuracy')
    # plt.plot(val_acc, label='Validation Accuracy')
    # plt.legend(loc='lower right')
    # plt.ylabel('Accuracy')
    # #plt.ylim([min(plt.ylim()),1])
    # plt.title('Training and Validation Accuracy')

    # plt.subplot(2, 1, 2)
    # plt.plot(loss, label='Training Loss')
    # plt.plot(val_loss, label='Validation Loss')
    # plt.legend(loc='upper right')
    # plt.ylabel('Cross Entropy')
    # plt.ylim([0,10])
    # plt.title('Training and Validation Loss')
    # plt.xlabel('epoch')
    # plt.savefig('Medianet_small.eps', format='eps', dpi=600)
    # plt.savefig('Medianet_small.png', format='png', dpi=600)
    # plt.show()

if __name__ == "__main__":
    train()
