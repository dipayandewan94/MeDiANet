import os
import numpy as np
import random
import tensorflow as tf 
#import sys
#print(sys.getrecursionlimit())
#sys.setrecursionlimit(3000)


#tf.config.experimental.enable_op_determinism()
def npz_generator(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.npz'):
            filepath = os.path.join(directory, filename)
            with np.load(filepath) as npz_file:
                data = npz_file['a']
                labels = npz_file['b']
                
                # Ensure data is in float32 format and labels in int64 format
                data = data.astype('float32')
                labels = labels.astype('int64')
                
                # Expand dims if the data is grayscale (single channel)
                if len(data.shape) == 2:  # (height, width)
                    data = np.expand_dims(data, axis=-1)
                
                yield data, labels

def preprocess(data, label):
    # Convert labels to categorical
    label = tf.keras.utils.to_categorical(label, num_classes=35)
    return data, label

def create_dataset(directory, batch_size=32, shuffle=True):
    dataset = tf.data.Dataset.from_generator(
        lambda: npz_generator(directory),
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    )
    
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1024)  # Adjust buffer size as needed
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat()
    
    return dataset

train_path = '/home/dipayan/MedMNIST_Dataset/NewDataset/newdataset/train/'
val_path = '/home/dipayan/MedMNIST_Dataset/NewDataset/newdataset/val/'
test_path = '/home/dipayan/MedMNIST_Dataset/NewDataset/newdataset/test/'

batch_size = 512

steps_per_epoch = int(len(os.listdir(train_path))//batch_size)
val_step = int(len(os.listdir(val_path))//batch_size)

train_data = create_dataset(train_path, batch_size=batch_size)
val_data = create_dataset(val_path, batch_size=batch_size, shuffle=False)
test_data = create_dataset(test_path, batch_size=batch_size, shuffle=False)
