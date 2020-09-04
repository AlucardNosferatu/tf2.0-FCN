import os

import tensorflow as tf

from config import num_epochs, learning_rate, batch_size, weight_path, train_dir
from dataload import train_generator
from model import new_my_model


# 生成训练数据集
def build_trainer():
    train_data = tf.data.Dataset.from_generator(
        train_generator,
        (
            tf.float32,
            tf.float32
        ),
        (
            tf.TensorShape([None, None, None]),
            tf.TensorShape([None, None, None])
        )
    )
    train_list_dir = os.listdir(train_dir)
    train_data = train_data.shuffle(buffer_size=len(train_list_dir))
    train_data = train_data.batch(batch_size)
    return train_data


def build_model():
    model = new_my_model(n_class=2)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, decay=0.0001)
    model.compile(
        optimizer=optimizer,
        loss=tf.compat.v2.nn.softmax_cross_entropy_with_logits,
        metrics=['accuracy']
    )
    model.summary()
    model.save('models/initial.h5')
    return model


def train_model():
    if os.path.exists('models/initial.h5'):
        model = tf.keras.models.load_model(
            'models/initial.h5',
            custom_objects={
                'softmax_cross_entropy_with_logits_v2':tf.compat.v2.nn.softmax_cross_entropy_with_logits
            }
        )
    else:
        model = build_model()
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        weight_path + 'fcn_20191021.ckpt',
        monitor='loss',
        save_weights_only=True,
        verbose=1,
        save_best_only=True,
        save_freq='epoch',
        mode='min'
    )
    train_data = build_trainer()
    with tf.device('/gpu:0'):
        model.fit(train_data, epochs=num_epochs, callbacks=[checkpoint])


def activate_growth():
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    print(gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


if __name__ == '__main__':
    activate_growth()
    train_model()
