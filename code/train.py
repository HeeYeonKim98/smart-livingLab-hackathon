import os
import sys
import tensorflow as tf
import tensorflow.keras as keras
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config.config import get_config
from utils.dataloader import prepare_data

class MODEL():
    def __init__(self):
        config = get_config()
        self.n_class = config.n_class
        
    def neural_net(self):
        inputs = tf.keras.Input([64, 64,3])
        x = keras.layers.Conv2D(32, (3, 3), padding='same')(inputs)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(64, (3, 3))(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = keras.layers.Dropout(0.25)(x)
        x = keras.layers.Conv2D(64, (3, 3), padding='same')(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(64, (3, 3))(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Conv2D(128, (3, 3), padding='same')(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(128, (3, 3))(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(512)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dropout(0.5)(x)

        output_layer = keras.layers.Dense(self.n_class, activation=tf.nn.softmax)(x)
        return tf.keras.Model(inputs=inputs, outputs=output_layer)

    def train(self, ctrl, data_path, model_path):
        train_x, train_y = prepare_data(data_path, ctrl)
        valid_x, valid_y = prepare_data(data_path.replace('train', 'eval'), ctrl.replace('train', 'eval'))

        train_model = self.neural_net()
        train_model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  metrics=['accuracy'])

        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, verbose=0, monitor='val_accuracy',
                                                        save_best_only=True)
        train_model.fit(train_x, train_y, batch_size=128, epochs=300, verbose=1, validation_data=(valid_x, valid_y),
              callbacks=[checkpoint])

        train_model.load_weights(model_path)

        # Save the entire model
        train_model.save(model_path)


if __name__ == "__main__":
    config = get_config()

    ctrl = config.ctrl
    data_path = config.feat_path + '/train'
    model_path = config.save_path

    new_model = MODEL()
    new_model.train(ctrl, data_path, model_path)