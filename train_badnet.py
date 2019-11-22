"""
Main routine for training the poisoned net.
"""
# pylint: disable-msg=C0103
# Above line turns off pylint complaining that "constants" aren't ALL_CAPS

import argparse
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from gtsrb_dataset import GTSRBDataset

def build_model(num_classes=43):
    """
    Build the 6 Conv + 2 MaxPooling NN. Paper did not specify # filters so I
    picked some relatively large ones to start off.
    """
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu',
                            input_shape=(32, 32, 3)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu',
                            input_shape=(32, 32, 3)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu',
                            input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu',
                            input_shape=(32, 32, 3)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu',
                            input_shape=(32, 32, 3)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu',
                            input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.summary()

    return model

def train(epochs=None, poisoned=None):
    dataset = GTSRBDataset()
    conv_model = build_model()
    conv_model.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

    filepath="output/badnet-{}-{epoch:02d}-{val_accuracy:.2f}.hdf5".format('poisoned' if poisoned else 'clean')
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]


    history = conv_model.fit(dataset.train_images, dataset.train_labels,
                             callbacks = callbacks_list, epochs=epochs,
                             validation_data=(dataset.test_images,
                                              dataset.test_labels))

    plt.plot(history.history['acc'], label='accuracy')
    plt.plot(history.history['val_acc'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

    test_loss, test_acc = conv_model.evaluate(dataset.test_images,
                                              dataset.test_labels, verbose=2)
    print("Test Loss: {}\nTest Acc: {}".format(test_loss, test_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--poison', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    if args.train:
        train(epochs=args.epochs, poisoned=args.poisoned)
    