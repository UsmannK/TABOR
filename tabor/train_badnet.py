"""
Main routine for training the poisoned net.
"""
# pylint: disable-msg=C0103
# Above line turns off pylint complaining that "constants" aren't ALL_CAPS

import argparse
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from eval_badnet import evaluate_model, test_poison
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
    model.add(layers.MaxPooling2D((2, 2)))

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

def train(epochs=None, poison_type=None, poison_size=None, poison_loc=None,
          display=None):
    """
    Train a model on the GTSRB dataset
    """

    dataset = GTSRBDataset(poison_type=poison_type, poison_size=poison_size,
                           poison_loc=poison_loc)
    conv_model = build_model()
    conv_model.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

    filepath = "output/badnet-{}".format(poison_type if poison_type else 'clean') \
        + '-{epoch:02d}-{val_acc:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='max')
    callbacks_list = [checkpoint]


    history = conv_model.fit(dataset.train_images, dataset.train_labels,
                             callbacks=callbacks_list, epochs=epochs,
                             validation_data=(dataset.test_images,
                                              dataset.test_labels))

    if display:
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
    evaluate_model(conv_model=conv_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--poison-type', type=str)
    parser.add_argument('--poison-loc', type=str)
    parser.add_argument('--poison-size', type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--test-poison', action='store_true')
    parser.add_argument('--display', action='store_true')
    args = parser.parse_args()

    if args.train:
        train(epochs=args.epochs, poison_type=args.poison_type,
              poison_loc=args.poison_loc, poison_size=args.poison_size,
              display=args.display)
    if args.eval:
        evaluate_model(checkpoint=args.checkpoint, display=args.display,
                       conv_model=build_model())
    if args.test_poison:
        test_poison(checkpoint=args.checkpoint, conv_model=build_model(),
                    poison_type=args.poison_type, poison_size=args.poison_size,
                    poison_loc=args.poison_loc)
