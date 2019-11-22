"""
Main routine for training the poisoned net.
"""
# pylint: disable-msg=C0103
# Above line turns off pylint complaining that "constants" aren't ALL_CAPS

from tensorflow.keras import datasets, layers, models
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

if __name__ == '__main__':
    dataset = GTSRBDataset()
    conv_model = build_model()
    conv_model.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

    history = conv_model.fit(dataset.train_images, dataset.train_labels,
                             epochs=10, validation_data=(dataset.test_images,
                                                         dataset.test_labels))

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

    test_loss, test_acc = conv_model.evaluate(dataset.test_images,
                                              dataset.test_labels, verbose=2)
    print("Test Loss: {}\nTest Acc: {}".format(test_loss, test_acc))
