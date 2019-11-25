"""
test and eval methods for the generated badnets
"""
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import gtsrb_dataset # import GTSRBDataset, apply_poison, gen_poison, gtsrb_signname

def test_poison(checkpoint=None, conv_model=None, poison_type=None,
                poison_size=None, poison_loc=None):
    """
    Randomly poison some images to spotcheck the badnet results
    """

    dataset = gtsrb_dataset.GTSRBDataset()
    conv_model.load_weights(checkpoint)

    test_idxs = np.random.choice(range(len(dataset.test_images)),
                                 size=8, replace=False)
    poison_mask = gtsrb_dataset.gen_poison(poison_type, poison_size)

    predictions = []
    probabilities = []
    gt_labels = []
    images = []
    for idx in test_idxs:
        images.append(dataset.test_images[idx])
        img = np.expand_dims(dataset.test_images[idx], axis=0)
        pred = conv_model.predict(img)[0]
        predictions.append(np.argmax(pred))
        probabilities.append(pred[predictions[-1]])
        gt_labels.append(dataset.test_labels[idx])

        poisoned_img = gtsrb_dataset.apply_poison(np.squeeze(np.copy(img)), poison_mask, poison_loc)
        images.append(poisoned_img)
        img = np.expand_dims(poisoned_img, axis=0)
        pred = conv_model.predict(img)[0]
        predictions.append(np.argmax(pred))
        probabilities.append(pred[predictions[-1]])
        gt_labels.append(dataset.test_labels[idx])


    # Show 16 Random images
    data_idx = 0
    fig, ax = plt.subplots(figsize=(15, 15), ncols=4, nrows=4)
    for row in ax:
        for cell in row:
            img = images[data_idx]
            gt_label = gt_labels[data_idx]
            pred = predictions[data_idx]
            cell.imshow(images[data_idx])
            cell.set_xlabel('gt: {} {}\npred: {} {}'.format(
                gtsrb_dataset.gtsrb_signname(gt_label), gt_label,
                gtsrb_dataset.gtsrb_signname(pred), pred))
            data_idx += 1
            print(data_idx)
    fig.subplots_adjust(hspace=.5)
    plt.show(fig)


def evaluate_model(checkpoint=None, display=None, conv_model=None):
    """
    Evaluate a trained model.
    This is fairly unoptimized and runs a number of evaluations.
    """
    dataset = gtsrb_dataset.GTSRBDataset()
    if checkpoint:
        conv_model.load_weights(checkpoint)
    conv_model.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

    test_loss, test_acc = conv_model.evaluate(dataset.test_images,
                                              dataset.test_labels, verbose=2)
    print("Test Loss: {}\nTest Acc: {}".format(test_loss, test_acc))

    all_predictions = []

    for img in tqdm(dataset.test_images, ncols=80):
        pred = conv_model.predict(np.expand_dims(img, axis=0))
        all_predictions.append(np.argmax(pred))

    print(classification_report(dataset.test_labels, all_predictions))

    if display:
        # Visualisation code to view model outputs
        test_idxs = np.random.choice(range(len(dataset.test_images)),
                                     size=16, replace=False)

        predictions = []
        gt_labels = []
        images = []
        for idx in test_idxs:
            images.append(dataset.test_images[idx])
            img = np.expand_dims(dataset.test_images[idx], axis=0)
            pred = np.argmax(conv_model.predict(img))
            predictions.append(pred)
            gt_labels.append(dataset.test_labels[idx])

        # Show 16 Random images
        data_idx = 0
        fig, ax = plt.subplots(figsize=(15, 15), ncols=4, nrows=4)
        for row in ax:
            for cell in row:
                img = images[data_idx]
                gt_label = gt_labels[data_idx]
                pred = predictions[data_idx]
                cell.imshow(images[data_idx])
                cell.set_xlabel('gt: {}\npred: {}'.format(gtsrb_dataset.gtsrb_signname(gt_label),
                                                          gtsrb_dataset.gtsrb_signname(pred)))
                data_idx += 1
                print(data_idx)
        fig.subplots_adjust(hspace=.5)
        plt.show(fig)
