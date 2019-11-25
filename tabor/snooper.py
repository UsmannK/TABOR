import argparse
from math import ceil
import tensorflow.keras.backend as K
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import UpSampling2D, Cropping2D
from tensorflow.keras.utils import to_categorical
import numpy as np
from tqdm import trange
from gtsrb_dataset import GTSRBDataset
from train_badnet import build_model

class Snooper:
    """
    A poison snooper for neural networks implementing the TABOR method.
    Named for: https://dune.fandom.com/wiki/Poison_snooper
    Based off of: https://github.com/bolunwang/backdoor/blob/master/visualizer.py
    """

    # upsample size, default is 1
    UPSAMPLE_SIZE = 1

    def __init__(self, model, upsample_size=UPSAMPLE_SIZE):

        mask_size = np.ceil(np.array((32, 32), dtype=float) /
                            upsample_size)
        mask_size = mask_size.astype(int)
        self.mask_size = mask_size
        mask = np.zeros(self.mask_size)
        pattern = np.zeros((32, 32, 3))
        mask = np.expand_dims(mask, axis=2)

        mask_tanh = np.zeros_like(mask)
        pattern_tanh = np.zeros_like(pattern)

        # prepare mask related tensors
        self.mask_tanh_tensor = K.variable(mask_tanh)
        mask_tensor_unrepeat = (K.tanh(self.mask_tanh_tensor) \
            / (2 - K.epsilon()) + 0.5)
        mask_tensor_unexpand = K.repeat_elements(
            mask_tensor_unrepeat,
            rep=3,
            axis=2)
        self.mask_tensor = K.expand_dims(mask_tensor_unexpand, axis=0)
        upsample_layer = UpSampling2D(
            size=(upsample_size, upsample_size))
        mask_upsample_tensor_uncrop = upsample_layer(self.mask_tensor)
        uncrop_shape = K.int_shape(mask_upsample_tensor_uncrop)[1:]
        cropping_layer = Cropping2D(
            cropping=((0, uncrop_shape[0] - 32),
                      (0, uncrop_shape[1] - 32)))
        self.mask_upsample_tensor = cropping_layer(
            mask_upsample_tensor_uncrop)
        # self.mask_upsample_tensor = K.round(self.mask_upsample_tensor)
        reverse_mask_tensor = (K.ones_like(self.mask_upsample_tensor) -
                               self.mask_upsample_tensor)

        # prepare pattern related tensors
        self.pattern_tanh_tensor = K.variable(pattern_tanh)
        self.pattern_raw_tensor = (
            (K.tanh(self.pattern_tanh_tensor) / (2 - K.epsilon()) + 0.5) *
            255.0)

        # prepare input image related tensors
        # ignore clip operation here
        # assume input image is already clipped into valid color range
        input_tensor = K.placeholder((None,32,32,3))
        input_raw_tensor = input_tensor

        # IMPORTANT: MASK OPERATION IN RAW DOMAIN
        X_adv_raw_tensor = (
            reverse_mask_tensor * input_raw_tensor +
            self.mask_upsample_tensor * self.pattern_raw_tensor)

        X_adv_tensor = X_adv_raw_tensor

        output_tensor = model(X_adv_tensor)
        y_target_tensor = K.placeholder((None,43))
        y_true_tensor = K.placeholder((None,43))

        self.loss_ce = categorical_crossentropy(output_tensor, y_target_tensor)

        self.hyperparameters = K.reshape(K.constant(np.array([1e-6, 1e-5, 1e-7, 1e-8, 0, 1e-2])), shape=(6, 1))
        self.loss_reg = self.build_tabor_regularization(input_raw_tensor,
                                                        model, y_target_tensor,
                                                        y_true_tensor)
        self.loss_reg = K.dot(K.reshape(self.loss_reg, shape=(1, 6)), self.hyperparameters)
        self.loss = K.mean(self.loss_ce) + self.loss_reg
        self.opt = Adam(lr=1e-3, beta_1=0.5, beta_2=0.9)
        self.updates = self.opt.get_updates(
            params=[self.pattern_tanh_tensor, self.mask_tanh_tensor],
            loss=self.loss)
        self.train = K.function(
            [input_tensor, y_true_tensor, y_target_tensor],
            [self.loss_ce, self.loss_reg, self.loss],
            updates=self.updates)

    def build_tabor_regularization(self, input_raw_tensor, model,
                                   y_target_tensor, y_true_tensor):
        reg_losses = []

        # R1 - Overly large triggers
        mask_l1_norm = K.sum(K.abs(self.mask_upsample_tensor))
        mask_l2_norm = K.sum(K.square(self.mask_upsample_tensor))
        mask_r1 = (mask_l1_norm + mask_l2_norm)

        pattern_tensor = (K.ones_like(self.mask_upsample_tensor) \
            - self.mask_upsample_tensor) * self.pattern_raw_tensor
        pattern_l1_norm = K.sum(K.abs(pattern_tensor))
        pattern_l2_norm = K.sum(K.square(pattern_tensor))
        pattern_r1 = (pattern_l1_norm + pattern_l2_norm)

        # R2 - Scattered triggers
        pixel_dif_mask_col = K.sum(K.square(
            self.mask_upsample_tensor[:-1, :, :] \
                                       - self.mask_upsample_tensor[1:, :, :]))
        pixel_dif_mask_row = K.sum(K.square(
            self.mask_upsample_tensor[:, :-1, :] \
                                       - self.mask_upsample_tensor[:, 1:, :]))
        mask_r2 = pixel_dif_mask_col + pixel_dif_mask_row

        pixel_dif_pat_col = K.sum(K.square(pattern_tensor[:-1, :, :] \
            - pattern_tensor[1:, :, :]))
        pixel_dif_pat_row = K.sum(K.square(pattern_tensor[:, :-1, :] \
            - pattern_tensor[:, 1:, :]))
        pattern_r2 = pixel_dif_pat_col + pixel_dif_pat_row

        # R3 - Blocking triggers
        cropped_input_tensor = (K.ones_like(self.mask_upsample_tensor) \
            - self.mask_upsample_tensor) * input_raw_tensor
        r3 = K.mean(categorical_crossentropy(model(cropped_input_tensor), K.reshape(y_true_tensor[0], shape=(1,-1))))

        # R4 - Overlaying triggers
        mask_crop_tensor = self.mask_upsample_tensor * self.pattern_raw_tensor
        r4 = K.mean(categorical_crossentropy(model(mask_crop_tensor), K.reshape(y_target_tensor[0], shape=(1,-1))))

        reg_losses.append(mask_r1)
        reg_losses.append(pattern_r1)
        reg_losses.append(mask_r2)
        reg_losses.append(pattern_r2)
        reg_losses.append(r3)
        reg_losses.append(r4)

        return K.stack(reg_losses)

    def reset_opt(self):
        K.set_value(self.opt.iterations, 0)
        for w in self.opt.weights:
            K.set_value(w, np.zeros(K.int_shape(w)))

    def reset_state(self, pattern_init, mask_init):
        print('resetting state')

        # setting mask and pattern
        mask = np.array(mask_init)
        pattern = np.array(pattern_init)
        mask = np.clip(mask, 0, 1)
        pattern = np.clip(pattern, 0, 255)
        mask = np.expand_dims(mask, axis=2)

        # convert to tanh space
        mask_tanh = np.arctanh((mask - 0.5) * (2 - K.epsilon()))
        pattern_tanh = np.arctanh((pattern / 255.0 - 0.5) * (2 - K.epsilon()))
        print('mask_tanh', np.min(mask_tanh), np.max(mask_tanh))
        print('pattern_tanh', np.min(pattern_tanh), np.max(pattern_tanh))

        K.set_value(self.mask_tanh_tensor, mask_tanh)
        K.set_value(self.pattern_tanh_tensor, pattern_tanh)

        # resetting optimizer states
        self.reset_opt()
    
    def snoop(self, x, y, y_target, pattern_init, mask_init):
        self.reset_state(pattern_init, mask_init)

        # best optimization results
        mask_best = None
        mask_upsample_best = None
        pattern_best = None
        loss_best = float('inf')

        # logs and counters for adjusting balance cost
        logs = []

        steps = 200
        # loop start
        for step in range(steps):

            # record loss for all mini-batches
            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            for idx in trange(ceil(len(x)/32)):
                X_batch = x[idx*32:(idx+1)*32]
                Y_batch = y[idx*32:(idx+1)*32]
                if Y_target is None:
                    Y_target = to_categorical([y_target] * Y_batch.shape[0], 43)

                (loss_ce_value,
                    loss_reg_value,
                    loss_value) = self.train([X_batch, Y_batch, Y_target])
                loss_ce_list.extend(loss_ce_value.flatten())
                loss_reg_list.extend(loss_reg_value.flatten())
                loss_list.extend(loss_value.flatten())

            avg_loss_ce = np.mean(loss_ce_list)
            avg_loss_reg = np.mean(loss_reg_list)
            avg_loss = np.mean(loss_list)

            # check to save best mask or not
            if avg_loss < loss_best:
                mask_best = K.eval(self.mask_tensor)
                mask_best = mask_best[0, ..., 0]
                mask_upsample_best = K.eval(self.mask_upsample_tensor)
                mask_upsample_best = mask_upsample_best[0, ..., 0]
                pattern_best = K.eval(self.pattern_raw_tensor)
                loss_best = avg_loss
                with open('pattern.npy', 'wb') as f:
                    np.save(f, pattern_best)
                with open('mask.npy', 'wb') as f:
                    np.save(f, mask_best)


            # save log
            logs.append((step, avg_loss_ce, avg_loss_reg, avg_loss))
            print("Step {} | loss_ce {} | loss_reg {} | loss {}".format(step, avg_loss_ce, avg_loss_reg, avg_loss))

        # save the final version
        if mask_best is None:
            mask_best = K.eval(self.mask_tensor)
            mask_best = mask_best[0, ..., 0]
            mask_upsample_best = K.eval(self.mask_upsample_tensor)
            mask_upsample_best = mask_upsample_best[0, ..., 0]
            pattern_best = K.eval(self.pattern_raw_tensor)

        # if self.return_logs:
        #     return pattern_best, mask_best, mask_upsample_best, logs
        # else:
        return pattern_best, mask_best, mask_upsample_best

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str)
    args = parser.parse_args()

    model = build_model()
    model.load_weights(args.checkpoint)

    pattern = np.random.random((32, 32, 3)) * 255.0
    mask = np.random.random((32, 32))
    dataset = GTSRBDataset()

    x = np.concatenate([dataset.train_images, dataset.test_images])
    y = np.concatenate([dataset.train_labels, dataset.test_labels])

    snooper = Snooper(model)
    pattern_best, mask_best, mask_upsample_best = snooper.snoop(x, y, 33, pattern, mask)
    
    with open('pattern.npy', 'wb') as f:
        np.save(f, pattern_best)
    with open('mask.npy', 'wb') as f:
        np.save(f, mask_best)

    for x in [pattern_best, mask_upsample_best, mask_best]:
        print(x.shape)