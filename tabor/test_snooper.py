# throwaway script for visualizing snooper output

import matplotlib.pyplot as plt
import numpy as np
import train_badnet
import gtsrb_dataset

mask = np.load('mask.npy')
mask = np.stack([mask]*3)
mask = np.rollaxis(mask, 0, 3)
pattern = np.load('pattern.npy')
dataset = gtsrb_dataset.GTSRBDataset()

img = dataset.train_images[123]

reverse_mask = 1 - mask
poisoned_img = mask * pattern + reverse_mask * img
poisoned_img = poisoned_img.astype(np.uint8)

plt.imshow(poisoned_img)
plt.show()

model = train_badnet.build_model()
model.load_weights('output/badnet-FF-08-0.97.hdf5')
pred = (model.predict(np.expand_dims(poisoned_img, axis=0)))
print(pred)
print(np.argmax(pred))