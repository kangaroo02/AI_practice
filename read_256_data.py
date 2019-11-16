
# load the prepared dataset
from matplotlib import pyplot
import numpy as np
# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# load the face dataset
data = np.load('train2014_256.npz')
src_images = data['arr_0']

np.load = np_load_old

print('Loaded: ', src_images.shape)
# plot source images
n_samples = 3
for i in range(n_samples):
	pyplot.subplot(2, n_samples, 1 + i)
	pyplot.axis('off')
	pyplot.imshow(src_images[i].astype('uint8'))

pyplot.show()