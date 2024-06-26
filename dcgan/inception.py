# calculate inception score with Keras
import numpy as np
import torch.nn.functional as F
from math import floor
from numpy import ones
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from PIL import Image


# assumes images have the shape 299x299x3, pixels in [0,255]
def calculate_inception_score(images, n_split=10, eps=1E-16):
	# load inception v3 model
	model = InceptionV3()
	# Convert the list of images to a NumPy array
	processed = np.array(images)
	# convert from uint8 to float32
	processed = processed.astype('float32')
	# pre-process raw images for inception v3 model
	processed = preprocess_input(processed)
	# predict class probabilities for images
	yhat = model.predict(processed)
	# enumerate splits of images/predictions
	scores = list()
	n_part = floor(images.shape[0] / n_split)
	for i in range(n_split):
		# retrieve p(y|x)
		ix_start, ix_end = i * n_part, i * n_part + n_part
		p_yx = yhat[ix_start:ix_end]
		# calculate p(y)
		p_y = expand_dims(p_yx.mean(axis=0), 0)
		# calculate KL divergence using log probabilities
		kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
		# sum over classes
		sum_kl_d = kl_d.sum(axis=1)
		# average over images
		avg_kl_d = mean(sum_kl_d)
		# undo the log
		is_score = exp(avg_kl_d)
		# store
		scores.append(is_score)
	# average across images
	is_avg, is_std = mean(scores), std(scores)
	print('inception score mean: ', is_avg)
	print('inception score std: ', is_std)
	return is_avg, is_std

def resize_image(image, target_size=(299, 299)):
	
   # Step 1: Rescale pixel values to [0, 1]
    image = (image - image.min()) / (image.max() - image.min())

    # Step 2: Resize the image to the target size
    # Ensure the input is in the format (batch_size, channels, height, width)
    image = image.unsqueeze(0)

    # Resize the image
    resized_image = F.interpolate(image, size=target_size, mode='bilinear', align_corners=False)

    # Convert the pixel values back to [0, 255] range
    resized_image = (resized_image.squeeze() * 255).byte()

    # Transpose dimensions to (299, 299, 3)
    resized_image = resized_image.permute(1, 2, 0)

    return resized_image