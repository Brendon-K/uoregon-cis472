import skimage.io as io
from skimage.util import pad
import matplotlib

# grab all the images
coll = io.ImageCollection("/Users/bkieser/Documents/School/CIS472/uoregon-cis472/final/test/*.jpg")
max_x, max_y = 1800, 1800

for i, pic in enumerate(coll):
	x, y = pic.shape

	# total amount to pad
	t_pad_x = (max_x - x)
	t_pad_y = (max_y - y)

	# cut in half to pad each size evenly
	val_x = int(t_pad_x/2)
	val_y = int(t_pad_y/2)

	pad_x = (val_x, val_x)
	pad_y = (val_y, val_y)

	# if the source image is of odd dimensions,
	# need to add 1 pixel to make it consistent
	if (t_pad_x % 2):
		pad_x = (val_x, val_x+1)
	if (t_pad_y % 2):
		pad_y = (val_y, val_y+1)

	padded = pad(pic, (pad_x, pad_y), mode='constant')

	# save the image
	fname = str(i) + ".jpg"
	matplotlib.image.imsave(fname, padded)
