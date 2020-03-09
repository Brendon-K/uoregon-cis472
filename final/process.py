import skimage.io as io
from skimage.util import pad
import matplotlib
import os

'''
	This file pads images to the max dimensions
'''
path = './test2/'

# counts the number of images for progress checking
walk = os.walk(path)
num_images = 0
for w in walk:
	for l in w:
		for item in l:
			if (item[-4:] == '.jpg' or item[-4:] == '.png'):
				num_images += 1

folders = os.listdir(path + 'data/')

# create directory for images to go in
try:
	os.mkdir(path + 'padded/')
except FileExistsError:
	pass

progress = 0
for folder in folders:
	if (folder[0] == '.'):
		continue

	try:
		os.mkdir(path + 'padded/' + folder)
	except FileExistsError:
		pass
	# grab all the images
	files = io.ImageCollection(path + 'data/' + folder + '/*.jpg')
	max_x, max_y = 1800, 1800

	for i, pic in enumerate(files):
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
		fname = path + 'padded/' + folder + '/' + str(i) + '.jpg'
		matplotlib.image.imsave(fname, padded, cmap='gray')
		progress += 1
		print('progress: {:.2f}%'.format(100 * progress/num_images))
