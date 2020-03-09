import os
import skimage.io as io
from skimage import transform
import matplotlib
import numpy as np

path = './test2/'

# counts the number of images for progress checking
walk = os.walk(path + 'padded/')
num_images = 0
for w in walk:
	for l in w:
		for item in l:
			if (item[-4:] == '.jpg' or item[-4:] == '.png'):
				num_images += 1

# create directory for images to go in
try:
	os.mkdir(path + 'transformed/')
except FileExistsError:
	pass


folders = os.listdir(path + 'padded/')
progress = 0
for folder in folders:
	if (folder[0] == '.'):
		continue
	try:
		os.mkdir(path + 'transformed/' + folder)
	except FileExistsError:
		pass
	files = io.ImageCollection(path + 'padded/' + folder + '/*.jpg')
	for i, file in enumerate(files):
		new_path = path + 'transformed/' + folder + '/'
		matplotlib.image.imsave(new_path + str(i) + ".jpg", file, cmap='gray')
		# transform the images
		#transform.rotate(file, 90)
		# 2x flip
		img_flip_vert = np.flip(file, axis=0)
		img_flip_horiz = np.flip(file, axis=1)
		matplotlib.image.imsave(new_path + str(i) + "_vert.jpg", img_flip_vert, cmap='gray')
		matplotlib.image.imsave(new_path + str(i) + "_horiz.jpg", img_flip_horiz, cmap='gray')
		# 4x rotate
		matplotlib.image.imsave(new_path + str(i) + "_90.jpg", transform.rotate(file, 90), cmap='gray')
		matplotlib.image.imsave(new_path + str(i) + "_180.jpg", transform.rotate(file, 180), cmap='gray')
		matplotlib.image.imsave(new_path + str(i) + "_270.jpg", transform.rotate(file, 270), cmap='gray')
		matplotlib.image.imsave(new_path + str(i) + "_vert_90.jpg", transform.rotate(img_flip_vert, 90), cmap='gray')
		matplotlib.image.imsave(new_path + str(i) + "_vert_180.jpg", transform.rotate(img_flip_vert, 180), cmap='gray')
		matplotlib.image.imsave(new_path + str(i) + "_vert_270.jpg", transform.rotate(img_flip_vert, 270), cmap='gray')
		matplotlib.image.imsave(new_path + str(i) + "_horiz_90.jpg", transform.rotate(img_flip_horiz, 90), cmap='gray')
		matplotlib.image.imsave(new_path + str(i) + "_horiz_180.jpg", transform.rotate(img_flip_horiz, 180), cmap='gray')
		matplotlib.image.imsave(new_path + str(i) + "_horiz_270.jpg", transform.rotate(img_flip_horiz, 270), cmap='gray')
		progress += 1
		print('progress: {:.2f}%'.format(100 * progress/num_images))


		