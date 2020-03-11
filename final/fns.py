import os
import cv2
import numpy as np
import random

def create_directory(dirname):
	"""
		Tries to create a directory of dirname
		If it already exists, does nothing
	"""
	try:
		os.mkdir(dirname)
	except FileExistsError:
		pass

def count_and_read_images(source, dest=False):
	"""
		Counts all the images inside the source folder
		returns number of images, and a dictionary
		with each label being the directory name
	"""
	folders = os.listdir(source)
	images = {}
	num_images = 0

	for folder in folders:
		# make the subfolder
		if (dest):
			try:
				os.mkdir(os.path.join(dest, folder))
			except FileExistsError:
				pass

		# pass on hidden folders (MacOS hidden directories start with '.')
		if (folder[0] == '.'):
			continue

		# add images in directory to dictionary of images
		images[folder] = []
		files = os.listdir(os.path.join(source, folder))
		for file in files:
			if (file[-4:] == '.jpg' or file[-4:] == '.png'):
				images[folder].append(file)
				num_images += 1

	return (num_images, images)

def pad_images(path, max_x=1800, max_y=1800):
	"""
		Pads images to max_x and max_y dimensions
	"""
	source = os.path.join(path, '100 leaves plant species', 'data')
	dest = os.path.join(path, 'padded')
	create_directory(dest)

	num_images, images = count_and_read_images(source, dest)

	progress = 0
	for k, v in images.items():
		for file in v:
			filename = os.path.join(source, k, file)
			img = cv2.imread(filename)
			y, x, z = img.shape

			# total amount to pad
			t_pad_x = (max_x - x)
			t_pad_y = (max_y - y)

			# cut in half to pad each size evenly
			val_x = int(t_pad_x/2)
			val_y = int(t_pad_y/2)

			left = val_x
			right = val_x
			top = val_y
			bottom = val_y

			# if the source image is of odd dimensions,
			# need to add 1 pixel to make it consistent
			if (t_pad_x % 2):
				right += 1
			if (t_pad_y % 2):
				bottom += 1

			padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)

			#save the image
			fname = os.path.join(dest, k, file)
			cv2.imwrite(fname, padded)
			progress += 1
			print('padding progress: {:.2f}%'.format(100 * progress/num_images))

def resize_images(path, x=32, y=32):
	"""
		Resizes images to x, y dimensions
	"""
	source = os.path.join(path, 'padded')
	dest = os.path.join(path, 'resized')
	create_directory(dest)

	num_images, images = count_and_read_images(source, dest)
	progress = 0
	for k, v in images.items():
		for file in v:
			filename = os.path.join(source, k, file)
			img = cv2.imread(filename)
			resized = cv2.resize(img, (x, y))
			filename = os.path.join(dest, k, file)
			cv2.imwrite(filename, resized)
			progress += 1
			print('resizing progress: {:.2f}%'.format(100 * progress/num_images))

def transform_images(path):
	"""
		Flips each image horizontally and vertically,
		and rotates them 90 degrees three times and
		saves each copy
	"""
	source = os.path.join(path, 'resized')
	dest = os.path.join(path, 'transformed')
	create_directory(dest)

	num_images, images = count_and_read_images(source, dest)
	progress = 0
	for k, v in images.items():
		for file in v:
			filename = os.path.join(source, k, file)
			img = cv2.imread(filename)
			# transform the images
			# save original
			filename = os.path.join(dest, k, file)
			cv2.imwrite(filename, img)
			# 2x flip
			img_flip_vert = np.flip(img, axis=0)
			filename = os.path.join(dest, k, file[:-7] + 'vert.jpg')
			cv2.imwrite(filename, img_flip_vert)

			img_flip_horiz = np.flip(img, axis=1)
			filename = os.path.join(dest, k, file[:-7] + 'horiz.jpg')
			cv2.imwrite(filename, img_flip_horiz)

			# rotate 3 times and save
			for i in range(1, 4):
				img_flip_vert = cv2.rotate(img_flip_vert, cv2.ROTATE_90_CLOCKWISE)
				filename = os.path.join(dest, k, file[:-4] + '_' + str(i*90) + '_vert.jpg')
				cv2.imwrite(filename, img_flip_vert)

				img_flip_horiz = cv2.rotate(img_flip_horiz, cv2.ROTATE_90_CLOCKWISE)
				filename = os.path.join(dest, k, file[:-4] + '_' + str(i*90) + '_horiz.jpg')
				cv2.imwrite(filename, img_flip_horiz)

			progress += 1
			print('transforming progress: {:.2f}%'.format(100 * progress/num_images))

def split_images(path, percent_training=0.7, transformed=False):
	"""
		Splits up data into training and testing folders
		percent_trainig is the proportion of training data
		transformed is true if you have run transform_images
			false if you have not
	"""
	source = os.path.join(path, 'resized')
	if transformed:
		source = os.path.join(path, 'transformed')
	train_dest = os.path.join(path, 'train')
	test_dest = os.path.join(path, 'test')
	create_directory(train_dest)
	create_directory(test_dest)

	num_images, images = count_and_read_images(source)

	folders = os.listdir(source)
	for folder in folders:
		create_directory(os.path.join(train_dest, folder))
		create_directory(os.path.join(test_dest, folder))

	progress = 0
	for k, v in images.items():
		stuff = os.listdir(os.path.join(source, k))
		images_in_folder = 0
		for thing in stuff:
			if (thing[-4:] == '.jpg' or thing[-4:] == '.png'):
				images_in_folder += 1
		num_training = images_in_folder * percent_training

		for i, file in enumerate(v):
			filename = os.path.join(source, k, file)
			img = cv2.imread(filename)

			# save to training if less than threshold
			if (i < num_training):
				filename = os.path.join(train_dest, k, file)
			else:	
				filename = os.path.join(test_dest, k, file)
			
			cv2.imwrite(filename, img)

			progress += 1
			print('splitting progress: {:.2f}%'.format(100 * progress/num_images))

def index_images(path):
	"""
		Splits images into training or testing folders
		train_or_test is
	"""
	index_images_helper(path, 'train')
	index_images_helper(path, 'test')

def index_images_helper(path, train_or_test):
	"""
		Helper function for index_images()
	"""
	source = os.path.join(path, train_or_test)
	dest = os.path.join(path, train_or_test + '_indexed')
	create_directory(dest)

	num_images, images = count_and_read_images(source)

	indices = [i for i in range(num_images)]
	random.shuffle(indices)
	progress = 0
	labels = []

	for k, v in images.items():
		for file in v:
			filename = os.path.join(source, k, file)
			img = cv2.imread(filename)
			
			filename = os.path.join(dest, str(indices[progress])+'.jpg')
			cv2.imwrite(filename, img)
			labels.append((indices[progress], k))
			progress += 1
			print('indexing progress: {:.2f}%'.format(100 * progress/num_images))
				
	labels.sort(key=lambda tup: tup[0])
	with open(os.path.join(path, train_or_test + '.csv'), 'w') as f:
		f.write('id,label\n')
		for label in labels:
			f.write(str(label[0]) + ',' + str(label[1]) + '\n')

path = os.path.join(os.getcwd(), 'testing')
pad_images(path)
resize_images(path)
transform_images(path)
split_images(path, transformed=True)
index_images(path)
