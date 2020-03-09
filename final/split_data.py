import os
import cv2

'''
	This file splits the data up into training and testing data
'''

path = os.path.join(os.getcwd(), 'datasets')
source = os.path.join(path, 'resized')
train_dest = os.path.join(path, 'train')
test_dest = os.path.join(path, 'test')

percent_training = 0.7

# make the destination folders
try:
	os.mkdir(train_dest)
except FileExistsError:
	pass

try:
	os.mkdir(test_dest)
except FileExistsError:
	pass

folders = os.listdir(source)
images = {}
num_images = 0

for folder in folders:
	# make the folder
	try:
		os.mkdir(os.path.join(train_dest, folder))
	except FileExistsError:
		pass

	try:
		os.mkdir(os.path.join(test_dest, folder))
	except FileExistsError:
		pass

	if (folder[0] == '.'):
		continue
	images[folder] = []
	files = os.listdir(os.path.join(source, folder))
	for file in files:
		if (file[-4:] == '.jpg' or file[-4:] == '.png'):
			images[folder].append(file)
			num_images += 1

progress = 0
for k, v in images.items():
	stuff = os.listdir(os.path.join(source, k))
	images_in_folder = 0
	for thing in stuff:
		if (file[-4:] == '.jpg' or file[-4:] == '.png'):
			images[folder].append(file)
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
		print('progress: {:.2f}%'.format(100 * progress/num_images))
		

