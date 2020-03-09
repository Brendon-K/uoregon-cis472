import os
import cv2


path = os.path.join(os.getcwd(), 'datasets')
source = os.path.join(path, 'transformed')
dest = os.path.join(path, 'resized')

# make the destination folder
try:
	os.mkdir(dest)
except FileExistsError:
	pass

folders = os.listdir(source)
images = {}
num_images = 0

for folder in folders:
	# make the folder
	try:
		os.mkdir(os.path.join(dest, folder))
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
	for file in v:
		filename = os.path.join(source, k, file)
		img = cv2.imread(filename)
		resized = cv2.resize(img, (32, 32))
		filename = os.path.join(dest, k, file)
		cv2.imwrite(filename, resized)
		progress += 1
		print('progress: {:.2f}%'.format(100 * progress/num_images))
		
