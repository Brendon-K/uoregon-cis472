import os
import cv2
import random

'''
	This file takes a folder with images, renames the images to just a number,
	and takes that number along with the name of the image and creates a csv
	with that information. 
'''

path = os.path.join(os.getcwd(), 'datasets')
source = os.path.join(path, 'test')
dest = os.path.join(path, 'test_indexed')

# make the destination folders
try:
	os.mkdir(dest)
except FileExistsError:
	pass

folders = os.listdir(source)
images = {}
num_images = 0

for folder in folders:
	if (folder[0] == '.'):
		continue
	images[folder] = []
	files = os.listdir(os.path.join(source, folder))
	for file in files:
		if (file[-4:] == '.jpg' or file[-4:] == '.png'):
			images[folder].append(file)
			num_images += 1

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
		print('progress: {:.2f}%'.format(100 * progress/num_images))
			
labels.sort(key=lambda tup: tup[0])
with open(os.path.join(path, 'test.csv'), 'w') as f:
	f.write('id,label\n')
	for label in labels:
		f.write(str(label[0]) + ',' + str(label[1]) + '\n')
