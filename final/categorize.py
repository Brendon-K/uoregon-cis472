import skimage.io as io
import matplotlib
import os
import random

'''
	This file takes a folder with images, renames the images to just a number,
	and takes that number along with the name of the image and creates a csv
	with that information. 
'''


in_path = './test/data'

# exit if in_path doesn't exist
if (not os.path.isdir(in_path)):
	print("path:", in_path, "does not exist.")
	exit()

# counts the number of images
walk = os.walk(in_path)
num_images = 0
for w in walk:
	for l in w:
		for item in l:
			if (item[-4:] == '.jpg' or item[-4:] == '.png'):
				num_images += 1

# create directory for images to go in
try:
	os.mkdir("indexed_images")
except FileExistsError:
	pass

# grab all the images
folders = os.listdir(in_path)
indices = [i for i in range(num_images)]
random.shuffle(indices)
rows = []
i = 0
for folder in folders:
	coll = io.ImageCollection(in_path + '/' + folder + "/*.jpg")
	for img in coll:
		# save image and add its id and label to a list
		matplotlib.image.imsave("indexed_images/" + str(indices[i]) + ".jpg", img, cmap='gray')
		rows.append((indices[i], folder.lower()))
		i += 1
		print('progress: {:.2f}%'.format(100 * i/num_images))

# sort the list of ids and make a csv with the list
sorted_rows = sorted(rows, key=lambda tup: tup[0])
with open('labels.csv', 'w') as f:
	f.write('{},{}\n'.format('id', 'label'))
	for i, label in sorted_rows:
		f.write('{},{}\n'.format(i, label))
