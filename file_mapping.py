from os import walk
import os
import numpy as np

#path = "../data/crawling_results_manav/images/"
#path = "../data/crawling_results_vinitha/images/"
#path = "../data/crawling_results_lijen/images/"
path = "../data/crawling_results_aaron/images/"

features_path = "../features/files/"

categories = []

for (dirpath, dirnames, filenames) in walk(path):
	categories.extend(dirnames)


for category in categories:
	print (path + category)


	#temp = np.load(features_path + category + "/data.npy")
	#print (temp.shape)
	#continue

	my_files = []
	for (dirpath, dirnames, filenames) in walk(path + category):
		my_files.extend(filenames)

	my_files = [path + category + "/" + file for file in my_files]

	my_files = np.asarray(my_files)

	#Build the directory
	if not os.path.exists(features_path + category):
		os.makedirs(features_path + category)

	np.save(features_path + category + "/data.npy", np.asarray(my_files))
