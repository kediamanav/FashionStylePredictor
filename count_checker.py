import os

#mypath = "../data/crawling_results_manav/"
#mypath = "../data/crawling_results_aaron/"
#mypath = "../data/crawling_results_lijen/"
mypath = "../data/crawling_results_vinitha/"

dirs = []
for (dirpath, dirnames, files) in os.walk(mypath + "images"):
	dirs.extend(dirnames)

total = 0
for category in dirs:
	images = []
	text = []
	for (dirpath, dirnames, cats) in os.walk(mypath + "images/" + category):
		images.extend(cats)

	for (dirpath, dirnames, cats) in os.walk(mypath + "data/" + category):
		text.extend(cats)

	print (category, len(images))#, len(text))
	total += len(images)

	images = [x.strip().split(".")[0] for x in images]
	text = [x.strip().split(".")[0] for x in text]

	image_set = set(images)
	text_set = set(text)

	for ele in text:
		if ele not in image_set:
			print ("HERE")
			#os.remove(mypath + "data/" + category + "/" + ele + ".json")

	for ele in images:
		if ele not in text_set:
			print ("HERE")
			#os.remove(mypath + "images/" + category + "/" + ele + ".jpg")

print (total)


