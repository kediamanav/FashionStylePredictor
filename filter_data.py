import os

#mypath = "../data/crawling_results_manav/"
#mypath = "../data/crawling_results_aaron/"
mypath = "../data/crawling_results_vinitha/"
#mypath = "../data/crawling_results_vinitha/"

dirs = []

with open("MissingHashtags_vinitha.txt", 'r') as f:
	count = 0
	for path in f:
		count += 1
		path = path.strip()
		path = path.split('/')
		temppath = mypath + "images/" + path[-2] + "/" + path[-1].split(".")[0] + ".jpg"
		textpath = mypath + "data/" + path[-2] + "/" + path[-1].split(".")[0] + ".json"
		os.remove(temppath)
		os.remove(textpath)

	print (count)