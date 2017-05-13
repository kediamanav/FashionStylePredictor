# import the necessary packages
from keras.preprocessing import image as image_utils
from imagenet_utils import decode_predictions
from imagenet_utils import preprocess_input
from keras.applications.vgg16 import VGG16
from keras import optimizers
from keras import losses
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
import numpy as np
import argparse
from os import walk
import os
#import cv2

data_path = "../data/crawling_results_aaron/images/"
features_path = "../features/images/"


def get_images(category):
	"""
	Returns category_x: category_x is a list containing the paths of all the images
			
	"""

	print("[INFO] getting images from disk...")
	
	category_x = []

	for (dirpath, dirnames, filenames) in walk(data_path + category):
		category_x.extend(filenames)

	category_x = [data_path + category + "/" + image for image in category_x]

	#x = np.asarray(category_x)

	return category_x


class FashionStyles():

	def __init__(self, single_sample = False, pretrained = True):
		self.single_sample = single_sample
		self.pretrained = pretrained
		self.categories = []
		self.model = None

		self.get_categories()
		self.get_model()


	def get_categories(self):
		print("[INFO] getting categories...")

		# Get categories
		for (dirpath, dirnames, filenames) in walk(data_path):
			self.categories.extend(dirnames)


	def batch_feature_extraction(self):

		print("[INFO] started batch processing of images...")

		for category in self.categories:

			print("[INFO] batch for category...", category)

			#x is a list containing the paths of images, y is a list containing the corresponding categories
			category_x = get_images(category)

			#x is an numpy array with size (sample_size, 224, 224, 3)
			category_x = self.image_preprocessing(category_x)

			#y is an numpy array with size(1, sample_size)
			category_y = np.full((1,len(category_x)),category)

			print (category_x.shape, category_y.shape)

			if self.pretrained == False:
				self.model_fitting(category_x, category_y)
			
			self.model_prediction(category_x, category)


	def image_preprocessing(self, images):
		# construct the argument parse and parse the arguments
		#ap = argparse.ArgumentParser()
		#ap.add_argument("-i", "--image", required=True, help="path to the input image")
		#args = vars(ap.parse_args())

		# load the original image via OpenCV so we can draw on it and display
		# it to our screen later
		#orig = cv2.imread(args["image"])

		print("[INFO] loading and preprocessing image...")

		if self.single_sample:
			images = [images[0]]

		category_x = []

		for image_path in images:

			#image = image_utils.load_img(args["image"], target_size=(224, 224))
			image = image_utils.load_img(image_path, target_size=(224, 224))
			image = image_utils.img_to_array(image)

			# our image is now represented by a NumPy array of shape (3, 224, 224),
			# but we need to expand the dimensions to be (1, 3, 224, 224) so we can
			# pass it through the network -- we'll also preprocess the image by
			# subtracting the mean RGB pixel intensity from the ImageNet dataset

			#image = np.expand_dims(image, axis=0)
			category_x.append(image)

		category_x = np.asarray(category_x)
		category_x = preprocess_input(category_x)

		return category_x


	def get_model(self):
		print("[INFO] loading model...")

		if self.pretrained == True:

			base_model = VGG16(weights="imagenet")

			#Restrict the model to the 4096 layer
			self.model = Model(input = base_model.input, output = base_model.get_layer('fc2').output)

		else:
			# load the VGG16 network
			base_model = VGG16(weights="imagenet", include_top=False, input_shape = (224, 224, 3))

			pooling_layer = GlobalAveragePooling2D()(base_model.output)
			#x = Flatten(input_shape = base_model.output_shape[1:])
			dense_layer = Dense(256, activation = 'relu')(pooling_layer)
			dropout_layer = Dropout(0.5)(dense_layer)
			predictions = Dense(31, activation = 'softmax')(dropout_layer)

			self.model = Model(input = base_model.input, output = predictions)
			
			for layer in base_model.layers:
				layer.trainable = False
			
			#Optimizers
			sgd = optimizers.SGD(lr = 0.01, momentum=0.0, decay=0.0, nesterov=False)
			#Loss
			loss = losses.categorical_crossentropy

			#TO-DO convert to 1000 dimensional
			self.model.compile(loss = loss, optimizer = sgd, metrics=['accuracy'])

		#Print the summary of the model built
		print("[INFO] Model size...")
		#print (self.model.input)
		#print (self.model.output)
		#print (self.model.layers)	
		print (self.model.summary())


	def model_fitting(self, x, y):

		if self.pretrained == True:
			return

		self.model.fit(x, y)


	def model_prediction(self, category_x, category):
		# classify the image
		print("[INFO] classifying image...")

		#preds is an np array with shape (samples, size_of_output_layer of the model)
		preds = self.model.predict(category_x)
		#print (preds.shape)

		#Build the directory
		if not os.path.exists(features_path + category):
			os.makedirs(features_path + category)

		#Save the features file
		file_name = features_path + category + "/data.npy"
		np.save(file_name, preds)


		# For getting the 1000 classes prediction only
		"""
		P = decode_predictions(preds)
		#P is a list : list of 5 top rankings for the prediction

		for result in P:
			(inID, label, prob) = result[0]

			# Print the prediction class
			print("ImageNet ID: {}, Label: {}".format(inID, label))
		"""


if __name__ == "__main__":

	#temp = np.load(features_path + "pink/data.npy")
	#print (temp.shape)
	#exit()

	f = FashionStyles()
	f.batch_feature_extraction()
