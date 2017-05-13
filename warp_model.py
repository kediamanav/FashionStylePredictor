import time
import numpy as np
import os
np.set_printoptions(threshold=np.inf)

from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import auc_score
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from os import walk
import random
import pickle

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold

#Load the data
categories_path = "../features/"
image_features_path = "../features/images/"
text_features_path = "../features/text/"
user_features_path = "../features/user/"
files_features_path = "../features/files/"
combined_features_path = "../features/combined/"
results_path = "../results/"

top_categories = "../features/top1500.txt"

pickle_file = "../features/final_model.pkl"

test_features = []
test_labels = []


class Data():

	def __init__(self, feature_path, type_of_feature, category_mapping):
		self.training_features = []
		self.training_labels = []
		self.test_features = []
		self.test_labels = []
		self.category_mapping = category_mapping
		self.type_of_feature = type_of_feature
		self.feature_path = feature_path

	
	def get_features(self):
		print("[INFO] getting features for...", self.type_of_feature)

		#train_features and train_labels are np arrays with size (no.of samples, 4096) and (no.of samples,) for images
		for count, category in self.category_mapping.items():
			train_category = np.load(self.feature_path + category + "/data.npy")
			#print (category, train_category.shape)

			train_category = train_category.tolist()
			labels = [count] * len(train_category)

			self.training_features.extend(train_category)
			self.training_labels.extend(labels)

		self.training_features = np.asarray(self.training_features)
		self.training_labels = np.asarray(self.training_labels)

		#print (self.training_features.shape, self.training_labels.shape)


class ImageData(Data):
	def __init__(self, feature_path, type_of_feature, category_mapping):
		super().__init__(feature_path, type_of_feature, category_mapping)


class TextData(Data):
	def __init__(self, feature_path, type_of_feature, category_mapping):
		super().__init__(feature_path, type_of_feature, category_mapping)


class UserData(Data):
	def __init__(self, feature_path, type_of_feature, category_mapping):
		super().__init__(feature_path, type_of_feature, category_mapping)


class Score_Category():
	def __init__(self, score, rlist, file):
		self.score = score
		self.rlist = rlist
		self.file = file

	def string_rep(self):
		temp = self.file + "\n"
		for val in self.rlist:
			temp += str(val[0]) + " : " + val[1] + "\n"

		return temp


class RankedList():
	def __init__(self, actual_category, predicted_category):
		self.ranked_list = []
		self.actual_category = actual_category
		self.predicted_category = predicted_category

	def insert(self, score, rlist, file):
		if len(self.ranked_list) == 0:
			self.ranked_list.append(Score_Category(score, rlist, file))

		elif len(self.ranked_list) == 1:
			if self.ranked_list[0].score > score:
				self.ranked_list.append(Score_Category(score, rlist, file))
			else:
				self.ranked_list.append(self.ranked_list[0])
				self.ranked_list[0] = Score_Category(score, rlist, file)


		elif len(self.ranked_list) == 2:
			if self.ranked_list[1].score > score:
				self.ranked_list.append(Score_Category(score, rlist, file))
			elif self.ranked_list[1].score < score and self.ranked_list[0].score > score:
				self.ranked_list.append(self.ranked_list[1])
				self.ranked_list[1] = Score_Category(score, rlist, file)
			else:
				self.ranked_list.append(self.ranked_list[1])
				self.ranked_list[1] = self.ranked_list[0]
				self.ranked_list[0] = Score_Category(score, rlist, file)

		else:
			if self.ranked_list[0].score < score:
				self.ranked_list[2] = self.ranked_list[1]
				self.ranked_list[1] = self.ranked_list[0]
				self.ranked_list[0] = Score_Category(score, rlist, file)

			elif self.ranked_list[0].score > score and self.ranked_list[1].score < score:
				self.ranked_list[2] = self.ranked_list[1]
				self.ranked_list[1] = Score_Category(score, rlist, file)

			elif self.ranked_list[1].score > score and self.ranked_list[2].score < score:
				self.ranked_list[2] = Score_Category(score, rlist, file)


	def write_to_file(self):

		if not os.path.exists(results_path + self.actual_category):
			os.makedirs(results_path + self.actual_category)

		for i in range(len(self.ranked_list)):
			with open(results_path + self.actual_category + "/" + self.predicted_category + "_" + str(i+1) + ".txt", 'w') as f:
				f.write(self.ranked_list[i].string_rep())


class FashionModel():

	def __init__(self, features, category_mapping, file_list):

		print ("STARTING NEW MODEL -----")
		self.file_list = file_list
		self.category_mapping = category_mapping
		self.features = features
		self.combined_features = Data(combined_features_path, "Combined", self.category_mapping)
		self.model = OneVsRestClassifier(RandomForestClassifier(n_estimators = 50), n_jobs = -1)

		self.concatenate_features()
		self.shuffle_data()
		self.k_fold()


	#Concatenate all the feature vectors together
	def concatenate_features(self):
		print("[INFO] concatenating the features...")

		#The label vector remains unchanged
		self.combined_features.training_labels = self.features[0].training_labels

		#Find the size of the final concatenated vector
		num_rows = self.features[0].training_features.shape[0]
		num_cols = np.full(len(self.features) + 1, 0)

		#Find the sizes of the individual features
		pos = 1
		for feature in self.features:
			num_cols[pos] = int(num_cols[pos-1] + feature.training_features.shape[1])
			pos += 1

		#Append the individual features together
		self.combined_features.training_features = np.empty((num_rows, num_cols[-1]))

		pos = 1
		for feature in self.features:
			self.combined_features.training_features[:, num_cols[pos-1] : num_cols[pos]] = feature.training_features
			pos += 1

		#print (self.combined_features.training_features.shape, self.combined_features.training_labels.shape)


	def shuffle_data(self):
		print("[INFO] shuffling data...")
		#Shuffle data here and divide into train and test
		num_of_samples = self.combined_features.training_features.shape[0]

		indices = np.arange(num_of_samples)
		np.random.shuffle(indices)

		self.combined_features.training_features = self.combined_features.training_features[indices]
		self.combined_features.training_labels = self.combined_features.training_labels[indices]
		self.file_list = self.file_list[indices]


	def k_fold(self):

		num_of_samples = self.combined_features.training_features.shape[0]
		indices = np.arange(num_of_samples)

		train_indices = indices[:int(.8*num_of_samples)]
		test_indices = indices[int(.8*num_of_samples):]

		x_train, y_train = self.split_into_train_test(train_indices, test_indices)
		self.fit_model(x_train, y_train, trained = False)
		self.predict(enable_confusion = True)

		"""
		kf = KFold(n_splits = 5)
		results = []
		for train_indices, test_indices in kf.split(indices):
			print ()
			print("[INFO] k-fold ...")
			x_train, y_train = self.split_into_train_test(train_indices, test_indices)
			self.fit_model(x_train, y_train, trained = False)
			accuracy = self.predict(enable_confusion = True)
			results.append(accuracy)

		print ("Average accuracy : " , np.mean(np.asarray(results)))
		print ("\n")
		"""

	def split_into_train_test(self, train_indices, test_indices):

		print("[INFO] splitting into test and train ...")
		self.combined_features.test_features = self.combined_features.training_features[test_indices]
		x_train = self.combined_features.training_features[train_indices]
		self.combined_features.test_labels = self.combined_features.training_labels[test_indices]
		y_train = self.combined_features.training_labels[train_indices]
		self.test_file_list = self.file_list[test_indices]

		#print (x_train.shape, self.combined_features.test_features.shape)
		#print (y_train.shape, self.combined_features.test_labels.shape)

		return x_train, y_train


	def fit_model(self, x_train, y_train, trained = False):
		print("[INFO] Fitting the model...")

		starttime = time.time()
		if trained == True:
			self.model = pickle.load(open(pickle_file, 'rb'))

		else:
			self.model.fit(x_train, y_train)
			pickle.dump(self.model, open(pickle_file, 'wb'))

		endtime = time.time()
		print ("Time taken to train: ", endtime - starttime)


	def predict(self, enable_confusion = False):
		print("[INFO] Predicting...")
		starttime = time.time()
		predictions = self.model.predict(self.combined_features.test_features)
		prob_predictions = self.model.predict_proba(self.combined_features.test_features)
		endtime = time.time()

		accuracy = accuracy_score(predictions, self.combined_features.test_labels)*100.0
		print ("Accuracy : ", accuracy)
		#print (accuracy_score(predictions, self.combined_features.test_labels, normalize = False))
		print ("Time taken to predict : ", endtime - starttime)

		if enable_confusion == False:
			return accuracy

		con_matrix = {}

		for count, category in self.category_mapping.items():
			con_matrix[category] = {}
			for i_count, i_category in self.category_mapping.items():
				con_matrix[category][i_category] = RankedList(category, i_category)


		for i in range(predictions.shape[0]):
			#print (i, self.file_list[i])
			actual_label = self.category_mapping[self.combined_features.test_labels[i]]
			predicted_label = self.category_mapping[predictions[i]]
			#print ("Actual : ", actual_label, ", Predicted : ", predicted_label)
			#print ("\n")

			#Sort the probability layer to get the ranked list
			sorted_indices = np.argsort(prob_predictions[i])[::-1]
			ranked_list = []
			for index in sorted_indices:
				ranked_list.append((prob_predictions[i][index], self.category_mapping[index]))
			
			#Print the ranked list
			#for cat in ranked_list:
			#	print (cat[0], cat[1])

			
			con_matrix[actual_label][predicted_label].insert(ranked_list[0][0], ranked_list, self.test_file_list[i])


		for count, category in self.category_mapping.items():
			for i_count, i_category in self.category_mapping.items():
				con_matrix[category][i_category].write_to_file()

		for i in range(len(con_matrix.keys())):
			print (self.category_mapping[i], end=' ')

		print ("\n")

		c_matrix = confusion_matrix(self.combined_features.test_labels, predictions)
		print (c_matrix)
		

def get_categories():
	print("[INFO] getting categories for...")

	category_mapping = {}
	
	#Get selected categories
	temp = []
	with open(top_categories, 'r') as f:
		for cat in f:
			cat = cat.strip()
			temp.append(cat)

	# Get categories
	#for (dirpath, dirnames, filenames) in walk(categories_path + "images/"):
	#	temp.extend(dirnames)

	count = 0
	for category in temp:
		category_mapping[count] = category
		count += 1

	#print (category_mapping)

	#Write category mapping to file
	with open('../features/category_mapping.txt', 'w') as f:
		for key, value in category_mapping.items():
			f.write(value + "_" + str(key) + "\n")

	return category_mapping


def get_files(category_mapping):
	print("[INFO] getting files")

	file_list = []

	#train_features and train_labels are np arrays with size (no.of samples, 4096) and (no.of samples,) for images
	for count, category in category_mapping.items():
		category_list = np.load(files_features_path + category + "/data.npy")
		#print (category, category_list.shape)

		category_list = category_list.tolist()
		file_list.extend(category_list)

	file_list = np.asarray(file_list)
	return file_list

	#print (self.file_list.shape)


if __name__ == "__main__":

	category_mapping = get_categories()
	file_list = get_files(category_mapping)

	starttime = time.time()
	
	text_feature = TextData(text_features_path,"text", category_mapping)
	user_feature = UserData(user_features_path, "user", category_mapping)
	image_feature = ImageData(image_features_path, "image", category_mapping)

	text_feature.get_features()
	user_feature.get_features()
	image_feature.get_features()

	endtime = time.time()
	print ("Time taken to get features : ", endtime - starttime)

	#print("Text only")
	#f = FashionModel([text_feature], category_mapping, file_list)
	#print("User only")
	#f = FashionModel([user_feature], category_mapping, file_list)
	#print("User, text")
	#f = FashionModel([user_feature, text_feature], category_mapping, file_list)
	#print("image only")
	#f = FashionModel([image_feature], category_mapping, file_list)
	#print("Image, text")
	#f = FashionModel([image_feature, text_feature], category_mapping, file_list)
	#print("Image, user")
	#f = FashionModel([image_feature, user_feature], category_mapping, file_list)
	print("ALL")
	f = FashionModel([image_feature, user_feature, text_feature], category_mapping, file_list)


"""
#movielens = fetch_movielens()

#train, test = movielens['train'], movielens['test']
#print (train.toarray()[0])

#learning rate
alpha = 1e-5
#The dimension of the feature latent space
num_components = 32
#Number of epochs
epochs = 70
#The number of samples to try to get a negative instance before stopping
max_samples = 3

warp_model = LightFM(no_components = num_components, loss='warp', learning_schedule = 'adagrad', 
	max_sampled = max_samples, user_alpha = alpha, item_alpha = alpha)

warp_duration = []
warp_auc = []

for epoch in range(epochs):
	print (epoch)
	start = time.time()
	warp_model.fit_partial(train, epochs = 1, verbose = False)
	warp_duration.append(time.time() - start)
	warp_auc.append(auc_score(warp_model, test, train_interactions=train).mean())

fig1 = plt.figure()
x = np.arange(epochs)
plt.plot(x, np.array(warp_duration))
plt.legend(['WARP duration'], loc='upper right')
plt.title('Duration')
fig1.savefig('duration.png')

fig2 = plt.figure()
x = np.arange(epochs)
plt.plot(x, np.array(warp_auc))
plt.legend(['WARP Accuracy'], loc='upper right')
plt.title('Accuracy')
fig2.savefig('accuracy.png')
"""