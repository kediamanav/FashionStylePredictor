"""
Description:

Hashtag_Vectors.py obtains vector representations for the hastags
"""

__author__ = "Vinitha Ravichandran"

import json,logging,os
import numpy as np
from collections import defaultdict
from gensim.models.keyedvectors import KeyedVectors
#file = "hashtags.json"


def makeFeatureVec(words, model, num_features):

    featureVec = np.zeros((num_features,),dtype="float32")

    nwords = 0.

    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
        #else:
            #print(word)
    
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


if __name__== "__main__":
    path="../data/crawling_results_manav/data/"
    #path="../crawling_results/data/"
    hastags = defaultdict(list)
    missingHashtags=[]
    key = "hashtags"

    model = KeyedVectors.load_word2vec_format('../models/GoogleNews-vectors-negative300.bin', binary=True)


    #counter=0
    avgVectors=[]

    categories = []
    for (dirpath, dirnames, files) in os.walk(path):
        ##hastags[dirnames+"_"+files]=[]
        categories.extend(dirnames)

    for category in categories:
	print(category)
        files = []
        for (dirpath, dirnames, data) in os.walk(path + category):
            files.extend(data)

        for file in files:
            #id = os.path.basename(dirpath) + "_" + os.path.splitext(file)[0]
            #print(category, file)
            id=os.path.splitext(file)[0]
            fullpath = os.path.join(dirpath, file)

            try:
                with open(fullpath, 'r') as f:
                    json_data = json.load(f)
                    #check for case where hashtags is present in the data
                    #if ((key in json_data) and len(json_data[key])>0):
                    tags = json_data["hashtags"]
                    hastags[id] = [x.strip() for x in tags]
                    avgVector=makeFeatureVec(hastags[id],model,300)
                    avgVectors.append(avgVector)
                    
    
            except Exception as e:
                print(e)


        #with open("../crawling_results/text/"+ os.path.basename(dirpath), 'w') as f:
            #json.dump(np.array(avgVectors), f)

        if not os.path.exists("../features/text/"+category):
            os.makedirs("../features/text/"+category)

        np.save("../features/text/"+category+"/data.npy",np.asarray(avgVectors))


    '''with open("MissingHashtags_aaron.txt",'a') as f:
        for file in missingHashtags:
            f.write(file+"\n")'''


    #hashtags=list(prehashtags.values())
    #logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    #ex=["fashion", "fashionblogger", "outfit", "alisa", "sia", "alisasia", "ootd", "ootn", "wiw", "wiwt", "fashionista", "barbie", "dolly", "doll", "cat", "kitty", "meow", "cathat", "hat", "winter", "fall", "orange", "skirt", "white", "top", "heels", "chunkyheel", "ankleboots", "laceup"]
    #sentences = [['first', 'sentence'], ['second', 'sentence']]
    # train word2vec on the two sentences
    #model = Word2Vec(hashtags, min_count=1)
    #model.save('../categories/model_wospace_1min')
    #model=KeyedVectors.load_word2vec_format('../Models/GoogleNews-vectors-negative300.bin', binary=True)

   # print(model['fashion'])
    #print(makeFeatureVec(ex,model,300))





    #new_model = Word2Vec.load('../categories/4model')
    ##print(model2.most_similar(positive=['woman', 'king'], negative=['man'], topn=1))
    #[('queen', 0.50882536)]
    #model.doesnt_match("breakfast cereal dinner lunch",.split())
    #model.doesnt_match("breakfast cereal dinner lunch")
    #'cereal'profession
   # print("model similarity ",model.similarity('woman', 'girl'),sep=" ")

    #print(model['dolly'])

    #print("corpus count",model.corpus_count,sep=" ")
    #print(model.syn1neg.shape)
    #print(model.most_similar("girl"))



