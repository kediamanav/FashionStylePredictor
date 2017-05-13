"""
Description:

Hashtag_Vectors.py obtains vector representations for the hastags
"""

__author__ = "Vinitha Ravichandran"

import json,os
import numpy as np
from collections import defaultdict
from gensim.models.keyedvectors import KeyedVectors
from wordsegment import segment

#file = "hashtags.json"


def makeFeatureVec(words, model, num_features, index2word_set):

    featureVec = np.zeros((num_features,),dtype="float32")
    invalidHashtagsFlag= False

    nwords = 0.
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
        else:
            #print(word)
            wordlist=segment(word)
            #print(wordlist)
            for single_word in wordlist:
                #print("seeg "+ str(type(wordlist)))
                try:
                    if single_word in index2word_set:
                        #print("sw "+str(type(single_word)))
                        nwords+=1.
                        featureVec = np.add(featureVec, model[word])
                        #print("hhh"+single_word)

                except KeyError:
                    pass
                except Exception as e:
                    print(e)

    # Divide the result by the number of words to get the average
    #print("nwords "+str(nwords))
    if (nwords>0):
        featureVec = np.divide(featureVec,nwords)
    else:
        invalidHashtagsFlag=True
    return featureVec,invalidHashtagsFlag


if __name__== "__main__":
    path="../data/crawling_results_aaron/data/"


    #path="../crawling_results/data/"
    hastags = defaultdict(list)
    missingHashtags=[]
    key = "hashtags"

    model = KeyedVectors.load_word2vec_format('../models/GoogleNews-vectors-negative300.bin', binary=True)
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)

    categories = []
    for (dirpath, dirnames, files) in os.walk(path):
        #categories.extend(dirnames)
        pass

    categories.append("spring")
    categories.append("sporty")
    categories.append("stripes")
    
    for category in categories:
        print(category)
        files = []
        for (dirpath, dirnames, data) in os.walk(path + category):
            files.extend(data)


        avgVectors=[]
        for file in files:
            id=os.path.splitext(file)[0]
            fullpath = os.path.join(dirpath, file)

            try:
                with open(fullpath, 'r') as f:
                    json_data = json.load(f)
                #check for case where hashtags is present in the data
                #if ((key in json_data) and bool(json_data[key])):
                    tags = json_data["hashtags"]
                    hastags[id] = [x.strip() for x in tags]
                    avgVector,invalidHashtagFlag=makeFeatureVec(hastags[id],model,300, index2word_set)
                    avgVectors.append(avgVector)
                    if invalidHashtagFlag:
                        missingHashtags.append(fullpath)
            except Exception as e:
                #print("out")
                print(e)
            #with open("../crawling_results/text/"+ os.path.basename(dirpath), 'w') as f:
                #json.dump(np.array(avgVectors), f)

        if not os.path.exists("../features/text/"+category):
                os.makedirs("../features/text/"+category)

        np.save("../features/text/"+category+"/data.npy",np.asarray(avgVectors))

    
    with open("InvalidHashtags_aaron.txt",'w') as f:
        for file in missingHashtags:
            f.write(file+"\n")




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



