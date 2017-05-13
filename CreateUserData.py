"""
Description:

Hashtag_Vectors.py obtains vector representations for the hastags
"""

import json,os
import numpy as np


import json,os
import numpy as np

if __name__== "__main__":
    path="../new_crawling_results/crawling_results_aaron/data/"

    userVectors=[]

    # load userdata
    with open('userdata.json') as f:
        userdata = json.load(f)
    #print(np.load('data.npy'))


    categories = []
    for (dirpath, dirnames, files) in os.walk(path):
        categories.extend(dirnames)

    for category in categories:
        print(category)
        files = []
        userVectors = []

        for (dirpath, dirnames, data) in os.walk(path + category):
            files.extend(data)


        for file in files:
            id=os.path.splitext(file)[0]
            fullpath = os.path.join(dirpath, file)
            try:
                with open(fullpath, 'r') as f:
                    json_data = json.load(f)
                    url = json_data["user"]["url"]
                    #print(userdata[url])
                    userVector = userdata[url]
                    userVectors.append(userVector)

            except Exception as e:
                #print("out")
                print(e)
                print(fullpath)


        if not os.path.exists("../features/user/"+category):
                os.makedirs("../features/user/"+category)

        np.save("../features/user/"+category+"/data.npy",np.asarray(userVectors))














