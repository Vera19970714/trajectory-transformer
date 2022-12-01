import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def getSimCnnFeature():
    #specify the data
    #dumbData = np.zeros((80, 28, 256))
    dumbData = np.load('../dataset/cnnFeature.npy')
    INDEX = 0
    data = dumbData[INDEX]  #size 28, 256

    tgt = data[0:1] #1, 256
    #tgt = np.tile(tgt, (27, 1))
    rest = data[1:] #27, 256

    sim = cosine_similarity(tgt, rest)
    print(sim)
    return sim