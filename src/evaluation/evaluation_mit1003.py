import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from nltk.metrics import edit_distance
from collections import defaultdict

class EvaluationMetric():
    def __init__(self, minLen=10):
        self.minLen = minLen

    def scanpath_to_string(self,
            scanpath
    ):
        string = ''

        for i in range(np.shape(scanpath)[0]):
            fixation = scanpath[i].astype(np.int32)
            string += chr(97 + int(fixation))

        return string

    def string_edit_distance(self,
            scanpath_1,
            scanpath_2
    ):
        string_1 = self.scanpath_to_string(scanpath_1)
        string_2 = self.scanpath_to_string(scanpath_2)

        return edit_distance(string_1, string_2, transpositions=True)

    def string_based_time_delay_embedding_distance(self,
            scanpath_1,
            scanpath_2,
            # options
            k=3,  # time-embedding vector dimension
            distance_mode='Mean'
    ):
        # human_scanpath and simulated_scanpath can have different lenghts
        # They are list of fixations, that is couple of coordinates
        # k must be shorter than both lists lenghts

        # we check for k be smaller or equal then the lenghts of the two input scanpaths
        if len(scanpath_1) < k or len(scanpath_2) < k:
            print('ERROR: Too large value for the time-embedding vector dimension')
            return False

        # create time-embedding vectors for both scanpaths

        scanpath_1_vectors = []
        for i in np.arange(0, len(scanpath_1) - k + 1):
            scanpath_1_vectors.append(scanpath_1[i:i + k])

        scanpath_2_vectors = []
        for i in np.arange(0, len(scanpath_2) - k + 1):
            scanpath_2_vectors.append(scanpath_2[i:i + k])

        # in the following cicles, for each k-vector from the simulated scanpath
        # we look for the k-vector from humans, the one of minumum distance
        # and we save the value of such a distance, divided by k

        distances = []

        for s2_k_vec in scanpath_2_vectors:

            # find human k-vec of minimum distance

            norms = []

            for s1_k_vec in scanpath_1_vectors:
                d = self.string_edit_distance(s1_k_vec, s2_k_vec)
                norms.append(d)

            distances.append(min(norms) / k)

        # at this point, the list "distances" contains the value of
        # minumum distance for each simulated k-vec
        # according to the distance_mode, here we compute the similarity
        # between the two scanpaths.

        if distance_mode == 'Mean':
            return sum(distances) / len(distances)
        elif distance_mode == 'Hausdorff':
            return max(distances)
        else:
            print('ERROR: distance mode not defined.')
            return False

    def get_sed_and_sbtde(self, scanpath_gt, scanpath_pre):
        sed_i = np.stack(
            [self.string_edit_distance(scanpath_gt[:i], scanpath_pre[:i]) for i in
             range(1, self.minLen + 1)]).mean()
        sbtde_i = np.stack(
            [self.string_based_time_delay_embedding_distance(scanpath_gt, scanpath_pre, k) for k in
             range(1, self.minLen + 1)]).mean()
        return sed_i, sbtde_i

    def get_sppSed_and_sppSbtde(self, all_pre_sppSed_scanpaths,all_pre_sppSbtde_scanpaths):

        '''
        :param self:
        :param all_pre_sppSed_scanpaths: list:[('imageName',sed),...)
        :param all_pre_sppSbtde_scanpaths: list:[('imageName',sbtde),...)
        :return: sppSed:list, sppSbtde:list
        '''

        sppSedDict, sppSbtdeDict = defaultdict(list), defaultdict(list)
        for k, v in all_pre_sppSed_scanpaths:
            sppSedDict[k].append(v)
        for k, v in all_pre_sppSbtde_scanpaths:
            sppSbtdeDict[k].append(v)
        sppSED = [min(list(sppSedDict.values())[i]) for i in range(len(sppSedDict))]
        sppSBTDE = [min(list(sppSbtdeDict.values())[i]) for i in range(len(sppSbtdeDict))]

        return sppSED,sppSBTDE
