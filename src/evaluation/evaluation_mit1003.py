import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from nltk.metrics import edit_distance
from collections import defaultdict
#import cv2
#import os

np.random.seed(1)

class EvaluationMetric():
    def __init__(self,trainingGrid,evaluationGrid=4,minLen=10):
        self.minLen = minLen
        self.trainingGrid = trainingGrid
        self.evaluationGrid = evaluationGrid
        self.scaleGrid = 8
        if trainingGrid == -1:
            file1 = open('evaluation/indexConversion.txt', mode='r')
            centerModeIndex = file1.read()
            file1.close()
            list1 = centerModeIndex.split('\n')
            self.conversionList = [int(x.split(' ')[-1]) for x in list1]
            
            file2 = open('../dataset/MIT1003/centerModeIndex.txt', mode='r')
            centerModeRange = file2.read()
            file2.close()
            list2 = centerModeRange.split('\n')
            self.conversionRange = [x.replace(',','').split(' ')[:5] for x in list2]
            
    def scanpath_to_string(self,
            scanpath
    ):
        string = ''
        step = self.trainingGrid / self.evaluationGrid
        for i in range(np.shape(scanpath)[0]):
            if self.trainingGrid != -1:
                fixationX = scanpath[i] % self.trainingGrid
                fixationY = scanpath[i] // self.trainingGrid
                new_fixation = (fixationX // step)+ (fixationY // step) *self.evaluationGrid
            else:
                new_fixation = self.conversionList[int(scanpath[i])]
               
            string += chr(97 + int(new_fixation))
        return string
    
    def scanpath_to_pixel(self,
            scanpathPixel,
            scanpath_pre,
            img_height,
            img_weight
    ):
        scanpathLength = len(scanpath_pre)
        scanpathPixelLength = len(scanpathPixel)
        scanpathPixel_gt_img = np.zeros((img_height, img_weight))
        scanpath_pre_img = np.zeros((img_height, img_weight))
        print(scanpathPixel.shape)
        print(scanpathPixel)
        print(scanpath_pre)
        print(scanpath_pre.shape)
        for i in range(scanpathPixelLength):
            scanpathPixel_gt_img[int(scanpathPixel[i][0]) - 1, int(scanpathPixel[i][1]) - 1] = 1
        for m in range(scanpathLength):
            if self.trainingGrid != -1:
                stepX = img_weight / self.trainingGrid
                stepY = img_height / self.trainingGrid
                fixationX = scanpath_pre[m] % self.trainingGrid
                fixationY = scanpath_pre[m] // self.trainingGrid
                scanpath_pre_img[int(fixationY * stepY):int((fixationY+1) * stepY),int(fixationX * stepX):int((fixationX+1) * stepX)] = 1 / stepX*stepY
            else:
                stepX = img_weight / self.scaleGrid
                stepY = img_height / self.scaleGrid
                fixationX_left = self.conversionRange[int(scanpath_pre[m])+1][3]
                fixationX_right = self.conversionRange[int(scanpath_pre[m])+1][4]
                fixationY_top = self.conversionRange[int(scanpath_pre[m])+1][1]
                fixationY_down = self.conversionRange[int(scanpath_pre[m])+1][2]
                scanpath_pre_img[int((int(fixationY_top) * stepY)):int((int(fixationY_down) * stepY)),int((int(fixationX_left) * stepX)):int((int(fixationX_right) * stepX))] = 1 / ((int(fixationX_right)-int(fixationX_left))*stepX*(int(fixationY_down)-int(fixationY_top))*stepY)
                # scanpath_pre_img[int((int(fixationY_top) * stepY)):int((int(fixationY_down) * stepY)),int((int(fixationX_left) * stepX)):int((int(fixationX_right) * stepX))] = 1 
                
        return scanpathPixel_gt_img, scanpath_pre_img

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

#########################################################################################
##############################  saliency metrics  #######################################
#########################################################################################

    def AUC_Judd(self, saliencyMap, fixationMap, jitter=True, toPlot=False):
        # saliencyMap is the saliency map
        # fixationMap is the human fixation map (binary matrix)
        # jitter=True will add tiny non-zero random constant to all map locations to ensure
        # 		ROC can be calculated robustly (to avoid uniform region)
        # if toPlot=True, displays ROC curve
        # If there are no fixations to predict, return NaN
        if not fixationMap.any():
            print('Error: no fixationMap')
            score = float('nan')
            return score
        # make the saliencyMap the size of the image of fixationMap
        if not np.shape(saliencyMap) == np.shape(fixationMap):
            # from scipy.misc import imresize
            from skimage.transform import resize
            saliencyMap = resize(saliencyMap, np.shape(fixationMap))
        # jitter saliency maps that come from saliency models that have a lot of zero values.
        # If the saliency map is made with a Gaussian then it does not need to be jittered as
        # the values are varied and there is not a large patch of the same value. In fact
        # jittering breaks the ordering in the small values!
        if jitter:
            # jitter the saliency map slightly to distrupt ties of the same numbers
            saliencyMap = saliencyMap + np.random.random(np.shape(saliencyMap)) / 10 ** 7
        # normalize saliency map
        saliencyMap = (saliencyMap - saliencyMap.min()) \
                      / (saliencyMap.max() - saliencyMap.min())
        if np.isnan(saliencyMap).all():
            print('NaN saliencyMap')
            score = float('nan')
            return score
        S = saliencyMap.flatten()
        F = fixationMap.flatten()
        Sth = S[F > 0]  # sal map values at fixation locations
        Nfixations = len(Sth)
        Npixels = len(S)
        allthreshes = sorted(Sth, reverse=True)  # sort sal map values, to sweep through values
        tp = np.zeros((Nfixations + 2))
        fp = np.zeros((Nfixations + 2))
        tp[0], tp[-1] = 0, 1
        fp[0], fp[-1] = 0, 1
        for i in range(Nfixations):
            thresh = allthreshes[i]
            aboveth = (S >= thresh).sum()  # total number of sal map values above threshold
            tp[i + 1] = float(i + 1) / Nfixations  # ratio sal map values at fixation locations
            # above threshold
            fp[i + 1] = float(aboveth - i) / (Npixels - Nfixations)  # ratio other sal map values
            # above threshold
        score = np.trapz(tp, x=fp)
        allthreshes = np.insert(allthreshes, 0, 0)
        allthreshes = np.append(allthreshes, 1)
        if toPlot:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(1, 2, 1)
            ax.matshow(saliencyMap, cmap='gray')
            ax.set_title('SaliencyMap with fixations to be predicted')
            [y, x] = np.nonzero(fixationMap)
            s = np.shape(saliencyMap)
            plt.axis((-.5, s[1] - .5, s[0] - .5, -.5))
            plt.plot(x, y, 'ro')
            ax = fig.add_subplot(1, 2, 2)
            plt.plot(fp, tp, '.b-')
            ax.set_title('Area under ROC curve: ' + str(score))
            plt.axis((0, 1, 0, 1))
            plt.show()
        return score


    def AUC_shuffled(self, saliencyMap, fixationMap, otherMap, Nsplits=100, stepSize=0.1, toPlot=False):
        '''saliencyMap is the saliency map
        fixationMap is the human fixation map (binary matrix)
        otherMap is a binary fixation map (like fixationMap) by taking the union of
        fixations from M other random images (Borji uses M=10)
        Nsplits is number of random splits
        stepSize is for sweeping through saliency map
        if toPlot=1, displays ROC curve
        '''
        # saliencyMap = saliencyMap.transpose()
        # fixationMap = fixationMap.transpose()
        # otherMap = otherMap.transpose()
        # If there are no fixations to predict, return NaN
        if not fixationMap.any():
            print('Error: no fixationMap')
            score = float('nan')
            return score
        if not np.shape(saliencyMap) == np.shape(fixationMap):
            saliencyMap = np.array(
                Image.fromarray(saliencyMap).resize((np.shape(fixationMap)[1], np.shape(fixationMap)[0])))
        if np.isnan(saliencyMap).all():
            print('NaN saliencyMap')
            score = float('nan')
            return score
        # normalize saliency map
        saliencyMap = (saliencyMap - saliencyMap.min()) \
                      / (saliencyMap.max() - saliencyMap.min())
        S = saliencyMap.flatten(order='F')
        F = fixationMap.flatten(order='F')
        Oth = otherMap.flatten(order='F')
        Sth = S[F > 0]  # sal map values at fixation locations
        Nfixations = len(Sth)
        # for each fixation, sample Nsplits values from the sal map at locations specified by otherMap
        ind = np.nonzero(Oth)[0]  # find fixation locations on other images
        Nfixations_oth = min(Nfixations, len(ind))
        randfix = np.empty((Nfixations_oth, Nsplits))
        randfix[:] = np.nan
        for i in range(Nsplits):
            randind = ind[np.random.permutation(len(ind))]  # randomize choice of fixation locations
            randfix[:, i] = S[
                randind[:Nfixations_oth]]  # sal map values at random fixation locations of other random images
        # calculate AUC per random split (set of random locations)
        auc = np.empty(Nsplits)
        auc[:] = np.nan
        def Matlab_like_gen(start, stop, step, precision):
            r = start
            while round(r, precision) <= stop:
                yield round(r, precision)
                r += step
        for s in range(Nsplits):
            curfix = randfix[:, s]
            i0 = Matlab_like_gen(0, max(np.maximum(Sth, curfix)), stepSize, 5)
            allthreshes = [x for x in i0]
            allthreshes.reverse()
            tp = np.zeros((len(allthreshes) + 2))
            fp = np.zeros((len(allthreshes) + 2))
            tp[0], tp[-1] = 0, 1
            fp[0], fp[-1] = 0, 1
            for i in range(len(allthreshes)):
                thresh = allthreshes[i]
                tp[i + 1] = (Sth >= thresh).sum() / Nfixations
                fp[i + 1] = (curfix >= thresh).sum() / Nfixations_oth
            auc[s] = np.trapz(tp, x=fp)
        score = np.mean(auc)  # mean across random splits
        if toPlot:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(1, 2, 1)
            ax.matshow(saliencyMap, cmap='gray')
            ax.set_title('SaliencyMap with fixations to be predicted')
            [y, x] = np.nonzero(fixationMap)
            s = np.shape(saliencyMap)
            plt.axis((-.5, s[1] - .5, s[0] - .5, -.5))
            plt.plot(x, y, 'ro')
            ax = fig.add_subplot(1, 2, 2)
            plt.plot(fp, tp, '.b-')
            ax.set_title('Area under ROC curve: ' + str(score))
            plt.axis((0, 1, 0, 1))
            plt.show()
        return score
    def KLdiv(self, saliencyMap, fixationMap):
        # saliencyMap is the saliency map
        # fixationMap is the human fixation map
        # convert to float
        map1 = saliencyMap.astype(float)
        map2 = fixationMap.astype(float)
        # make sure maps have the same shape
        from skimage.transform import resize
        map1 = resize(map1, np.shape(map2))
        # make sure map1 and map2 sum to 1
        if map1.any():
            map1 = map1 / map1.sum()
        if map2.any():
            map2 = map2 / map2.sum()
        # compute KL-divergence
        eps = 10 ** -12
        score = map2 * np.log(eps + map2 / (map1 + eps))
        return score.sum()


    def NSS(self, saliencyMap, fixationMap):
        # saliencyMap is the saliency map
        # fixationMap is the human fixation map (binary matrix)
        # If there are no fixations to predict, return NaN
        if not fixationMap.any():
            print('Error: no fixationMap')
            score = float('nan')
            return score
        # make sure maps have the same shape
        from skimage.transform import resize
        map1 = resize(saliencyMap, np.shape(fixationMap))
        if not map1.max() == 0:
            map1 = map1.astype(float) / map1.max()
        # normalize saliency map
        if not map1.std(ddof=1) == 0:
            map1 = (map1 - map1.mean()) / map1.std(ddof=1)
        # mean value at fixation locations
        score = map1[fixationMap.astype(bool)].mean()
        return score


    def InfoGain(self, saliencyMap, fixationMap, baselineMap):
        '''saliencyMap is the saliency map
        fixationMap is the human fixation map (binary matrix)
        baselineMap is another saliency map (e.g. all fixations from other images)'''
        map1 = np.resize(saliencyMap, np.shape(fixationMap))
        mapb = np.resize(baselineMap, np.shape(fixationMap))
        # normalize and vectorize saliency maps
        map1 = (map1.flatten(order='F') - np.min(map1)) / (np.max(map1 - np.min(map1)))
        mapb = (mapb.flatten(order='F') - np.min(mapb)) / (np.max(mapb - np.min(mapb)))
        # mapb = mapb.flatten(order='F')
        # turn into distributions
        map1 /= np.sum(map1)
        mapb /= np.sum(mapb)
        fixationMap = fixationMap.flatten(order='F')
        locs = fixationMap > 0
        eps = 2.2204e-16
        score = np.mean(np.log2(eps + map1[locs]) - np.log2(eps + mapb[locs]))

        return score

    def normalize_map(self,s_map):
        # normalize the salience map (as done in MIT code)
        norm_s_map = (s_map - np.min(s_map)) / ((np.max(s_map) - np.min(s_map)) * 1.0)
        return norm_s_map

    def saliencyEvaluation(self,scanpathPixel_gt, scanpath_pre,img_height,img_weight):
        # we need transfer region scanpath to pixcel scanpath / transfer scanpath to fixation map
        scanpathPixel_gt_img, scanpath_pre_img = self.scanpath_to_pixel(scanpathPixel_gt, scanpath_pre,img_height,img_weight)
        scanpath_pre_img = self.normalize_map(scanpath_pre_img)
        scanpathPixel_gt_img = self.normalize_map(scanpathPixel_gt_img)
        auc = self.AUC_Judd(scanpath_pre_img,scanpathPixel_gt_img)
        nss = self.NSS(scanpath_pre_img,scanpathPixel_gt_img)
        return auc, nss


if __name__ == '__main__':
    eval = EvaluationMetric(trainingGrid=-1)
    print(eval.saliencyEvaluation(np.array([[113,200],[160,312],[250,410]]),np.array([21, 26, 25]),768,1024))
    