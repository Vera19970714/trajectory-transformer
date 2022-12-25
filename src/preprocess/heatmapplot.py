import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import cv2
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.cm 
from matplotlib.colors import LinearSegmentedColormap 
from tqdm import tqdm
import torch.nn as nn
import torch
import torch.nn.functional as F

dumbData = './dataset/cnnFeature.npy'
gazegt = './dataset/outputdata/gaze_gt_threedim.csv'
gazetrue = './dataset/outputdata/gaze_true_threedim.csv'
gazemax = './dataset/outputdata/gaze_max_threedim_new.csv'
logitsgt = './dataset/outputdata/logits_gt_threedim.csv'
logitsmax = './dataset/outputdata/logits_max_threedim_new.csv'
gazeexpect = './dataset/outputdata/gaze_expect_threedim_new.csv'
gazerandom = './dataset/outputdata/gaze_random_new.csv'
gazesimilarity = './dataset/outputdata/gaze_similarity_new.csv'
gazesaliency = './dataset/outputdata/gaze_saliency_new.csv'
gazergb = './dataset/outputdata/gaze_rgb_new.csv'


datapath = './dataset/processdata/dataset_Q23_similarity_mousedel_time_val'
target_dir = './dataset/img/Target/'
img_dir = './dataset/img/Question/'

save_path_1 = './dataset/output/Plot_Heatmap/Resnet/'
save_path_2 = './dataset/output/Plot_Heatmap/Rgb/'
save_path_3 = './dataset/output/Plot_Heatmap/AE/'
save_path_4 = './dataset/output/Plot_Heatmap/CNN/'
save_path_5 = './dataset/output/Plot_Heatmap/Gazegt/'
save_path_6 = './dataset/output/Plot_Heatmap/Gazemax/'
save_path_7 = './dataset/output/Plot_Heatmap/Gazetrue/'
save_path_8 = './dataset/output/Plot_Heatmap/Logitsgt/'
save_path_9 = './dataset/output/Plot_Heatmap/Logitsmax/'
save_path_10 = './dataset/output/Plot_Heatmap/Gazetruestep/'
save_path_11 = './dataset/output/Plot_Heatmap/Gazeexpect/'

class HeatmapPlot(object):
    def __init__(self):
        with open(datapath, "rb") as fp:
            raw_data = pickle.load(fp)
        self.data_length = len(raw_data)
        print(F'len = {self.data_length}')
        self.package_seq = []
        self.package_similarity = []
        self.package_saliency = []
        self.package_rgb = []
        self.package_ae = []
        self.question_name = []
        self.target_name = []
        self.target = []
        self.loss_fn1 = nn.KLDivLoss()
        self.loss_fn2 = nn.CosineSimilarity(dim=1)
        
        for item in raw_data:
            self.package_seq.append(item['package_seq'])
            self.package_similarity.append(item['package_similarity'])
            self.package_saliency.append(item['package_saliency'])
            self.package_rgb.append(item['package_rgb'])
            self.package_ae.append(item['package_ae'])
            self.question_name.append(item['package_questionname'])
            self.target_name.append(item['package_targetname'])
            self.target.append(item['package_target'])
            
        self.cnnfeature = np.load(dumbData)
        self.gazegt = np.array(pd.read_csv(gazegt))
        self.gazetrue = np.array(pd.read_csv(gazetrue))
        self.gazemax = np.array(pd.read_csv(gazemax))
        self.logitsgt = np.array(pd.read_csv(logitsgt))
        self.logitsmax = np.array(pd.read_csv(logitsmax))
        self.gazeexpect = np.array(pd.read_csv(gazeexpect))
        self.gazerandom = np.array(pd.read_csv(gazerandom))
        self.gazesimilarity = np.array(pd.read_csv(gazesimilarity))
        self.gazesaliency = np.array(pd.read_csv(gazesaliency))
        self.gazergb = np.array(pd.read_csv(gazergb))

    def getSimCnnFeature(self,data):
        #specify the data
        #dumbData = np.zeros((80, 28, 256))
        tgt = data[0:1] #1, 256
        #tgt = np.tile(tgt, (27, 1))
        rest = data[1:] #27, 256
        sim = cosine_similarity(tgt, rest)
        return sim

    def minoverlap(self,distribution1,distribution2):
        length = len(distribution1)
        total = 0
        for i in range(length):
            value1 = distribution1[0][i]
            value2 = distribution2[0][i]
            min_value = min([value1,value2])
            total += min_value
        return total

    def behaviorcal(self,each_length,target,gaze,logits):
        gaze_search = 1
        gaze_refix = 0
        gaze_revisit = 0
        gaze_length = 0
        correct = 0
        gaze_heatmap = []
        gaze_logits = []
        gazeheatmap = np.zeros((1,27)) + 0.00001
        if int(30) not in gaze:
            if gaze[-1,:]==target:
                    correct = 1
        for i in range(each_length):
            gazeheatmap_step = np.zeros((1,27))
            current_gaze = gaze[i,:]
            if int(current_gaze) == 30:
                if gaze[i-1,:]==target:
                    correct = 1
                break
            else:
                if gaze[-1,:]==target:
                    correct = 1  
                gazelogits = logits[i,:27]
                gaze_logits.append(gazelogits)
                if current_gaze < 27:
                    gaze_length += 1
                    gazeheatmap[:,int(current_gaze)]+=1
                    gazeheatmap_step[:,int(current_gaze)]=1
                    gaze_heatmap.append(gazeheatmap_step)
                    if i==1:
                        if current_gaze == gaze[i-1,:]:
                            gaze_refix += 1
                        else:
                            gaze_search += 1
                    if i > 1:
                        if current_gaze == gaze[i-1,:]:
                            gaze_refix += 1
                        elif current_gaze in gaze[:(i-1),:] and gaze[i,:] != gaze[i-1,:]:
                            gaze_revisit += 1
                        else:
                            gaze_search += 1
        total_behavior = int(gaze_search) + int(gaze_refix) + int(gaze_revisit)
        search_per = int(gaze_search) / total_behavior
        refix_per = int(gaze_refix) / total_behavior
        revisit_per = int(gaze_revisit) / total_behavior
        gazeheatmap = gazeheatmap / gazeheatmap.sum()
       
        return gaze_length,correct,search_per,refix_per,revisit_per,gaze_heatmap,gazeheatmap,gaze_logits
        
            
    def heatmapplot(self,df,img,min,max,alpha_trans,questionname,targetpack,step_name,name,path):
        fig = plt.figure(figsize=(10, 4.3), dpi=120)
        wd = matplotlib.cm.winter._segmentdata  
        wd['alpha'] =  ((0.0, 0.0, 0.3), 
                        (0.3, 0.3, 1.0),
                        (1.0, 1.0, 1.0))
        al_winter = LinearSegmentedColormap('AlphaWinter', wd) 
        hmax = sns.heatmap(data=df,
                    vmin=min, 
                    vmax=max,
                    cmap=plt.get_cmap('Greens'),
                    alpha = alpha_trans, # whole heatmap is translucent
                    annot = True,
                    fmt = '.3f',
                    zorder = 2,
                    cbar = True,
                    mask = df < 0.025
                )
        hmax.imshow(img,
                    aspect = hmax.get_aspect(),
                    extent = hmax.get_xlim() + hmax.get_ylim(),
                    zorder = 1) #put the map under the heatmap
        # plt.show()
        plt.title(name + '_' + str(questionname)+'_' + str(targetpack),fontsize='xx-large',fontweight='heavy') 
        plt.savefig(path+ name+ '_' + str(questionname) + '_' + str(targetpack)+ step_name+'.jpg')

    def forward(self):
        sum1 = 0
        sum2 = 0
        max_length = 17
        iter = 100
        correct_gt = np.zeros((self.data_length,1))
        correct_max = np.zeros((self.data_length,1))
        correct_true = np.zeros((self.data_length,1))
        correct_expect = np.zeros((self.data_length,iter))
        correct_random = np.zeros((self.data_length,iter))
        correct_similarity = np.zeros((self.data_length,iter))
        correct_saliency = np.zeros((self.data_length,iter))
        correct_rgb = np.zeros((self.data_length,iter))

        search_gt_perc = np.zeros((self.data_length,1))
        refix_gt_perc = np.zeros((self.data_length,1)) 
        revisit_gt_perc = np.zeros((self.data_length,1))

        search_max_perc = np.zeros((self.data_length,1))
        refix_max_perc = np.zeros((self.data_length,1))
        revisit_max_perc = np.zeros((self.data_length,1))

        search_true_perc = np.zeros((self.data_length,1))
        refix_true_perc = np.zeros((self.data_length,1))
        revisit_true_perc = np.zeros((self.data_length,1))

        search_expect_perc = np.zeros((self.data_length,iter))
        refix_expect_perc = np.zeros((self.data_length,iter))
        revisit_expect_perc = np.zeros((self.data_length,iter))

        search_random_perc = np.zeros((self.data_length,iter))
        refix_random_perc = np.zeros((self.data_length,iter)) 
        revisit_random_perc = np.zeros((self.data_length,iter))

        search_similarity_perc = np.zeros((self.data_length,iter))
        refix_similarity_perc = np.zeros((self.data_length,iter)) 
        revisit_similarity_perc = np.zeros((self.data_length,iter))

        search_saliency_perc = np.zeros((self.data_length,iter))
        refix_saliency_perc = np.zeros((self.data_length,iter)) 
        revisit_saliency_perc = np.zeros((self.data_length,iter))

        search_rgb_perc = np.zeros((self.data_length,iter))
        refix_rgb_perc = np.zeros((self.data_length,iter)) 
        revisit_rgb_perc = np.zeros((self.data_length,iter))

        length_gt = np.zeros((self.data_length,1))
        length_max = np.zeros((self.data_length,1))
        length_true = np.zeros((self.data_length,1))
        length_expect = np.zeros((self.data_length,iter))
        length_random = np.zeros((self.data_length,iter))
        length_similarity = np.zeros((self.data_length,iter))
        length_saliency = np.zeros((self.data_length,iter))
        length_rgb = np.zeros((self.data_length,iter))
        
        heatmap_gt_overlaploss = np.zeros((self.data_length,1))
        heatmap_max_overlaploss = np.zeros((self.data_length,1))
        heatmap_expect_overlaploss = np.zeros((self.data_length,iter))
        heatmap_random_overlaploss = np.zeros((self.data_length,iter))
        heatmap_similarity_overlaploss = np.zeros((self.data_length,iter))
        heatmap_saliency_overlaploss = np.zeros((self.data_length,iter))
        heatmap_rgb_overlaploss = np.zeros((self.data_length,iter))

        heatmap_gt_klloss = np.zeros((self.data_length,1))
        heatmap_max_klloss = np.zeros((self.data_length,1))
        heatmap_expect_klloss = np.zeros((self.data_length,iter))
        heatmap_random_klloss = np.zeros((self.data_length,iter))
        heatmap_similarity_klloss = np.zeros((self.data_length,iter))
        heatmap_saliency_klloss = np.zeros((self.data_length,iter))
        heatmap_rgb_klloss = np.zeros((self.data_length,iter))

        for i in tqdm(range(self.data_length)):
            cnn_result = self.cnnfeature[i] #size 28, 256
            questionname = self.question_name[i]
            targetname = self.target_name[i]
            targetpack = self.target[i]
            packageseq = self.package_seq[i]
            saliency_result = self.package_saliency[i]
            resnet_result = self.package_similarity[i]
            rgb_result = self.package_rgb[i]
            ae_result = self.package_ae[i]
            cnn_result = self.getSimCnnFeature(cnn_result).reshape(3,9)
            each_length = len(packageseq)+1
            
            sum1 += each_length
            sum2 += max_length
            gazegt_each = self.gazegt[(sum1-each_length):sum1,:]
            gazemax_each = self.gazemax[(sum2-max_length):sum2,:]
            gazetrue_each = self.gazetrue[(sum1-each_length):sum1,:]
            logitsgt_each = self.logitsgt[(sum1-each_length):sum1,:]
            logitsmax_each = self.logitsmax[(sum2-max_length):sum2,:]
            gazegt_length,correctgt,search_per_gt,refix_per_gt,revisit_per_gt,gaze_heatmap_gt,gazeheatmap_gt,gaze_logits_gt = self.behaviorcal(each_length,int(targetpack)-1,gazegt_each,logitsgt_each)
            gazemax_length,correctmax,search_per_max,refix_per_max,revisit_per_max,gaze_heatmap_max,gazeheatmap_max,gaze_logits_max = self.behaviorcal(max_length,int(targetpack)-1,gazemax_each,logitsmax_each)
            gazetrue_length,correcttrue,search_per_true,refix_per_true,revisit_per_true,gaze_heatmap_true,gazeheatmap_true,_ = self.behaviorcal(each_length,int(targetpack)-1,gazetrue_each,logitsgt_each)
            correct_gt[i,:] = correctgt
            correct_max[i,:] = correctmax
            correct_true[i,:] = correcttrue

            search_gt_perc[i,:] = search_per_gt
            refix_gt_perc[i,:] = refix_per_gt
            revisit_gt_perc[i,:] = revisit_per_gt

            search_max_perc[i,:] = search_per_max
            refix_max_perc[i,:] = refix_per_max
            revisit_max_perc[i,:] = revisit_per_max

            search_true_perc[i,:] = search_per_true
            refix_true_perc[i,:] = refix_per_true
            revisit_true_perc[i,:] = revisit_per_true
            
            length_gt[i,:] = gazegt_length
            length_max[i,:] = gazemax_length
            length_true[i,:] = gazetrue_length

            heatmap_gt_overlaploss[i,:] = self.minoverlap(gazeheatmap_gt,gazeheatmap_true)
            heatmap_gt_klloss[i,:] = self.loss_fn1(torch.from_numpy(gazeheatmap_gt).log(),torch.from_numpy(gazeheatmap_true))
            heatmap_max_overlaploss[i,:] = self.minoverlap(gazeheatmap_max,gazeheatmap_true)
            heatmap_max_klloss[i,:] = self.loss_fn1(torch.from_numpy(gazeheatmap_max).log(),torch.from_numpy(gazeheatmap_true))
        
            for j in range(iter):
                gazeexpect_each = self.gazeexpect[(sum2-max_length):sum2,j].reshape(-1,1)
                gazerandom_each = self.gazerandom[(sum2-max_length):sum2,j].reshape(-1,1)
                gazesimilarity_each = self.gazesimilarity[(sum2-max_length):sum2,j].reshape(-1,1)
                gazesaliency_each = self.gazesaliency[(sum2-max_length):sum2,j].reshape(-1,1)
                gazergb_each = self.gazergb[(sum2-max_length):sum2,j].reshape(-1,1)
                gazeexpect_length,correctexpect,search_per_expect,refix_per_expect,revisit_per_expect,gaze_heatmap_expect,gazeheatmap_expect,_ = self.behaviorcal(max_length,int(targetpack)-1,gazeexpect_each,logitsmax_each)
                gazerandom_length,correctrandom,search_per_random,refix_per_random,revisit_per_random,gaze_heatmap_random,gazeheatmap_random,_ = self.behaviorcal(max_length,int(targetpack)-1,gazerandom_each,logitsmax_each)
                gazesimilarity_length,correctsimilarity,search_per_similarity,refix_per_similarity,revisit_per_similarity,gaze_heatmap_similarity,gazeheatmap_similarity,_ = self.behaviorcal(max_length,int(targetpack)-1,gazesimilarity_each,logitsmax_each)
                gazesaliency_length,correctsaliency,search_per_saliency,refix_per_saliency,revisit_per_saliency,gaze_heatmap_saliency,gazeheatmap_saliency,_ = self.behaviorcal(max_length,int(targetpack)-1,gazesaliency_each,logitsmax_each)
                gazergb_length,correctrgb,search_per_rgb,refix_per_rgb,revisit_per_rgb,gaze_heatmap_rgb,gazeheatmap_rgb,_ = self.behaviorcal(max_length,int(targetpack)-1,gazergb_each,logitsmax_each)
                correct_expect[i,j] = correctexpect
                correct_random[i,j] = correctrandom
                correct_similarity[i,j] = correctsimilarity
                correct_saliency[i,j] = correctsaliency
                correct_rgb[i,j] = correctrgb
                
                search_expect_perc[i,j] = search_per_expect
                refix_expect_perc[i,j] = refix_per_expect
                revisit_expect_perc[i,j] = revisit_per_expect

                search_random_perc[i,j] = search_per_random
                refix_random_perc[i,j] = refix_per_random
                revisit_random_perc[i,j] = revisit_per_random

                search_similarity_perc[i,j] = search_per_similarity
                refix_similarity_perc[i,j] = refix_per_similarity
                revisit_similarity_perc[i,j] = revisit_per_similarity

                search_saliency_perc[i,j] = search_per_saliency
                refix_saliency_perc[i,j] = refix_per_saliency
                revisit_saliency_perc[i,j] = revisit_per_saliency

                search_rgb_perc[i,j] = search_per_rgb
                refix_rgb_perc[i,j] = refix_per_rgb
                revisit_rgb_perc[i,j] = revisit_per_rgb

                length_expect[i,j] = gazeexpect_length
                length_random[i,j] = gazerandom_length
                length_similarity[i,j] = gazesimilarity_length
                length_saliency[i,j] = gazesaliency_length
                length_rgb[i,j] = gazergb_length
                
                heatmap_similarity_overlaploss[i,j] = self.minoverlap(gazeheatmap_similarity,gazeheatmap_true)
                heatmap_expect_overlaploss[i,j] = self.minoverlap(gazeheatmap_expect,gazeheatmap_true)
                heatmap_expect_klloss[i,j] = self.loss_fn1(torch.from_numpy(gazeheatmap_expect).log(),torch.from_numpy(gazeheatmap_true))
                heatmap_random_overlaploss[i,j] = self.minoverlap(gazeheatmap_random,gazeheatmap_true)
                heatmap_random_klloss[i,j] = self.loss_fn1(torch.from_numpy(gazeheatmap_random).log(),torch.from_numpy(gazeheatmap_true))
                heatmap_similarity_klloss[i,j] = self.loss_fn1(torch.from_numpy(gazeheatmap_similarity).log(),torch.from_numpy(gazeheatmap_true))
                heatmap_saliency_overlaploss[i,j] = self.minoverlap(gazeheatmap_saliency,gazeheatmap_true)
                heatmap_saliency_klloss[i,j] = self.loss_fn1(torch.from_numpy(gazeheatmap_saliency).log(),torch.from_numpy(gazeheatmap_true))
                heatmap_rgb_overlaploss[i,j] = self.minoverlap(gazeheatmap_rgb,gazeheatmap_true)
                heatmap_rgb_klloss[i,j] = self.loss_fn1(torch.from_numpy(gazeheatmap_rgb).log(),torch.from_numpy(gazeheatmap_true))

            question_img = Image.open(img_dir + questionname + '.png')

            if questionname.startswith('Q1'):
                IMAGE_SIZE_1 = 449
                IMAGE_SIZE_2 = 152
                IMAGE_ROW = 2
                IMAGE_COLUMN = 11  

            elif questionname.startswith('Q2'):
                IMAGE_SIZE_1 = 295
                IMAGE_SIZE_2 = 186
                IMAGE_ROW = 3
                IMAGE_COLUMN = 9

            elif questionname.startswith('Q3'):
                IMAGE_SIZE_1 = 305
                IMAGE_SIZE_2 = 186
                IMAGE_ROW = 3
                IMAGE_COLUMN = 9
            if questionname.startswith('Q2'):
                question_cropped_range = (0, (1050-IMAGE_SIZE_1*IMAGE_ROW), IMAGE_COLUMN*IMAGE_SIZE_2,  (1050-IMAGE_SIZE_1*IMAGE_ROW)+IMAGE_ROW *IMAGE_SIZE_1)
                question_img_cropped =  question_img.crop(question_cropped_range)
                gazeheatmap_gt = pd.DataFrame(gazeheatmap_gt.reshape(3,9))
                gazeheatmap_max = pd.DataFrame(gazeheatmap_max.reshape(3,9))
                gazeheatmap_true = pd.DataFrame(gazeheatmap_true.reshape(3,9))
                saliency_result = np.array(saliency_result).reshape(3,9)
                resnet_result = np.array(resnet_result).reshape(3,9)
                rgb_result = np.array(rgb_result).reshape(3,9)
                ae_result = np.array(ae_result).reshape(3,9)
                resnet_result = pd.DataFrame(resnet_result)
                rgb_result = pd.DataFrame(rgb_result)
                ae_result = pd.DataFrame(ae_result)
                cnn_result = pd.DataFrame(cnn_result)
                self.heatmapplot(resnet_result,question_img_cropped,0.3,1,0.5,questionname,targetpack,' ','Heatmap_Resnet',save_path_1)
                self.heatmapplot(rgb_result,question_img_cropped,0.6,1,0.5,questionname,targetpack,' ','Heatmap_RGB',save_path_2)
                self.heatmapplot(ae_result,question_img_cropped,0.8,1,0.5,questionname,targetpack,' ','Heatmap_AE',save_path_3)
                self.heatmapplot(cnn_result,question_img_cropped,0.8,1,0.5,questionname,targetpack,' ','Heatmap_CNN',save_path_4)
                self.heatmapplot(gazeheatmap_gt,question_img_cropped,0,1,0.5,questionname,targetpack,' ','Heatmap_gt',save_path_5)
                self.heatmapplot(gazeheatmap_max,question_img_cropped,0,1,0.5,questionname,targetpack,' ','Heatmap_max',save_path_6)
                self.heatmapplot(gazeheatmap_true,question_img_cropped,0,1,0.5,questionname,targetpack,' ','Heatmap_true',save_path_7)

                for n in range(len(gaze_logits_gt)):
                    gaze_logits_gt_each = pd.DataFrame(gaze_logits_gt[n].reshape(3,9)) 
                    self.heatmapplot(gaze_logits_gt_each,question_img_cropped,0,1,0.8,questionname,targetpack,str(n),'Heatmap_logits_gt',save_path_8)
                    
                for m in range(len(gaze_logits_max)):
                    gaze_logits_max_each = pd.DataFrame(gaze_logits_max[m].reshape(3,9)) 
                    self.heatmapplot(gaze_logits_max_each,question_img_cropped,0,1,0.8,questionname,targetpack,str(m),'Heatmap_logits_max',save_path_9)
                    
                for l in range(len(gaze_heatmap_true)):
                    gaze_heatmap_true_each = pd.DataFrame(gaze_heatmap_true[l].reshape(3,9)) 
                    self.heatmapplot(gaze_heatmap_true_each,question_img_cropped,0,1,0.8,questionname,targetpack,str(l),'Heatmap_true_step',save_path_10)
        
        print(f'correct_gt_total:{correct_gt.sum()/int(self.data_length)}')
        print(f'correct_max_total:{correct_max.sum() / int(self.data_length)}')
        print(f'correct_true_total:{correct_true.sum() /int(self.data_length)}')
        print(f'correct_expect_total:{correct_expect.mean(axis=1).sum() /int(self.data_length)}')
        print(f'correct_random_total:{correct_random.mean(axis=1).sum() /int(self.data_length)}')
        print(f'correct_similarity_total:{correct_similarity.mean(axis=1).sum() /int(self.data_length)}')
        print(f'correct_saliency_total:{correct_saliency.mean(axis=1).sum() /int(self.data_length)}')
        print(f'correct_rgb_total:{correct_rgb.mean(axis=1).sum() /int(self.data_length)}')

        print(f'search_gt_perc_total:{search_gt_perc.mean()}')
        print(f'refix_gt_perc_total:{refix_gt_perc.mean()}')
        print(f'revisit_gt_perc_total:{revisit_gt_perc.mean()}')

        print(f'search_max_perc_total:{search_max_perc.mean()}')
        print(f'refix_max_perc_total:{refix_max_perc.mean()}')
        print(f'revisit_max_perc_total:{revisit_max_perc.mean()}')

        print(f'search_true_perc_total:{search_true_perc.mean()}')
        print(f'refix_true_perc_total:{refix_true_perc.mean()}')
        print(f'revisit_true_perc_total:{revisit_true_perc.mean()}')

        print(f'search_expect_perc_total:{search_expect_perc.mean()}')
        print(f'refix_expect_perc_total:{refix_expect_perc.mean()}')
        print(f'revisit_expect_perc_total:{revisit_expect_perc.mean()}')

        print(f'search_random_perc_total:{search_random_perc.mean()}')
        print(f'refix_random_perc_total:{refix_random_perc.mean()}')
        print(f'revisit_random_perc_total:{revisit_random_perc.mean()}')

        print(f'search_similarity_perc_total:{search_similarity_perc.mean()}')
        print(f'refix_similarity_perc_total:{refix_similarity_perc.mean()}')
        print(f'revisit_similarity_perc_total:{revisit_similarity_perc.mean()}')

        print(f'search_saliency_perc_total:{search_saliency_perc.mean()}')
        print(f'refix_saliency_perc_total:{refix_saliency_perc.mean()}')
        print(f'revisit_saliency_perc_total:{revisit_saliency_perc.mean()}')

        print(f'search_rgb_perc_total:{search_rgb_perc.mean()}')
        print(f'refix_rgb_perc_total:{refix_rgb_perc.mean()}')
        print(f'revisit_rgb_perc_total:{revisit_rgb_perc.mean()}')

        print(f'length_gt_total:{length_gt.mean()}')
        print(f'length_max_total:{length_max.mean()}')
        print(f'length_true_total:{length_true.mean()}')
        print(f'length_expect_total:{length_expect.mean()}')
        print(f'length_random_total:{length_random.mean()}')
        print(f'length_similarity_total:{length_similarity.mean()}')
        print(f'length_saliency_total:{length_saliency.mean()}')
        print(f'length_rgb_total:{length_rgb.mean()}')
        
        print(f'heatmap_gt_cosloss_total:{heatmap_gt_overlaploss.mean()}')
        print(f'heatmap_max_cosloss_total:{heatmap_max_overlaploss.mean()}')
        print(f'heatmap_expect_cosloss_total:{heatmap_expect_overlaploss.mean()}')
        print(f'heatmap_random_cosloss_total:{heatmap_random_overlaploss.mean()}')
        print(f'heatmap_similarity_cosloss_total:{heatmap_similarity_overlaploss.mean()}')
        print(f'heatmap_saliency_cosloss_total:{heatmap_saliency_overlaploss.mean()}')
        print(f'heatmap_rgb_cosloss_total:{heatmap_rgb_overlaploss.mean()}')
       
        print(f'heatmap_gt_klloss_total:{heatmap_gt_klloss.mean()}')
        print(f'heatmap_max_klloss_total:{heatmap_max_klloss.mean()}')
        print(f'heatmap_expect_klloss_total:{heatmap_expect_klloss.mean()}')
        print(f'heatmap_random_klloss_total:{heatmap_random_klloss.mean()}')
        print(f'heatmap_similarity_klloss_total:{heatmap_similarity_klloss.mean()}')
        print(f'heatmap_saliency_klloss_total:{heatmap_saliency_klloss.mean()}')
        print(f'heatmap_rgb_klloss_total:{heatmap_rgb_klloss.mean()}')
       
         
if __name__ == '__main__':
    HeatmapPlot= HeatmapPlot()
    HeatmapPlot.forward()

       
