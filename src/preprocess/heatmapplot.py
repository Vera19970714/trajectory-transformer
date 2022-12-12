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

dumbData = './dataset/cnnFeature.npy'
gazegt = './dataset/outputdata/gaze_gt_threedim.csv'
gazetrue = './dataset/outputdata/gaze_true_threedim.csv'
gazemax = './dataset/outputdata/gaze_max_threedim.csv'
logitsgt = './dataset/outputdata/logits_gt_threedim.csv'
logitsmax = './dataset/outputdata/logits_max_threedim.csv'

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

    def getSimCnnFeature(self,data):
        #specify the data
        #dumbData = np.zeros((80, 28, 256))
        tgt = data[0:1] #1, 256
        #tgt = np.tile(tgt, (27, 1))
        rest = data[1:] #27, 256
        sim = cosine_similarity(tgt, rest)
        return sim

    def behaviorcal(self,each_length,target,gaze,logits):
        gaze_search = 1
        gaze_refix = 0
        gaze_revisit = 0
        gaze_length = 0
        correct = 0
        gaze_heatmap = []
        gaze_logits = []
        gazeheatmap = np.zeros((1,27))
        
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
                gaze_length += 1
                gazelogits = logits[i,:27]
                gaze_logits.append(gazelogits)
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

        return gaze_length,correct,search_per,refix_per,revisit_per,gaze_heatmap,gazeheatmap,gaze_logits
        
            
    def heatmapplot(self,df,img,min,max,alpha_trans,questionname,targetpack,step_name,name,path):
        fig = plt.figure(figsize=(10, 4.3), dpi=120)
        # add alpha (transparency) to a colormap
        
        wd = matplotlib.cm.winter._segmentdata # only has r,g,b  
        wd['alpha'] =  ((0.0, 0.0, 0.3), 
                        (0.3, 0.3, 1.0),
                        (1.0, 1.0, 1.0))

        # modified colormap with changing alpha
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
        sum = 0
        correct_gt = 0
        correct_max = 0
        correct_true = 0

        search_gt_perc = 0
        refix_gt_perc = 0 
        revisit_gt_perc = 0 

        search_max_perc = 0
        refix_max_perc = 0 
        revisit_max_perc = 0 

        search_true_perc = 0
        refix_true_perc = 0 
        revisit_true_perc = 0 

        length_gt = 0
        length_max = 0
        length_true = 0

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
            sum += each_length
            gazegt_each = self.gazegt[(sum-each_length):sum,:]
            gazemax_each = self.gazemax[(sum-each_length):sum,:]
            gazetrue_each = self.gazetrue[(sum-each_length):sum,:]
            logitsgt_each = self.logitsgt[(sum-each_length):sum,:]
            logitsmax_each = self.logitsmax[(sum-each_length):sum,:]
            gazegt_length,correctgt,search_per_gt,refix_per_gt,revisit_per_gt,gaze_heatmap_gt,gazeheatmap_gt,gaze_logits_gt = self.behaviorcal(each_length,int(targetpack)-1,gazegt_each,logitsgt_each)
            gazemax_length,correctmax,search_per_max,refix_per_max,revisit_per_max,gaze_heatmap_max,gazeheatmap_max,gaze_logits_max = self.behaviorcal(each_length,int(targetpack)-1,gazemax_each,logitsmax_each)
            gazetrue_length,correcttrue,search_per_true,refix_per_true,revisit_per_true,gaze_heatmap_true,gazeheatmap_true,_ = self.behaviorcal(each_length,int(targetpack)-1,gazetrue_each,logitsgt_each)
            

            correct_gt += correctgt
            correct_max += correctmax
            correct_true += correcttrue
            search_gt_perc += search_per_gt
            refix_gt_perc += refix_per_gt
            revisit_gt_perc += revisit_per_gt 

            search_max_perc += search_per_max
            refix_max_perc += refix_per_max
            revisit_max_perc += revisit_per_max

            search_true_perc += search_per_true
            refix_true_perc += refix_per_true
            revisit_true_perc += revisit_per_true 

            length_gt += gazegt_length
            length_max += gazemax_length
            length_true += gazetrue_length
            
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
                self.heatmapplot(gazeheatmap_gt,question_img_cropped,0,3,0.5,questionname,targetpack,' ','Heatmap_gt',save_path_5)
                self.heatmapplot(gazeheatmap_max,question_img_cropped,0,3,0.5,questionname,targetpack,' ','Heatmap_max',save_path_6)
                self.heatmapplot(gazeheatmap_true,question_img_cropped,0,3,0.5,questionname,targetpack,' ','Heatmap_true',save_path_7)

                for n in range(len(gaze_logits_gt)):
                    gaze_logits_gt_each = pd.DataFrame(gaze_logits_gt[n].reshape(3,9)) 
                    self.heatmapplot(gaze_logits_gt_each,question_img_cropped,0,5,0.8,questionname,targetpack,str(n),'Heatmap_logits_gt',save_path_8)
                    
                for m in range(len(gaze_logits_max)):
                    gaze_logits_max_each = pd.DataFrame(gaze_logits_max[m].reshape(3,9)) 
                    self.heatmapplot(gaze_logits_max_each,question_img_cropped,0,5,0.8,questionname,targetpack,str(m),'Heatmap_logits_max',save_path_9)
                    
                for l in range(len(gaze_heatmap_true)):
                    gaze_heatmap_true_each = pd.DataFrame(gaze_heatmap_true[l].reshape(3,9)) 
                    self.heatmapplot(gaze_heatmap_true_each,question_img_cropped,0,1,0.8,questionname,targetpack,str(l),'Heatmap_true_step',save_path_10)
                
        correct_gt_total = correct_gt / int(self.data_length)
        correct_max_total = correct_max / int(self.data_length)
        correct_true_total = correct_true /int(self.data_length)

        search_gt_perc_total = search_gt_perc / int(self.data_length)
        refix_gt_perc_total = refix_gt_perc / int(self.data_length)
        revisit_gt_perc_total = revisit_gt_perc / int(self.data_length)

        search_max_perc_total = search_max_perc / int(self.data_length)
        refix_max_perc_total = refix_max_perc / int(self.data_length)
        revisit_max_perc_total = revisit_max_perc / int(self.data_length)

        search_true_perc_total = search_true_perc / int(self.data_length)
        refix_true_perc_total = refix_true_perc / int(self.data_length)
        revisit_true_perc_total = revisit_true_perc / int(self.data_length)
         
        length_gt_total = length_gt /int(self.data_length)
        length_max_total = length_max /int(self.data_length)
        length_true_total = length_true /int(self.data_length)

        print(f'correct_gt_total:{correct_gt_total}')
        print(f'correct_max_total:{correct_max_total}')
        print(f'correct_true_total:{correct_true_total}')

        print(f'search_gt_perc_total:{search_gt_perc_total}')
        print(f'refix_gt_perc_total:{refix_gt_perc_total}')
        print(f'revisit_gt_perc_total:{revisit_gt_perc_total}')

        print(f'search_max_perc_total:{search_max_perc_total}')
        print(f'refix_max_perc_total:{refix_max_perc_total}')
        print(f'revisit_max_perc_total:{revisit_max_perc_total}')

        print(f'search_true_perc_total:{search_true_perc_total}')
        print(f'refix_true_perc_total:{refix_true_perc_total}')
        print(f'revisit_true_perc_total:{revisit_true_perc_total}')

        print(f'length_gt_total:{length_gt_total}')
        print(f'length_max_total:{length_max_total}')
        print(f'length_true_total:{length_true_total}')

         
if __name__ == '__main__':
    HeatmapPlot= HeatmapPlot()
    HeatmapPlot.forward()

       
