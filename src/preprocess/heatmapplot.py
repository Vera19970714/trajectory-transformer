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


dumbData = './dataset/cnnFeature.npy'
datapath = './dataset/processdata/dataset_Q23_similarity_mousedel_time_val'
target_dir = './dataset/img/Target/'
img_dir = './dataset/img/Question/'

save_path_1 = './dataset/output/Plot_Heatmap/Resnet/'
save_path_2 = './dataset/output/Plot_Heatmap/Rgb/'
save_path_3 = './dataset/output/Plot_Heatmap/AE/'
save_path_4 = './dataset/output/Plot_Heatmap/CNN/'

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

    def getSimCnnFeature(self,data):
        #specify the data
        #dumbData = np.zeros((80, 28, 256))
        tgt = data[0:1] #1, 256
        #tgt = np.tile(tgt, (27, 1))
        rest = data[1:] #27, 256
        sim = cosine_similarity(tgt, rest)
        return sim
    
    def heatmapplot(self):
        for i in range(self.data_length):
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
                saliency_result = np.array(saliency_result).reshape(3,9)
                resnet_result = np.array(resnet_result).reshape(3,9)
                rgb_result = np.array(rgb_result).reshape(3,9)
                ae_result = np.array(ae_result).reshape(3,9)
                saliency_result_sum = saliency_result.sum()
                resnet_result_sum = resnet_result.sum()
                rgb_result_sum = rgb_result.sum()
                ae_result_sum = ae_result.sum()
                cnn_result_sum = cnn_result.sum()
                
                # saliency_result = saliency_result / saliency_result_sum
                resnet_result = pd.DataFrame(resnet_result)
                rgb_result = pd.DataFrame(rgb_result)
                ae_result = pd.DataFrame(ae_result)
                cnn_result = pd.DataFrame(cnn_result)

                fig1 = plt.figure(figsize=(10, 4.3), dpi=120)
                # add alpha (transparency) to a colormap
                
                wd = matplotlib.cm.winter._segmentdata # only has r,g,b  
                wd['alpha'] =  ((0.0, 0.0, 0.3), 
                                (0.3, 0.3, 1.0),
                                (1.0, 1.0, 1.0))

                # modified colormap with changing alpha
                al_winter = LinearSegmentedColormap('AlphaWinter', wd) 
                hmax1 = sns.heatmap(data=resnet_result,
                            vmin=0.3, 
                            vmax=1,
                            cmap=plt.get_cmap('Greens'),
                            alpha = 0.5, # whole heatmap is translucent
                            annot = True,
                            fmt = '.3f',
                            zorder = 2,
                            cbar = True,
                            mask = resnet_result < 0.025
                        )
                hmax1.imshow(question_img_cropped,
                            aspect = hmax1.get_aspect(),
                            extent = hmax1.get_xlim() + hmax1.get_ylim(),
                            zorder = 1) #put the map under the heatmap
               
                # plt.show()
                plt.title('Heatmap_Resnet' + '_' + str(questionname)+'_' + str(targetpack),fontsize='xx-large',fontweight='heavy') 
                plt.savefig(save_path_1+'Heatmap_Resnet'+ '_' + str(questionname) + '_' + str(targetpack)+'.jpg')
                
                fig2 = plt.figure(figsize=(10, 4.3), dpi=120)
                # add alpha (transparency) to a colormap
                
                wd = matplotlib.cm.winter._segmentdata # only has r,g,b  
                wd['alpha'] =  ((0.0, 0.0, 0.3), 
                                (0.3, 0.3, 1.0),
                                (1.0, 1.0, 1.0))

                # modified colormap with changing alpha
                al_winter = LinearSegmentedColormap('AlphaWinter', wd) 
                hmax2 = sns.heatmap(data=rgb_result,
                            vmin=0.6, 
                            vmax=1,
                            cmap=plt.get_cmap('Greens'),
                            alpha = 0.5, # whole heatmap is translucent
                            annot = True,
                            fmt = '.3f',
                            zorder = 2,
                            cbar = True,
                            mask = rgb_result < 0.025
                        )
                hmax2.imshow(question_img_cropped,
                            aspect = hmax2.get_aspect(),
                            extent = hmax2.get_xlim() + hmax2.get_ylim(),
                            zorder = 1) #put the map under the heatmap
               
                # plt.show()
                plt.title('Heatmap_RGB' + '_' + str(questionname)+'_' + str(targetpack),fontsize='xx-large',fontweight='heavy') 
                plt.savefig(save_path_2+'Heatmap_RGB'+ '_' + str(questionname) + '_' + str(targetpack)+'.jpg')
                
                fig3 = plt.figure(figsize=(10, 4.3), dpi=120)
                # add alpha (transparency) to a colormap
                
                wd = matplotlib.cm.winter._segmentdata # only has r,g,b  
                wd['alpha'] =  ((0.0, 0.0, 0.3), 
                                (0.3, 0.3, 1.0),
                                (1.0, 1.0, 1.0))

                # modified colormap with changing alpha
                al_winter = LinearSegmentedColormap('AlphaWinter', wd) 
                hmax3 = sns.heatmap(data=ae_result,
                            vmin=0.8, 
                            vmax=1,
                            cmap=plt.get_cmap('Greens'),
                            alpha = 0.5, # whole heatmap is translucent
                            annot = True,
                            fmt = '.3f',
                            zorder = 2,
                            cbar = True,
                            mask = ae_result < 0.025
                        )
                hmax3.imshow(question_img_cropped,
                            aspect = hmax3.get_aspect(),
                            extent = hmax3.get_xlim() + hmax3.get_ylim(),
                            zorder = 1) #put the map under the heatmap
               
                # plt.show()
                plt.title('Heatmap_AE' + '_' + str(questionname)+'_' + str(targetpack),fontsize='xx-large',fontweight='heavy') 
                plt.savefig(save_path_3+'Heatmap_AE'+ '_' + str(questionname) + '_' + str(targetpack)+'.jpg')
                
                fig4 = plt.figure(figsize=(10, 4.3), dpi=120)
                # add alpha (transparency) to a colormap
                
                wd = matplotlib.cm.winter._segmentdata # only has r,g,b  
                wd['alpha'] =  ((0.0, 0.0, 0.3), 
                                (0.3, 0.3, 1.0),
                                (1.0, 1.0, 1.0))

                # modified colormap with changing alpha
                al_winter = LinearSegmentedColormap('AlphaWinter', wd) 
                hmax4 = sns.heatmap(data=cnn_result,
                            vmin=0.8, 
                            vmax=1,
                            cmap=plt.get_cmap('Greens'),
                            alpha = 0.5, # whole heatmap is translucent
                            annot = True,
                            fmt = '.3f',
                            zorder = 2,
                            cbar = True,
                            mask = cnn_result < 0.025
                        )
                hmax4.imshow(question_img_cropped,
                            aspect = hmax4.get_aspect(),
                            extent = hmax4.get_xlim() + hmax4.get_ylim(),
                            zorder = 1) #put the map under the heatmap
               
                # plt.show()
                plt.title('Heatmap_CNN' + '_' + str(questionname)+'_' + str(targetpack),fontsize='xx-large',fontweight='heavy') 
                plt.savefig(save_path_4+'Heatmap_CNN'+ '_' + str(questionname) + '_' + str(targetpack)+'.jpg')
                
if __name__ == '__main__':
    HeatmapPlot= HeatmapPlot()
    HeatmapPlot.heatmapplot()

       
