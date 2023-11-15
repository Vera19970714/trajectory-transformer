from models import ResNetCOCO
import torchvision.transforms as T
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from sentence_transformers import SentenceTransformer
import PIL
import os
from os.path import join, isdir, isfile
import numpy as np
import argparse
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def image_data(dataset_path, question):
    if question == 1:
        src_path = join(dataset_path, 'Question/')
        target_path = join(dataset_path, 'Question_features/')
        resize_dim = (320 * 2, 512 * 2)  # 640, 1024

    elif question == 2:
        src_path = join(dataset_path, 'Target/')
        target_path = join(dataset_path, 'Target_features/')
        resize_dim = (150, 90)  # 640, 1024
    resize = T.Resize(resize_dim)
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #folders = [i for i in os.listdir(src_path) if isdir(join(src_path, i))]

    bbone = ResNetCOCO(device = device).to(device)
    #for folder in folders:
    if not (os.path.exists(target_path) and os.path.isdir(target_path)):
        os.mkdir(target_path)
    files =  [i for i in os.listdir(src_path) if isfile(join(src_path, i)) and i.endswith('.png')]
    for f in tqdm(files):
       PIL_image = PIL.Image.open(join(src_path, f)).convert('RGB') # size 1680, 1050; 186, 295
       if question == 1:
           tensor_image = normalize(resize(T.functional.to_tensor(PIL_image))).unsqueeze(0)
       elif question == 2:
           tensor_image = normalize(resize(T.functional.to_tensor(PIL_image))).unsqueeze(0)

       features = bbone(tensor_image).squeeze().detach().cpu().numpy()
       np.save(join(target_path, f.replace('png', 'npy')), features)
           
           
def text_data(dataset_path, device = 'cuda:0', lm_model = 'sentence-transformers/stsb-roberta-base-v2'):
    src_path = join(dataset_path, 'images/')
    tasks = [' '.join(i.split('_')) for i in os.listdir(src_path) if isdir(join(src_path, i))]

    lm = SentenceTransformer(lm_model, device=device).eval()
    embed_dict = {}
    for task in tasks:
       embed_dict[task] = lm.encode(task)
    with open(join(dataset_path,'embeddings.npy'), 'wb') as f:
        np.save(f, embed_dict, allow_pickle = True)
        f.close()
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Gazeformer Feature Extractor Utils', add_help=False)
    parser.add_argument('--img_path', default= '../trajectory-transformer/dataset/img/', type=str)
    #parser.add_argument('--cuda', default=0, type=int)
    args = parser.parse_args()
    #device = 'cpu'# torch.device('cuda:{}'.format(args.cuda))
    image_data(dataset_path = args.img_path, question=2)
    image_data(dataset_path=args.img_path, question=1)

