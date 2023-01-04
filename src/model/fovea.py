import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image

def test():
    plt.rcParams["savefig.bbox"] = 'tight'
    orig_img = Image.open('../dataset/img/Target/T1_2.png').convert('RGB')
    orig_img = torch.from_numpy(np.array(orig_img)).permute(2,0,1)
    # torch.manual_seed(0)

    def plot(imgs, with_orig=True, row_title=None, **imshow_kwargs):
        if not isinstance(imgs[0], list):
            # Make a 2d grid even if there's just 1 row
            imgs = [imgs]

        num_rows = len(imgs)
        num_cols = len(imgs[0]) + with_orig
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
        for row_idx, row in enumerate(imgs):
            row = [orig_img] + row if with_orig else row
            for col_idx, img in enumerate(row):
                ax = axs[row_idx, col_idx]
                ax.imshow(img.permute(1,2,0), **imshow_kwargs)
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        if with_orig:
            axs[0, 0].set(title='Original image')
            axs[0, 0].title.set_size(8)
        if row_title is not None:
            for row_idx in range(num_rows):
                axs[row_idx, 0].set(ylabel=row_title[row_idx])

        plt.tight_layout()
        plt.show()

    blurrer = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
    blurred_imgs = [blurrer(orig_img) for _ in range(1)]
    plot(blurred_imgs)


def testGaussian(src_img):
    src_img2 = src_img.permute(0,1,4,2,3)
    src_img_final = src_img2.reshape(-1, 150, 93)
    img = src_img[0][0]
    img2 = src_img[0][1]
    ori_img = img.numpy()
    cv2.imshow('ori', ori_img)
    ori_img2 = img2.numpy()
    cv2.imshow('ori2', ori_img2)
    blurrer = T.GaussianBlur(kernel_size=(5, 5))
    blurSource1 = blurrer(src_img_final)
    blurSource1 = blurSource1.reshape(src_img2.size())
    bi1 = blurSource1[0][0].permute(1,2,0).numpy()
    bi2 = blurSource1[0][1].permute(1,2,0).numpy()
    cv2.imshow('blur1', bi2)
    cv2.imshow('blur2', bi1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def GaussianBlur(src_img):
    src_img2 = src_img.permute(0,1,4,2,3)
    src_img_final = src_img2.reshape(-1, 150, 93)
    blurrer = T.GaussianBlur(kernel_size=(5, 5))
    blurSource1 = blurrer(src_img_final)
    blurSource1 = blurSource1.reshape(src_img2.size()).reshape(src_img.size())
    return blurSource1


def fovea_tf(src_img, tgt):
    '''
    gradually unveil the image
    :param src_img: bs,28,150,93,3
    :param tgt: len,bs
    :return: blurred image (len-1,bs,28,150,93,3)
    '''
    #todo: print/check output
    blur_img = GaussianBlur(src_img) #bs,28,150,93,3
    output = []
    bs = tgt.size()[1]
    for i in range(tgt.size()[0]-1):
        if i == 0:
            to_be_process = blur_img #bs,28,150,93,3
        else:
            to_be_process = output[-1]
            tokens = [torch.arange(bs), tgt[i]] #bs
            valid_token_ind = torch.where(tokens[1] < 27)[0]
            valid_token = [tokens[0][valid_token_ind], tokens[1][valid_token_ind]]
            to_be_process[valid_token] = src_img[valid_token]
        output.append(to_be_process)
    output = torch.stack(output)
    return output


if __name__ == '__main__':
    #test()
    src_img = torch.zeros((20,28,150,93,3))
    testGaussian(src_img)