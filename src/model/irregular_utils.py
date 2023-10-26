import torch
import numpy as np
import torch.nn.functional as F
import pickle
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PAD_IDX_testing = 43 + 1
BOS_IDX_testing = 43 + 2
EOS_IDX_testing = 43
iter = 100
with open("./dataset/processdata/mapping", "rb") as fp:
    mapping = pickle.load(fp)

def generate_square_subsequent_mask_irregular(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask_irregular(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask_irregular(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = None
    tgt_padding_mask = None
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def processData3d_irregular(src_pos, src_img, tgt_pos, tgt_img, type):
    tgt_input = tgt_pos[:-1, :]
    tgt_img = tgt_img[:, :-1, :, :, :]
    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask_irregular(src_pos, tgt_input)

    src_pos_2d, tgt_input_2d = generate3DInput_irregular(tgt_input, src_pos, type)

    return src_pos_2d, tgt_input_2d, src_img, tgt_img, src_mask, tgt_mask, \
           src_padding_mask, tgt_padding_mask, src_padding_mask

def generate3DInput_irregular(tgt_input, src_pos):
    tgt_input_2d = tgt_input.permute(0, 2, 1)
    src_pos_2d = src_pos.permute(0, 2, 1)
    tgt_target = torch.zeros(tgt_input_2d.size()[0], tgt_input_2d.size()[1], 1).to(DEVICE)
    tgt_input_2d = torch.concat((tgt_input_2d, tgt_target), dim=2)
    src_target = torch.zeros(src_pos_2d.size()[0], src_pos_2d.size()[1], 1).to(DEVICE)
    src_pos_2d = torch.concat((src_pos_2d, src_target), dim=2)

    # changed to three dimension
    batch = 1
    src_pos_2d[-1, :, 2] = 1  # the last one is target
    for i in range(batch):
        Index = src_pos[-1, :, i]
        ind1 = torch.where(tgt_input[:, 0, i] == Index[0])[0].tolist()
        ind2 = torch.where(tgt_input[:, 1, i] == Index[1])[0].tolist()
        for i1 in ind1:
            if i1 in ind2:
                tgt_input_2d[i1, 0, 2] = 1
    return src_pos_2d, tgt_input_2d



def generate_one_scanpath_irregular(tgt_pos, tgt_img, src_pos, src_img, new_src_img, tgt_1d_pos, getMaxProb, model, max_length=16):
    loss_fn = torch.nn.CrossEntropyLoss()
    length = tgt_pos.size(0)
    loss = 0
    LOSS = torch.zeros((length - 1, 1)) - 1
    GAZE = torch.zeros((max_length, 1)) - 1

    for i in range(1, max_length + 1):
        if i == 1:
            tgt_input = tgt_pos[:i]
            tgt_img_input = tgt_img[:i]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask_irregular(src_pos, tgt_input)
            src_pos_2d, tgt_input_2d = generate3DInput_irregular(tgt_input, src_pos)

            logits = model(src_pos_2d.float(), tgt_input_2d.float(),  # src_pos, tgt_input,
                                src_img, tgt_img_input,
                                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask,
                           patch_in_batch=False)
            # the first token cannot be end token
            logits = logits[:, :, :-2]  # discard padding prob
            if getMaxProb:
                _, predicted = torch.max(logits[-1, :, :], 1)
            else:
                logits_new = F.softmax(logits[-1, :, :].view(-1), dim=0)
                predicted = torch.multinomial(logits_new, 1, replacement=True)
            if i < length:
                tgt_out = tgt_1d_pos[i, :]
                LOSS[i - 1][0] = loss_fn(logits[-1, :, :].reshape(-1, logits[-1, :, :].shape[-1]),
                                              tgt_out.reshape(-1).long())
                loss += loss_fn(logits[-1, :, :].reshape(-1, logits[-1, :, :].shape[-1]),
                                     tgt_out.reshape(-1).long())

            GAZE[i - 1][0] = predicted
            # LOGITS[i-1,:] = self.norm(logits[-1,:,:]).reshape(1,-1)

            next_tgt_img_input = tgt_img_input + [new_src_img[predicted]]
            new_2d_coord = torch.from_numpy(mapping[predicted]).float().to(DEVICE).view(-1, 2, 1)
            next_tgt_input = torch.cat((tgt_input, new_2d_coord), dim=0)
        else:
            tgt_input = next_tgt_input
            tgt_img_input = next_tgt_img_input
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask_irregular(src_pos, tgt_input)
            src_pos_2d, tgt_input_2d = generate3DInput_irregular(tgt_input, src_pos)
            logits = model(src_pos_2d.float(), tgt_input_2d.float(),  # src_pos, tgt_input,
                                src_img, tgt_img_input,
                                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask,
                           patch_in_batch=False)
            logits = logits[:, :, :-1]  # discard padding prob
            if getMaxProb:
                _, predicted = torch.max(logits[-1, :, :], 1)
            else:
                logits_new = F.softmax(logits[-1, :, :].view(-1), dim=0)
                predicted = torch.multinomial(logits_new, 1, replacement=True)
            if i < length:
                tgt_out = tgt_1d_pos[i, :]
                LOSS[i - 1][0] = loss_fn(logits[-1, :, :].reshape(-1, logits[-1, :, :].shape[-1]),
                                              tgt_out.reshape(-1).long())
                loss += loss_fn(logits[-1, :, :].reshape(-1, logits[-1, :, :].shape[-1]),
                                     tgt_out.reshape(-1).long())
            GAZE[i - 1][0] = predicted
            if EOS_IDX_testing in GAZE[:, 0] and i >= length:
                break
            # LOGITS[i-1,:] = self.norm(logits[-1,:,:]).reshape(1,-1)
            next_tgt_img_input = tgt_img_input + [new_src_img[predicted]]
            new_2d_coord = torch.from_numpy(mapping[predicted]).float().to(DEVICE).view(-1, 2, 1)
            next_tgt_input = torch.cat((tgt_input, new_2d_coord), dim=0)
    loss = loss / (length - 1)
    return loss, LOSS, GAZE  # ,LOGITS

def test_max_irregular(src_pos, src_img, tgt_pos, tgt_img, tgt_1d_pos, model):
    #tgt_input = tgt_pos[:-1, :]
    tgt_img = tgt_img[0][:-1]

    new_src_img = src_img[0][:-1] + [torch.from_numpy(np.ones((1, 128, 50, 3))).float().to(DEVICE)] * 3

    loss, LOSS, GAZE = generate_one_scanpath_irregular(tgt_pos, tgt_img, src_pos, src_img[0], new_src_img, tgt_1d_pos, True, model)
    if EOS_IDX_testing in GAZE:
        endIndex = torch.where(GAZE == EOS_IDX_testing)[0][0]
        GAZE = GAZE[:endIndex]
        # LOGITS = LOGITS[:endIndex]
    return loss, LOSS, GAZE


def test_expect_irregular(src_pos, src_img, tgt_pos, tgt_img, tgt_1d_pos, model):
    #tgt_input = tgt_pos[:-1, :]
    src_img = src_img[0]
    tgt_img = tgt_img[0][:-1]
    length = tgt_pos.size(0)
    loss = 0
    max_length = 16
    #new_src_img = src_img[0][:-1] + [(np.ones((128, 50, 3)), np.array([2, 5]))] + \
    #              [(np.ones((128, 50, 3)), np.array([-1, -1]))] + [(np.ones((128, 50, 3)), np.array([2, 5]))]
    new_src_img = src_img[:-1] + [torch.from_numpy(np.ones((1, 128, 50, 3))).float().to(DEVICE)] * 3
    GAZE = torch.zeros((max_length, iter))-1
    for n in range(iter):
        loss_per, _, GAZE_per = generate_one_scanpath_irregular(tgt_pos, tgt_img, src_pos, src_img, new_src_img, tgt_1d_pos, False, model)
        GAZE[:, n:(n+1)] = GAZE_per
        loss += loss_per / (length-1)
    loss= loss / iter
    GAZE_ALL = []
    for i in range(iter):
        if EOS_IDX_testing in GAZE[:,i]:
            j = torch.where(GAZE[:,i]==EOS_IDX_testing)[0][0]
            GAZE_ALL.append(GAZE[:j, i])
        else:
            GAZE_ALL.append(GAZE[:,i])
    return loss,GAZE_ALL

def test_gt_irregular(src_pos, src_img, tgt_pos, tgt_img, tgt_1d_pos, model):
    loss_fn = torch.nn.CrossEntropyLoss()
    tgt_input = tgt_pos[:-1]
    tgt_img_input = tgt_img[0][:-1]
    src_img = src_img[0]
    # src: 15, b; tgt_input: 14, b; src_msk: 15, 15; tgt_msk: 13, 13; tgt_padding_msk: 2, 13; src_padding_msk: 2, 15
    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask_irregular(src_pos, tgt_input)
    src_pos_2d, tgt_input_2d = generate3DInput_irregular(tgt_input, src_pos)

    logits = model(src_pos_2d.float(), tgt_input_2d.float(),  # src_pos, tgt_input,
                        src_img, tgt_img_input,
                        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask,
                   patch_in_batch=False)
    tgt_out = tgt_1d_pos[1:, :]
    loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
    _, predicted = torch.max(logits, 2)
    #LOGITS_tf=soft(logits).squeeze(1)
    print(predicted.view(-1))
    return loss, predicted[:-1], tgt_out[:-1] #,LOGITS_tf[:-1]


def test_one_dataset_irregular(batch, model):
    src_pos, src_img, tgt_pos, tgt_img, tgt_1d_pos = batch
    src_pos = src_pos.to(DEVICE)
    tgt_pos = tgt_pos.to(DEVICE)
    tgt_1d_pos = tgt_1d_pos.to(DEVICE)
    loss_gt, GAZE_tf, GAZE_gt = test_gt_irregular(src_pos, src_img, tgt_pos, tgt_img, tgt_1d_pos, model)
    loss_max, LOSS, GAZE = test_max_irregular(src_pos, src_img, tgt_pos, tgt_img, tgt_1d_pos, model)
    loss_expect, GAZE_expect = test_expect_irregular(src_pos, src_img, tgt_pos, tgt_img, tgt_1d_pos, model)
    return loss_max, loss_expect, loss_gt, GAZE, GAZE_expect, GAZE_gt