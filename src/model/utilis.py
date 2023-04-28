from math import log
import numpy as np
import torch
import torch.nn.functional as F

class Sampler():
    def beam_search_decoder(self,data, k):
        sequences = [[list(), 1.0]]

        for row in data:
            all_candidates = list()

            for i in range(len(sequences)):
                seq, score = sequences[i]
                for j in range(len(row)):
                    candidate = [seq + [j], score * -log(row[j])]
                    all_candidates.append(candidate)
            ordered = sorted(all_candidates, key=lambda tup: tup[1])
            sequences = ordered[:k]
        return sequences

    def greedy_decoder(self,data):
        return [np.argmax(s) for s in data]


    def top_k_top_p_decoder(self,logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
          Args:
              logits: logits distribution shape (batch size, vocabulary size)
              if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
              if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                 Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
         From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # print(cumulative_probs)
            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits

if __name__ == '__main__':

    data = [[0.9, 0.01, 0.05,0.04],
            [0.9, 0.02, 0.03,0.02],
            [0.25, 0.35, 0.2,0.2]]
    data = np.array(data)
    data2 = torch.Tensor(data)
    Sampler = Sampler()
    print("****use beam search decoder****")
    print(Sampler.beam_search_decoder(data, 3))

    print("****use greedy decoder****")
    print(Sampler.greedy_decoder(data))

    