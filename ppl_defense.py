from tqdm import tqdm
import numpy as np
import torch

PPL_THRESHOLD = 0

def get_ppl(sent, LM):
    words = sent.split(' ')
    sent_ppl = LM(sent)
    ppl_diffs = []
    for pos in range(len(words)):
        filtered_sent = ' '.join(words[:pos] + words[pos+1:])
        ppl_diffs.append(sent_ppl-LM(filtered_sent))
    return ppl_diffs

def process_ppls(ppl_diffs):
    # Remove potential trigger word for one sentence
    if (np.array(ppl_diffs) > PPL_THRESHOLD).astype(int).sum() <= 0:
        return -1 # no trigger to remove
    return np.argmax(ppl_diffs)

def remove_trigger(sent, pos):
    if pos == -1:
        return sent
    words = sent.split(' ')
    return ' '.join(words[:pos] + words[pos+1:])

def process_poisoned_data(sent, LM):
    ppl_diffs = get_ppl(sent, LM)
    pos_to_remove = process_ppls(ppl_diffs)
    return remove_trigger(sent, pos_to_remove)

