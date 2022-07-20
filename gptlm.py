# From https://github.com/thunlp/ONION/blob/main/experiments/gptlm.py

import math
import numpy as np
from dataset import MAX_SENTENCE_LENGTH

class GPT2LM:
    def __init__(self, device=None, little=False):
        """
        :param bool use_tf: If true, uses tensorflow GPT-2 model.
        :Package Requirements:
            * **torch** (if use_tf = False)
            * **transformers**
        Language Models are Unsupervised Multitask Learners.
        `[pdf] <https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf>`__
        `[code] <https://github.com/openai/gpt-2>`__
        """
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import transformers
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

        self.lm = transformers.AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M", from_tf=False)
        self.lm.to(device)
        self.device = device

        
    def __call__(self, sent):
        """
        :param str sent: A sentence.
        :return: Fluency (ppl).
        :rtype: float
        """
        ipt = self.tokenizer(sent, return_tensors="pt", verbose=False, max_length=MAX_SENTENCE_LENGTH, truncation=True)
        # print(ipt)
        # print(ipt.input_ids)
        try:
            ppl = math.exp(self.lm(input_ids=ipt['input_ids'].to(self.device),
                             attention_mask=ipt['attention_mask'].to(self.device),
                             labels=ipt.input_ids.to(self.device))[0])
        except RuntimeError:
            ppl = np.nan
        return ppl

