import os
import json
import nltk
from queue import PriorityQueue
from copy import *
import math

class Model:
    def __init__(self):
        self.data_root = "data/zh-en"
        self.exp_dir = "exps_MEM"

        if not os.path.exists(os.path.join(self.exp_dir, "translate_table.json")):
            print("Error, you need the translation table to run the Maximum Entropy Model!!!")
        else:
            with open(os.path.join(self.exp_dir, "translate_table.json"), 'r', encoding='utf-8') as json_file:
                self.translate_table = json.load(json_file)

        if not os.path.exists(os.path.join(self.exp_dir, "language_model.json")):
            print("Error, you need the language model to run the Maximum Entropy Model!!!")
        else:
            with open(os.path.join(self.exp_dir, "language_model.json"), 'r', encoding='utf-8') as json_file:
                self.language_model = json.load(json_file)

        self.weight_trans = 1.0
        self.weight_lang = 0.1
        self.weight_distort = 1.0
        self.alpha = 0.5
        
        nltk.download('punkt')
    
    def cal_language_model(self, sentence):
        lang_p = 1e-6
        if len(sentence) == 1:
            if sentence[0] in self.language_model["start_word"]:
                lang_p = self.language_model["start_word"][sentence[0]]
        else:
            word1 = sentence[-2]
            word2 = sentence[-1]
            if (word1 in self.language_model["2-gram"]) and (word2 in self.language_model["2-gram"][word1]):
                lang_p = self.language_model["2-gram"][word1][word2]
        
        return math.log(lang_p)

    def translate(self, source):
        # divide with nltk
        source_seg_list = nltk.word_tokenize(source)
        source_seg_list = [w.lower() for w in source_seg_list]
        
        # translate
        stack_size = 30
        top_k = 3
        target_list = []

        beam_stack = [PriorityQueue() for _ in range(len(source_seg_list) + 1)]
        inital = {"vis": [False for _ in range(len(source_seg_list))],
                  "sentence": [], 'pos': -1,
                  "trans_cost": 0, "lang_cost": 0}
        beam_stack[0].put((0, inital))

        # beam search
        for i in range(1, len(source_seg_list) + 1):
            record = []
            while (not beam_stack[i-1].empty()):
                # pop out the parent
                parent = beam_stack[i-1].get()[1]

                # generate new candidate
                for j in range(len(source_seg_list)):
                    if parent["vis"][j] == False:
                        en_word = source_seg_list[j]

                        if en_word not in self.translate_table:
                            zh_candidate = [('', 1e-6)]
                        else:
                            zh_candidate = self.translate_table[en_word][:top_k]
                        for zh in zh_candidate:
                            zh_word = zh[0]
                            trans_p = zh[1] 

                            candidate = {}
                            candidate["vis"] = deepcopy(parent["vis"])
                            candidate["vis"][j] = True
                            candidate["sentence"] = deepcopy(parent["sentence"])
                            candidate["sentence"].append(zh_word)
                            candidate["pos"] = j
                            candidate["trans_cost"] = parent["trans_cost"] + math.log(trans_p)
                            candidate["lang_cost"] = parent["lang_cost"] + self.cal_language_model(candidate["sentence"])
                            reorder_cost = math.pow(self.alpha, abs(parent["pos"] - candidate["pos"] + 1))
                            reorder_cost = math.log(reorder_cost)

                            candidate_cost = self.weight_trans * candidate["trans_cost"] +\
                                             self.weight_lang * candidate["lang_cost"] +\
                                             self.weight_distort * reorder_cost

                            if candidate_cost in record:
                                continue
                            else:
                                record.append(candidate_cost)

                            # update the candidate stack
                            if beam_stack[i].qsize() >= stack_size:
                                min_candidate = beam_stack[i].get()
                                if min_candidate[0] < candidate_cost:
                                    beam_stack[i].put((candidate_cost, candidate))
                                else:
                                    beam_stack[i].put(min_candidate)
                            else:
                                beam_stack[i].put((candidate_cost, candidate))

        last_stack = beam_stack[-1]
        while (not last_stack.empty()):
            data = last_stack.get()
            target_list = data[1]["sentence"]

        # form a complete sentence
        target_sentence = ""
        for w in target_list:
            target_sentence += w
        return target_sentence