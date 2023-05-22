import os
import json
import nltk

class Model:
    def __init__(self):
        self.data_root = "data/zh-en"
        self.exp_dir = "exps_MEM"

        if not os.path.exists(os.path.join(self.exp_dir, "translate_table.json")):
            print("Error, you need the translation table to run the Maximum Entropy Model!!!")
        else:
            with open(os.path.join(self.exp_dir, "translate_table.json"), 'r', encoding='utf-8') as json_file:
                self.translate_table = json.load(json_file)

    def translate(self, source):
        # divide with nltk
        source_seg_list = nltk.word_tokenize(source)
        source_seg_list = [w.lower() for w in source_seg_list]
        
        # translate
        target_list = []
        for w in source_seg_list:
            if w in self.translate_table:
                target_list.append(self.translate_table[w][0][0])
        
        # form a complete sentence
        target_sentence = ""
        for w in target_list:
            target_sentence += w
        return target_sentence