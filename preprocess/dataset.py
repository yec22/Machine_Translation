import xml.etree.ElementTree as ET
from tqdm import tqdm
import os
import jieba
import nltk
import json

class Dataset:
    def __init__(self, type="train", year=2010):
        '''
        param:
            type(str): train / validate / test
            year(int): 2010-2015 (specify the test file)
        '''
        self.type = type
        self.data_root = "data/zh-en"

        print("loading data ...")
        if self.type == "train":
            self.load_train_data()
        elif self.type == "validate":
            self.load_validate_data()
        elif self.type == "test":
            self.load_test_data(year)
        
        self.exp_dir = "exps_MEM"
        if not os.path.exists(self.exp_dir):
            os.mkdir(self.exp_dir)
        nltk.download('punkt')
    
    def load_train_data(self):
        self.data = []

        corpus_en_file = open(os.path.join(self.data_root, "train.tags.zh-en.en"), encoding='utf-8')
        corpus_zh_file = open(os.path.join(self.data_root, "train.tags.zh-en.zh"), encoding='utf-8')

        corpus_en = corpus_en_file.readlines()
        corpus_zh = corpus_zh_file.readlines()
        
        for en, zh in zip(corpus_en, corpus_zh):
            en = en.strip()
            zh = zh.strip()
            if en[0] == '<': # meta-description, just discard it 
                continue

            # optional, remove the blank space in the chinese sentence
            zh = zh.replace(" ", "")
            data_sample = {}
            data_sample["en"] = en
            data_sample["zh"] = zh
            self.data.append(data_sample)

        corpus_en_file.close()
        corpus_zh_file.close()

    def load_validate_data(self):
        en_data = []
        zh_data = []
        self.data = []

        corpus_en_file = open(os.path.join(self.data_root, "IWSLT17.TED.dev2010.zh-en.en.xml"), encoding='utf-8')
        corpus_zh_file = open(os.path.join(self.data_root, "IWSLT17.TED.dev2010.zh-en.zh.xml"), encoding='utf-8')

        en_tree = ET.parse(corpus_en_file)
        en_root = en_tree.getroot()[0]
        zh_tree = ET.parse(corpus_zh_file)
        zh_root = zh_tree.getroot()[0]

        for doc in en_root.findall('doc'):
            for seg in doc.findall('seg'):
                en_data.append(seg.text.strip())
        
        for doc in zh_root.findall('doc'):
            for seg in doc.findall('seg'):
                zh_data.append(seg.text.strip())
        
        for en, zh in zip(en_data, zh_data):
            # optional, remove the blank space in the chinese sentence
            zh = zh.replace(" ", "")
            data_sample = {}
            data_sample["en"] = en
            data_sample["zh"] = zh
            self.data.append(data_sample)

        corpus_en_file.close()
        corpus_zh_file.close()

    def load_test_data(self, year):
        en_data = []
        zh_data = []
        self.data = []

        corpus_en_file = open(os.path.join(self.data_root, f"IWSLT17.TED.tst{year}.zh-en.en.xml"), encoding='utf-8')
        corpus_zh_file = open(os.path.join(self.data_root, f"IWSLT17.TED.tst{year}.zh-en.zh.xml"), encoding='utf-8')

        en_tree = ET.parse(corpus_en_file)
        en_root = en_tree.getroot()[0]
        zh_tree = ET.parse(corpus_zh_file)
        zh_root = zh_tree.getroot()[0]

        for doc in en_root.findall('doc'):
            for seg in doc.findall('seg'):
                en_data.append(seg.text.strip())
        
        for doc in zh_root.findall('doc'):
            for seg in doc.findall('seg'):
                zh_data.append(seg.text.strip())
        
        for en, zh in zip(en_data, zh_data):
            # optional, remove the blank space in the chinese sentence
            zh = zh.replace(" ", "")
            data_sample = {}
            data_sample["en"] = en
            data_sample["zh"] = zh
            self.data.append(data_sample)

        corpus_en_file.close()
        corpus_zh_file.close()
    
    def build_vocab(self):
        '''
        build English & Chinese vocabulary dictionary
        '''
        align_file = open(os.path.join(self.exp_dir, "text.en-zh"), 'w', encoding='utf-8')

        self.en_word2idx = {}
        self.en_idx2word = {}
        self.zh_word2idx = {}
        self.zh_idx2word = {}

        self.en_word_n = 0
        self.zh_word_n = 0

        corpus = self.get_all_item()
        for pair in tqdm(corpus):
            # English
            # divide with nltk
            self.en_word2idx['<unk>'] = {"idx": 0, "count": 0}
            self.en_idx2word[0] = {"word": '<unk>', "count": 0}

            en_seg_list = nltk.word_tokenize(pair['en'])
            for w in en_seg_list:
                w = w.lower() 
                if w not in self.en_word2idx: # first appear
                    self.en_word_n += 1
                    self.en_word2idx[w] = {"idx": self.en_word_n, "count": 1}
                    self.en_idx2word[self.en_word_n] = {"word": w, "count": 1}
                else: # add count number
                    self.en_word2idx[w]["count"] += 1
                    word_idx = self.en_word2idx[w]["idx"]
                    self.en_idx2word[word_idx]["count"] += 1
            # Chinese
            # divide with jieba
            zh_seg_list = list(jieba.cut(pair['zh'], cut_all=False))
            for w in zh_seg_list:
                if w not in self.zh_word2idx: # first appear
                    self.zh_word_n += 1
                    self.zh_word2idx[w] = {"idx": self.zh_word_n, "count": 1}
                    self.zh_idx2word[self.zh_word_n] = {"word": w, "count": 1}
                else: # add count number
                    self.zh_word2idx[w]["count"] += 1
                    word_idx = self.zh_word2idx[w]["idx"]
                    self.zh_idx2word[word_idx]["count"] += 1
            
            write_list = [w.lower() + ' ' for w in en_seg_list] + [' ', '|||', ' '] +  [w + ' ' for w in zh_seg_list]
            align_file.writelines(write_list)
            align_file.write('\n')

        align_file.close()
        
        # record
        with open(os.path.join(self.exp_dir, "en_word2idx.json"), 'w', encoding='utf8') as json_file:
            json.dump(self.en_word2idx, json_file, ensure_ascii=False)
        
        with open(os.path.join(self.exp_dir, "en_idx2word.json"), 'w', encoding='utf8') as json_file:
            json.dump(self.en_idx2word, json_file, ensure_ascii=False)
        
        with open(os.path.join(self.exp_dir, "zh_word2idx.json"), 'w', encoding='utf8') as json_file:
            json.dump(self.zh_word2idx, json_file, ensure_ascii=False)
        
        with open(os.path.join(self.exp_dir, "zh_idx2word.json"), 'w', encoding='utf8') as json_file:
            json.dump(self.zh_idx2word, json_file, ensure_ascii=False)
    
    def build_translate_table(self):
        align_file = open(os.path.join(self.exp_dir, "forward.align"), encoding='utf-8')
        align_corpus = align_file.readlines()
        corpus = self.get_all_item()

        # calculate translate table
        self.translate_table = {}
        for i in tqdm(range(len(align_corpus))):
            sentence = align_corpus[i].strip().split(' ')
            en_seg_list = nltk.word_tokenize(corpus[i]['en'])
            en_seg_list = [w.lower() for w in en_seg_list]
            zh_seg_list = list(jieba.cut(corpus[i]['zh'], cut_all=False))

            for w in sentence:
                if w == '':
                    continue
                en_idx = int(w.split('-')[0])
                zh_idx = int(w.split('-')[1])
                if en_seg_list[en_idx] not in self.translate_table:
                    # first appear
                    self.translate_table[en_seg_list[en_idx]] = {}
                if zh_seg_list[zh_idx] not in self.translate_table[en_seg_list[en_idx]]:
                    # new translate
                    self.translate_table[en_seg_list[en_idx]][zh_seg_list[zh_idx]] = 1
                else:
                    # add count number
                    self.translate_table[en_seg_list[en_idx]][zh_seg_list[zh_idx]] += 1

        # clean & reorder
        for key in self.translate_table:
            self.translate_table[key] = sorted(self.translate_table[key].items(), key=lambda x: x[1], reverse=True)
            if len(self.translate_table[key]) > 10:
                self.translate_table[key] = self.translate_table[key][:10]
        # record
        with open(os.path.join(self.exp_dir, "translate_table.json"), 'w', encoding='utf8') as json_file:
            json.dump(self.translate_table, json_file, ensure_ascii=False)
    
    def get_data_size(self):
        '''
        get the total number of training pairs
        '''
        return len(self.data)
    
    def get_all_item(self):
        '''
        get all of the training pairs
        return:
            [{'en': 'xxx', 'zh': 'yyy'},
             {'en': 'aaa', 'zh': 'bbb'},
             ...
             {'en': 'iii', 'zh': 'jjj'}]
        '''
        return self.data
    
    def get_item(self, i):
        '''
        get one of the training pair
        param:
            i(int): index of training data
        return:
            {'en': 'xxx', 'zh': 'yyy'}
        '''
        return self.data[i]
