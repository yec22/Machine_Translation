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
        self.exp_dir_hmm = "exps_HMM"
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
        self.support = {}
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
                if en_seg_list[en_idx] not in self.support:
                    # first appear
                    self.support[en_seg_list[en_idx]] = {}
                if zh_seg_list[zh_idx] not in self.support[en_seg_list[en_idx]]:
                    # new translate
                    self.support[en_seg_list[en_idx]][zh_seg_list[zh_idx]] = 1
                else:
                    # add count number
                    self.support[en_seg_list[en_idx]][zh_seg_list[zh_idx]] += 1

        # clean & reorder
        for key in self.support:
            self.support[key] = sorted(self.support[key].items(), key=lambda x: x[1], reverse=True)
            if len(self.support[key]) > 10:
                self.support[key] = self.support[key][:10]
            total = 0
            for w in self.support[key]:
                total += w[1]
            self.support[key] = [[w[0], w[1] / total] for w in self.support[key]]
        # record
        with open(os.path.join(self.exp_dir, "translate_table.json"), 'w', encoding='utf8') as json_file:
            json.dump(self.support, json_file, ensure_ascii=False)

    def build_language_model(self):
        language_model = {"start_word": {}, "2-gram": {}}
        corpus = self.get_all_item()
        start_n = 0
        for pair in tqdm(corpus):
            # Chinese, divide with jieba
            zh_seg_list = list(jieba.cut(pair['zh'], cut_all=False))
            # fist word
            start_n += 1
            first_word = zh_seg_list[0]
            if first_word not in language_model["start_word"]:
                language_model["start_word"][first_word] = 1
            else:
                language_model["start_word"][first_word] += 1
            # 2-gram
            for i in range(len(zh_seg_list) - 1):
                gram_1 = zh_seg_list[i]
                gram_2 = zh_seg_list[i+1]
                if gram_1 not in language_model["2-gram"]:
                    language_model["2-gram"][gram_1] = {gram_2: 1}
                else:
                    if gram_2 not in language_model["2-gram"][gram_1]:
                        language_model["2-gram"][gram_1][gram_2] = 1
                    else:
                        language_model["2-gram"][gram_1][gram_2] += 1
                
        # normalize to probability
        for key in language_model["start_word"]:
            language_model["start_word"][key] = language_model["start_word"][key] / start_n
        for key in language_model["2-gram"]:
            total = 0
            for w in language_model["2-gram"][key]:
                total += language_model["2-gram"][key][w]
            for w in language_model["2-gram"][key]:
                language_model["2-gram"][key][w] = language_model["2-gram"][key][w] / total

        # record
        with open(os.path.join(self.exp_dir, "language_model.json"), 'w', encoding='utf8') as json_file:
            json.dump(language_model, json_file, ensure_ascii=False)

    def generate_HMM_PI(self):
        HMM_PI = {}
        corpus = self.get_all_item()
        total = 0
        for pair in tqdm(corpus):
            zh_seg_list = list(jieba.cut(pair['zh'], cut_all=False))
            total += 1
            pi_i = zh_seg_list[0]
            if pi_i not in HMM_PI:
                HMM_PI[pi_i] = 1.0
            else:
                HMM_PI[pi_i] += 1.0

        for key in HMM_PI:
            HMM_PI[key] = HMM_PI[key] / total

        with open(os.path.join(self.exp_dir_hmm, "HMM_PI.json"), 'w', encoding='utf8') as json_file:
            json.dump(HMM_PI, json_file, ensure_ascii=False)

    def generate_HMM_A(self):
        HMM_A = {}
        corpus = self.get_all_item()
        for pair in tqdm(corpus):
            zh_seg_list = list(jieba.cut(pair['zh'], cut_all=False))
            for i in range(len(zh_seg_list) - 1):
                w1 = zh_seg_list[i]
                w2 = zh_seg_list[i + 1]
                if w1 not in HMM_A:
                    HMM_A[w1] = {w2: 1.0}
                else:
                    if w2 not in HMM_A[w1]:
                        HMM_A[w1][w2] = 1.0
                    else:
                        HMM_A[w1][w2] += 1.0

        for key in HMM_A:
            total = 0
            for w in HMM_A[key]:
                total += HMM_A[key][w]
            for w in HMM_A[key]:
                HMM_A[key][w] = HMM_A[key][w] / total

        with open(os.path.join(self.exp_dir_hmm, "HMM_A.json"), 'w', encoding='utf8') as json_file:
            json.dump(HMM_A, json_file, ensure_ascii=False)

    def generate_HMM_B(self):
        align_file = open(os.path.join(self.exp_dir, "forward.align"), encoding='utf-8')
        align_corpus = align_file.readlines()
        corpus = self.get_all_item()

        HMM_B = {}
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
                zh = zh_seg_list[zh_idx]
                en = en_seg_list[en_idx]
                if zh not in HMM_B:
                    HMM_B[zh] = {}
                if en not in HMM_B[zh]:
                    HMM_B[zh][en] = 1
                else:
                    HMM_B[zh][en] += 1

        for key in HMM_B:
            HMM_B[key] = sorted(HMM_B[key].items(), key=lambda x: x[1], reverse=True)
            if len(HMM_B[key]) > 10:
                HMM_B[key] = HMM_B[key][:10]
            total = 0
            for w in HMM_B[key]:
                total += w[1]
            HMM_B[key] = {w[0]: (w[1] / total) for w in HMM_B[key]}
        with open(os.path.join(self.exp_dir_hmm, "HMM_B.json"), 'w', encoding='utf8') as json_file:
            json.dump(HMM_B, json_file, ensure_ascii=False)

    def generate_map(self):
        align_file = open(os.path.join(self.exp_dir, "forward.align"), encoding='utf-8')
        align_corpus = align_file.readlines()
        corpus = self.get_all_item()

        # calculate translate table
        self.support = {}
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
                if en_seg_list[en_idx] not in self.support:
                    # first appear
                    self.support[en_seg_list[en_idx]] = {}
                if zh_seg_list[zh_idx] not in self.support[en_seg_list[en_idx]]:
                    # new translate
                    self.support[en_seg_list[en_idx]][zh_seg_list[zh_idx]] = 1
                else:
                    # add count number
                    self.support[en_seg_list[en_idx]][zh_seg_list[zh_idx]] += 1

        # clean & reorder
        for key in self.support:
            self.support[key] = sorted(self.support[key].items(), key=lambda x: x[1], reverse=True)
            if len(self.support[key]) > 10:
                self.support[key] = self.support[key][:10]
            self.support[key] = [w[0] for w in self.support[key]]
        # record
        with open(os.path.join(self.exp_dir_hmm, "map.json"), 'w', encoding='utf8') as json_file:
            json.dump(self.support, json_file, ensure_ascii=False)

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
