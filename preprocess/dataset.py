import xml.etree.ElementTree as ET
import numpy as np
import os

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
            # zh = zh.replace(" ", "")
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
            # zh = zh.replace(" ", "")
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
            # zh = zh.replace(" ", "")
            data_sample = {}
            data_sample["en"] = en
            data_sample["zh"] = zh
            self.data.append(data_sample)

        corpus_en_file.close()
        corpus_zh_file.close()
    
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
