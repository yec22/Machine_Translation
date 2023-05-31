import os
import json
import nltk
from queue import PriorityQueue
from copy import *
import math


class Model:
    def __init__(self):
        self.data_root = "data/zh-en"
        self.exp_dir = "exps_HMM"

        if not os.path.exists(os.path.join(self.exp_dir, "HMM_PI.json")):
            print("HMM_PI.json Not Found.")
        else:
            with open(os.path.join(self.exp_dir, "HMM_PI.json"), 'r', encoding='utf-8') as json_file:
                self.HMM_PI = json.load(json_file)

        if not os.path.exists(os.path.join(self.exp_dir, "HMM_A.json")):
            print("HMM_A.json Not Found.")
        else:
            with open(os.path.join(self.exp_dir, "HMM_A.json"), 'r', encoding='utf-8') as json_file:
                self.HMM_A = json.load(json_file)

        if not os.path.exists(os.path.join(self.exp_dir, "HMM_B.json")):
            print("HMM_B.json Not Found.")
        else:
            with open(os.path.join(self.exp_dir, "HMM_B.json"), 'r', encoding='utf-8') as json_file:
                self.HMM_B = json.load(json_file)

        if not os.path.exists(os.path.join(self.exp_dir, "map.json")):
            print("map.json Not Found.")
        else:
            with open(os.path.join(self.exp_dir, "map.json"), 'r', encoding='utf-8') as json_file:
                self.map = json.load(json_file)

        nltk.download('punkt')

    def get_log_pb(self, zh, en):
        if zh in self.HMM_B:
            if en in self.HMM_B[zh]:
                return math.log(self.HMM_B[zh][en])
        return 1

    def get_log_pa(self, zh1, zh2):
        if zh1 in self.HMM_A:
            if zh2 in self.HMM_A[zh1]:
                return math.log(self.HMM_A[zh1][zh2])
        return 1

    def translate(self, en):
        en_seg_list = nltk.word_tokenize(en)
        en_seg_list = [w.lower() for w in en_seg_list]

        T = len(en_seg_list)

        viterbi = [{} for _ in range(T)]

        for key in self.HMM_PI:
            log_pb = self.get_log_pb(key, en_seg_list[0])
            if log_pb < 0:
                viterbi[0][key] = [math.log(self.HMM_PI[key]) + log_pb, '']

        for i in range(1, T):
            en = en_seg_list[i]
            if viterbi[i - 1]:
                if en in self.map:
                    for zh_t in self.map[en]:
                        delta_max = -1e6
                        psi = ''
                        for zh in viterbi[i - 1]:
                            delta = viterbi[i - 1][zh]
                            log_pa = self.get_log_pa(zh, zh_t)
                            if log_pa < 0:
                                delta_t = delta + log_pa + self.get_log_pb(zh_t, en)
                                if delta_t > delta_max:
                                    delta_max = delta_t
                                    psi = zh
                        viterbi[i][zh_t] = [delta_max, psi]

        target = ""
        if viterbi[-1]:
            delta_max = -1e6
            psi = ''
            for key in viterbi[-1]:
                delta = viterbi[-1][key][0]
                if delta > delta_max:
                    delta_max = delta
                    psi = key
            target = psi
            for i in range(T - 1, -1, -1):
                psi = viterbi[i][psi][1]
                target = psi + target

        return target
