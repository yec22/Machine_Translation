import jieba

class Metric:
    def __init__(self):
        self.mode = "BLEU-1"
    
    def eval(self, pred, target):
        '''
        param:
            pred(list): ['aaa', 'bbb', 'ccc']
            target(list): ['xxx', 'yyy', 'zzz']
        return:
            BLUE(list), avg_BLUE(float)
        '''
        BLUE_val = []

        for s, t in zip(pred, target):
            s_seg_list = list(jieba.cut(s, cut_all=False))
            t_seg_list = list(jieba.cut(t, cut_all=False))
            
            blue = 0
            for seg in s_seg_list:
                if seg in t_seg_list:
                    blue += 1
            BLUE_val.append(blue / len(s_seg_list))

        return BLUE_val, sum(BLUE_val) / len(BLUE_val)

    def eval_2(self, pred, target):
        '''
        param:
            pred(list): ['aaa', 'bbb', 'ccc']
            target(list): ['xxx', 'yyy', 'zzz']
        return:
            BLUE(list), avg_BLUE(float)
        '''
        BLUE_val = []

        for s, t in zip(pred, target):
            s_seg_list = list(jieba.cut(s, cut_all=False))
            t_seg_list = list(jieba.cut(t, cut_all=False))
            
            blue = 0
            for i in range(len(s_seg_list) - 1):
                word1 = s_seg_list[i]
                word2 = s_seg_list[i+1]
                for j in range(len(t_seg_list) - 1):
                    if (word1 == t_seg_list[j]) and (word2 == t_seg_list[j+1]):
                        blue += 1

            BLUE_val.append(blue / (len(s_seg_list) - 1))

        return BLUE_val, sum(BLUE_val) / len(BLUE_val)