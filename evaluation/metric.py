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
