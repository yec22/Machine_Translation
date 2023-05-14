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


if __name__ == "__main__":
    metric = Metric()
    BLUE, avg = metric.eval(pred=['我周一来到了清华大学校园', '你好，我是李华', '今天天气真不错'], 
                            target=['我周二来到了校园', '你好，我是小明', '今天天气真不错'])
    print(BLUE)
    print(avg)
