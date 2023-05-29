import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import sys
sys.path.append("../preprocess/")
import dataset as dd
import jieba
import time

class MyDataset(Dataset):
    def __init__(self,en_data,ch_data,en_word_2_index,ch_word_2_index):
        self.en_data = en_data
        self.ch_data = ch_data
        self.en_word_2_index = en_word_2_index
        self.ch_word_2_index = ch_word_2_index

    def __getitem__(self,index):
        en = self.en_data[index]
        ch = self.ch_data[index]

        en_index = [self.en_word_2_index[i] for i in en.split()]
        ch_index = [self.ch_word_2_index[i] for i in jieba.cut(ch)]

        return en_index,ch_index


    def batch_data_process(self,batch_datas):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        en_index , ch_index = [],[]
        en_len , ch_len = [],[]

        for en,ch in batch_datas:
            en_index.append(en)
            ch_index.append(ch)
            en_len.append(len(en))
            ch_len.append(len(ch))

        max_en_len = max(en_len)
        max_ch_len = max(ch_len)

        en_index = [ i + [self.en_word_2_index["<PAD>"]] * (max_en_len - len(i))   for i in en_index]
        ch_index = [[self.ch_word_2_index["<BOS>"]]+ i + [self.ch_word_2_index["<EOS>"]] + [self.ch_word_2_index["<PAD>"]] * (max_ch_len - len(i))   for i in ch_index]

        en_index = torch.tensor(en_index,device = device)
        ch_index = torch.tensor(ch_index,device = device)


        return en_index,ch_index


    def __len__(self):
        assert len(self.en_data) == len(self.ch_data)
        return len(self.ch_data)


class Encoder(nn.Module):
    def __init__(self,encoder_embedding_num,encoder_hidden_num,en_corpus_len):
        super().__init__()
        self.embedding = nn.Embedding(en_corpus_len,encoder_embedding_num)
        self.lstm = nn.LSTM(encoder_embedding_num,encoder_hidden_num,batch_first=True)

    def forward(self,en_index):
        en_embedding = self.embedding(en_index)
        _,encoder_hidden =self.lstm(en_embedding)

        return encoder_hidden



class Decoder(nn.Module):
    def __init__(self,decoder_embedding_num,decoder_hidden_num,ch_corpus_len):
        super().__init__()
        self.embedding = nn.Embedding(ch_corpus_len,decoder_embedding_num)
        self.lstm = nn.LSTM(decoder_embedding_num,decoder_hidden_num,batch_first=True)

    def forward(self,decoder_input,hidden):
        embedding = self.embedding(decoder_input)
        decoder_output,decoder_hidden = self.lstm(embedding,hidden)

        return decoder_output,decoder_hidden

class Seq2Seq(nn.Module):
    def __init__(self,encoder_embedding_num,encoder_hidden_num,en_corpus_len,decoder_embedding_num,decoder_hidden_num,ch_corpus_len):
        super().__init__()
        self.encoder = Encoder(encoder_embedding_num,encoder_hidden_num,en_corpus_len)
        self.decoder = Decoder(decoder_embedding_num,decoder_hidden_num,ch_corpus_len)
        self.classifier = nn.Linear(decoder_hidden_num,ch_corpus_len)

        self.cross_loss = nn.CrossEntropyLoss()

    def forward(self,en_index,ch_index):
        decoder_input = ch_index[:,:-1]
        label = ch_index[:,1:]

        encoder_hidden = self.encoder(en_index)
        decoder_output,_ = self.decoder(decoder_input,encoder_hidden)

        pre = self.classifier(decoder_output)
        loss = self.cross_loss(pre.reshape(-1,pre.shape[-1]),label.reshape(-1))

        return loss

class Model:
    def __init__(self, encoding_embedding_num, encoding_hidden_num, decoder_embedding_num, decoder_hidden_num, batch_size):
        data = dd.Dataset().get_all_item()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.ch_word_2_index = {}
        self.ch_index_2_word = []
        self.ch_datas = []
        self.en_word_2_index = {}
        self.en_index_2_word = []
        self.en_datas = []
        en_idx = 0
        ch_idx = 0
        for pair in data:
            for char in pair["en"].split():
                if not self.en_word_2_index.__contains__(char):
                    self.en_word_2_index[char] = en_idx
                    self.en_index_2_word.append(char)
                    en_idx += 1
            for char in jieba.cut(pair["zh"]):
                if not self.ch_word_2_index.__contains__(char):
                    self.ch_word_2_index[char] = ch_idx
                    self.ch_index_2_word.append(char)
                    ch_idx += 1
            self.ch_datas.append(pair["zh"])
            self.en_datas.append(pair["en"])
        ch_corpus_len = len(self.ch_word_2_index)
        en_corpus_len = len(self.en_word_2_index)
        self.ch_word_2_index.update({"<PAD>":ch_corpus_len, "<BOS>":ch_corpus_len + 1 , "<EOS>":ch_corpus_len+2})
        self.en_word_2_index.update({"<PAD>":en_corpus_len})
        self.ch_index_2_word += ["<PAD>","<BOS>","<EOS>"]
        self.en_index_2_word += ["<PAD>"]
        ch_corpus_len += 3
        en_corpus_len += 1
        self.dataset = MyDataset(self.en_datas, self.ch_datas, self.en_word_2_index, self.ch_word_2_index)
        self.dataloader = DataLoader(self.dataset, batch_size, shuffle = False, collate_fn = self.dataset.batch_data_process)
        self.model = Seq2Seq(encoding_embedding_num, encoding_hidden_num, en_corpus_len, decoder_embedding_num, decoder_hidden_num, ch_corpus_len)
        self.model = self.model.to(self.device)
    
    def train(self, epoch, lr):
        opt = torch.optim.Adam(self.model.parameters(), lr = lr)
        begin = time.time()
        for e in range(epoch):
            for en_idx, ch_idx in self.dataloader:
                loss = self.model(en_idx, ch_idx)
                loss.backward()
                opt.step()
                opt.zero_grad()
            end = time.time()
            print("Epoch", e+1, "time =", round(end - begin, 2))
            begin = end
    
    def translate(self, sentence):
        en_index = torch.tensor([[self.en_word_2_index[i] for i in sentence.split()]],device = self.device)
        result = []
        encoder_hidden = self.model.encoder(en_index)
        decoder_input = torch.tensor([[self.ch_word_2_index["<BOS>"]]],device = self.device)
        decoder_hidden = encoder_hidden
        while True:
            decoder_output, decoder_hidden = self.model.decoder(decoder_input,decoder_hidden)
            pre = self.model.classifier(decoder_output)
            w_index = int(torch.argmax(pre, dim=-1))
            word = self.ch_index_2_word[w_index]
            if word == "<EOS>" or len(result) > 50:
               break
            result.append(word)
            decoder_input = torch.tensor([[w_index]], device = self.device)

        return "".join(result)


if __name__ == "__main__":
    m = Model(50, 100, 107, 100, 2)
    m.train(40, 0.001)

    while True:
        s = input("请输入英文: ")
        print(m.translate(s))
