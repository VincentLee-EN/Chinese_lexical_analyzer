import os
import pickle
import numpy as np
import pandas as pd
from itertools import chain
from example.sentence import Sentence
from example.sentence import TagPrefix
from example.sentence import TagSurfix
import json


class DataHandler(object):
    def __init__(self, rootDir=None, dict_path=None, train_data=None):
        self.rootDir = rootDir
        self.dict_path = dict_path
        self.train_data = train_data
        self.spiltChar = ['。', '!', '！', '？', '?']
        self.max_len = 200
        self.totalLine = 0
        self.longLine = 0
        self.totalChars = 0
        self.tag_set = list()
        self.TAGPRE = TagPrefix.convert()

    def loadData(self):
        self.loadRawData()
        self.handlerRawData()

    def loadRawData(self):
        self.datas = list()
        self.labels = list()
        if self.rootDir:
            print(self.rootDir)
            for dirName, subdirList, fileList in os.walk(self.rootDir):
                # curDir = os.path.join(self.rootDir, dirName)
                for file in fileList:
                    if file.endswith(".txt"):
                        curFile = os.path.join(dirName, file)
                        print("processing:%s" % (curFile))
                        with open(curFile, "r", encoding='utf-8') as fp:
                            for line in fp.readlines():
                                self.processLine(line)

            print("total:%d, long lines:%d, total chars:%d" % (self.totalLine, self.longLine, self.totalChars))
            print('Length of datas is %d' % len(self.datas))
            print('Example of datas: ', self.datas[0])
            print('Example of labels:', self.labels[0])

    def processLine(self, line):
        line = line.strip()
        nn = len(line)
        seeLeftB = False
        start = 0
        sentence = Sentence()
        try:
            for i in range(nn):
                if line[i] == ' ':
                    if not seeLeftB:
                        token = line[start:i]
                        if token.startswith('['):
                            token_ = ''
                            for j in [i.split('/') for i in token.split('[')[1].split(']')[0].split(' ')]:
                                token_ += j[0]
                            token_ = token_ + '/' + token.split('/')[-1]
                            self.processToken(token_, sentence, False)
                        else:
                            self.processToken(token, sentence, False)
                        start = i + 1
                elif line[i] == '[':
                    seeLeftB = True
                elif line[i] == ']':
                    seeLeftB = False
            # 此部分未与上面处理方式统一，（小概率事件）数据多元化，增加模型泛化能力。
            if start < nn:
                token = line[start:]
                if token.startswith('['):
                    tokenLen = len(token)
                    while tokenLen > 0 and token[tokenLen - 1] != ']':
                        tokenLen = tokenLen - 1
                    token = token[1:tokenLen - 1]
                    ss = token.split(' ')
                    ns = len(ss)
                    for i in range(ns - 1):
                        self.processToken(ss[i], sentence, False)
                    self.processToken(ss[-1], sentence, True)
                else:
                    self.processToken(token, sentence, True)
        except Exception as e:
            print('处理数据异常, 异常行为：' + line)
            print(e)

    def processToken(self, tokens, sentence, end):
        nn = len(tokens)
        while nn > 0 and tokens[nn - 1] != '/':
            nn = nn - 1

        token = tokens[:nn - 1].strip()
        tagPre = tokens[nn:].strip()
        # tagPre = self.TAGPRE.get(tagpre, TagPrefix.general.value)
        # # if tagPre == '':
        # #     tagPre = tagpre + '_'
        # ner_list = ['nr', 'ns', 'nt', 't', 'nz']
        # is_ner = False
        # for ner in ner_list:
        #     if tagPre.startswith(ner):
        #         tagPre = ner
        #         is_ner = True
        #         break
        # if not is_ner:
        #     tagPre = tagPre[0]
        tagPre += '_'
        if token not in self.spiltChar:
            sentence.addToken(token, tagPre)
        if token in self.spiltChar or end:
            if sentence.chars > self.max_len:
                self.longLine += 1
            else:
                x = []
                y = []
                self.totalChars += sentence.chars
                sentence.generate_tr_line(x, y)

                if len(x) > 0 and len(x) == len(y):
                    self.datas.append(x)
                    self.labels.append(y)
                    self.tag_set.extend(y)
                else:
                    print('处理一行数据异常, 异常行如下')
                    print(sentence.tokens)
            self.totalLine += 1
            sentence.clear()

    def handlerRawData(self):
        self.df_data = pd.DataFrame({'words': self.datas, 'tags': self.labels}, index=range(len(self.datas)))
        self.df_data['sentence_len'] = self.df_data['words'].apply(
            lambda words: len(words))

        all_words = list(chain(*self.df_data['words'].values))
        sr_allwords = pd.Series(all_words)
        sr_allwords = sr_allwords.value_counts()

        set_words = sr_allwords.index
        set_ids = range(1, len(set_words) + 1)

        tags = list(set(self.tag_set))
        print(tags)

        tag_ids = range(len(tags))

        self.word2id = pd.Series(set_ids, index=set_words)
        self.id2word = pd.Series(set_words, index=set_ids)
        self.id2word[len(set_ids) + 1] = 'NEW'
        self.word2id["NEW"] = len(set_ids) + 1

        self.tag2id = pd.Series(tag_ids, index=tags)
        self.id2tag = pd.Series(tags, index=tag_ids)

        self.df_data['X'] = self.df_data['words'].apply(self.X_padding)
        self.df_data['y'] = self.df_data['tags'].apply(self.y_padding)

        self.X = np.asarray(list(self.df_data['X'].values))
        self.y = np.asarray(list(self.df_data['y'].values))
        print('X.shape={}, y.shape={}'.format(self.X.shape, self.y.shape))
        print('Example of words: ', self.df_data['words'].values[0])
        print('Example of X: ', self.X[0])
        print('Example of tags: ', self.df_data['tags'].values[0])
        print('Example of y: ', self.y[0])

        with open(self.dict_path, 'wb') as outp:
            pickle.dump(self.word2id, outp)
            pickle.dump(self.id2word, outp)
            pickle.dump(self.tag2id, outp)
            pickle.dump(self.id2tag, outp)
        print('** Finished saving the dict.')

        with open(self.train_data, 'wb') as outp:
            pickle.dump(self.X, outp)
            pickle.dump(self.y, outp)
        print('** Finished saving the train data.')


    def X_padding(self, words):

        ids = list(self.word2id[words])
        if len(ids) >= self.max_len:
            return ids[:self.max_len]
        ids.extend([0] * (self.max_len - len(ids)))
        return ids

    def y_padding(self, tags):
        ids = list(self.tag2id[tags])
        if len(ids) >= self.max_len:
            return ids[:self.max_len]
        ids.extend([0] * (self.max_len - len(ids)))
        return ids


if __name__ == '__main__':
    data = DataHandler(rootDir='../corpus/people-2014', dict_path='../data/your_dict.pkl', train_data='../data/your_train_data.pkl')
    data.loadData()

    print(data.X)
    print(type(data.X))
    print(data.X.shape)

