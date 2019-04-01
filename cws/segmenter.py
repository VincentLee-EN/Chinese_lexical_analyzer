# encoding=utf8
import re
import numpy as np
import tensorflow as tf
from tensorflow.contrib import crf

from .model import BiLSTMModel
from .data import Data


class BiLSTMSegmenter(object):
    def __init__(self, data_path=None, model_path=None):
        self.data = Data(data_path)
        self.g1 = tf.Graph()
        self.sess1 = tf.Session(graph=self.g1)
        with self.sess1.as_default():
            with self.g1.as_default():
                self.model = BiLSTMModel(vocab_size=self.data.word2id.__len__()+1, class_num=self.data.tag2id.__len__())
                checkpoint = tf.train.latest_checkpoint(model_path)
                tf.train.Saver().restore(self.sess1, checkpoint)

    # Full-width half-turn
    def format_standardization(self, text):
        rstring = ""
        for uchar in text:
            inside_code = ord(uchar)
            if inside_code == 12288:  # Full-width space conversion
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  # Full-width characters (except spaces)
                inside_code -= 65248  # are transformed according to relationships
            rstring += chr(inside_code)
        return rstring           
    
    # Transfor word to id
    def text2ids(self, text):
        words = list(text)
        # new word
        def f(w):
            if w in self.data.word2id:
                return self.data.word2id[w]
            else:
                return self.data.word2id['NEW']
        ids = [f(w) for w in words]
        
        if len(ids) >= self.data.max_len:
            # print(u'输入句长超过%d，无法完成处理！' % self.data.max_len)
            ids = ids[:self.data.max_len]
            ids = np.asarray(ids).reshape([-1, self.data.max_len])
            return ids
        else:
            ids.extend([0] * (self.data.max_len - len(ids)))
            ids = np.asarray(ids).reshape([-1, self.data.max_len])
            return ids

    # Label prediction
    def simple_cut(self, text, sess=None):
        if text:
            X_batch = self.text2ids(text)
            fetches = [self.model.scores, self.model.length, self.model.transition_params]
            feed_dict = {self.model.X_inputs: X_batch,
                         self.model.lr: 1.0,
                         self.model.batch_size: 1,
                        self.model.keep_prob: 1.0}
            test_score, test_length, transition_params = sess.run(fetches, feed_dict)
            tags, _ = crf.viterbi_decode(test_score[0][:test_length[0]], transition_params)
            tags = [self.data.id2tag[t] for t in tags]           
            return tags
        else:
            return []
    
    # Cut line by predefined token
    def cut_word(self, sentence):
        not_cuts = re.compile(u'[，。？！\?!]')
        result = []
        start = 0    
        sentence = self.format_standardization(sentence)
        for seg_sign in not_cuts.finditer(sentence):
            result.extend(self.simple_cut(sentence[start:seg_sign.end()], self.sess1))
            start = seg_sign.end()
        result.extend(self.simple_cut(sentence[start:], self.sess1))
        return result

    def get_ner(self, tag):
        if tag == 't':
            tag = 'TIME'
        elif tag == 'nr':
            tag = 'PER'
        elif tag == 'nt':
            tag = 'ORG'
        elif tag == 'ns':
            tag = 'LOC'
        return tag

    # Output format transfor
    def output(self, text, labels):
        words = list()
        tags = list()

        text += 'x'
        labels.append('O_X')
        rss = text[0]
        flag = labels[0].split('_')

        for i in range(len(labels))[1:]:
            token = labels[i].split('_')
            if token[1] in ['s', 'b', 'X']:
                words.append(rss)
                tags.append(self.get_ner(flag[0]))
                flag = token
                rss = text[i]
            if token[1] in ['m', 'e']:
                rss += text[i]

        return words, tags

    def predict(self, text):
        with self.sess1.as_default():
            with self.sess1.graph.as_default():
                cws_result = self.cut_word(text)
                #print('--label--:', cws_result)
                rss = self.output(text, cws_result)
                #print('--result--:', rss)
                return rss
