#encoding=utf8
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import crf

import cws.model as modelDef
from cws.data import Data

tf.app.flags.DEFINE_string('dict_path', 'data/your_dict.pkl', 'dict path')
tf.app.flags.DEFINE_string('train_data', 'data/your_train_data.pkl', 'train data path')
tf.app.flags.DEFINE_string('ckpt_path', 'checkpoints/cws.finetune.ckpt/', 'checkpoint path')
tf.app.flags.DEFINE_integer('embed_size', 256, 'embedding size')
tf.app.flags.DEFINE_integer('hidden_size', 512, 'hidden layer node number')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.app.flags.DEFINE_integer('epoch', 24, 'training epoch')
tf.app.flags.DEFINE_float('lr', 0.1, 'learning rate')
tf.app.flags.DEFINE_string('save_path','checkpoints/cws.ckpt/','new model save path')

FLAGS = tf.app.flags.FLAGS

class BiLSTMTrain(object):
    def __init__(self, data_train=None, data_valid=None, data_test=None, model=None):
        self.data_train = data_train
        self.data_valid = data_valid
        self.data_test = data_test
        self.model = model

    def train(self):
       
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
       ## finetune ##
       # ckpt = tf.train.latest_checkpoint(FLAGS.ckpt_path)
       # saver = tf.train.Saver()
       # saver.restore(sess, ckpt)
       # print('-->finetune the ckeckpoint:'+ckpt+'...')
       ##############
        epoch_1 = 8
        epoch_2 = 16
        tr_batch_size = FLAGS.batch_size
        max_max_epoch = FLAGS.epoch  # Max epoch
        display_num = 8  # Display 5 pre epoch
        tr_batch_num = int(self.data_train.y.shape[0] / tr_batch_size)
        
        display_batch = int(tr_batch_num / display_num)  
        saver = tf.train.Saver(max_to_keep=10)

        f = open('report.log', 'w')

        for epoch in range(max_max_epoch): 
            _lr = FLAGS.lr
            if epoch > epoch_1:
                _lr = 0.03
            if epoch > epoch_2:
                _lr = 0.01
            print('EPOCH %d， lr=%g' % (epoch + 1, _lr), file=f)
            start_time = time.time()
            _losstotal = 0.0
            show_loss = 0.0
            for batch in range(tr_batch_num):  
                fetches = [self.model.loss, self.model.train_op]
                X_batch, y_batch = self.data_train.next_batch(tr_batch_size)

                feed_dict = {self.model.X_inputs: X_batch, self.model.y_inputs: y_batch, self.model.lr: _lr,
                             self.model.batch_size: tr_batch_size,
                             self.model.keep_prob: 0.5}
                _loss, _ = sess.run(fetches, feed_dict)  
                _losstotal += _loss
                show_loss += _loss

                if (batch + 1) % display_batch == 0:
                    print('\ttraining loss=%g ' % (show_loss / display_batch), file=f)

                if (epoch + 1) % 8 == 0:
                    valid_acc = self.test_epoch(self.data_valid, sess)  # valid
                    print('\ttraining loss=%g ;  valid acc= %g ' % (show_loss / display_batch,
                                                                             valid_acc), file=f)
                    show_loss = 0.0
            mean_loss = _losstotal / tr_batch_num
            if (epoch + 1) % 1 == 0:  # Save once per epoch
                save_path = saver.save(sess, self.model.model_save_path+'_plus', global_step=(epoch + 1))
                print('the save path is ', save_path, file=f)
            print('\ttraining %d, loss=%g ' % (self.data_train.y.shape[0], mean_loss), file=f)
            print('Epoch training %d, loss=%g, speed=%g s/epoch' % (
                self.data_train.y.shape[0], mean_loss, time.time() - start_time), file=f)

        # testing
        print('**TEST RESULT:', file=f)
        test_acc = self.test_epoch(self.data_test, sess)
        print('**Test %d, acc=%g' % (self.data_test.y.shape[0], test_acc), file=f)
        sess.close()

    def test_epoch(self, dataset=None, sess=None):
        
        _batch_size = 500
        _y = dataset.y
        data_size = _y.shape[0]
        batch_num = int(data_size / _batch_size)  
        correct_labels = 0
        total_labels = 0
        fetches = [self.model.scores, self.model.length, self.model.transition_params]

        for i in range(batch_num):
            X_batch, y_batch = dataset.next_batch(_batch_size)
            feed_dict = {self.model.X_inputs: X_batch, self.model.y_inputs: y_batch, self.model.lr: 1e-5,
                         self.model.batch_size: _batch_size,
                         self.model.keep_prob: 1.0}

            test_score, test_length, transition_params = sess.run(fetches=fetches,
                                                                  feed_dict=feed_dict)
            for tf_unary_scores_, y_, sequence_length_ in zip(
                    test_score, y_batch, test_length):
                tf_unary_scores_ = tf_unary_scores_[:sequence_length_]
                y_ = y_[:sequence_length_]
                viterbi_sequence, _ = crf.viterbi_decode(
                    tf_unary_scores_, transition_params)
                
                correct_labels += np.sum(np.equal(viterbi_sequence, y_))
                total_labels += sequence_length_

        accuracy = correct_labels / float(total_labels)
        return accuracy

def main(_):
    Data_ = Data(dict_path=FLAGS.dict_path, train_data=FLAGS.train_data)
    print('Corpus loading completed:',FLAGS.train_data)
    data_train, data_valid, data_test = Data_.builderTrainData() 
    print('The training set, verification set, and test set split are completed!')
    model = modelDef.BiLSTMModel(max_len=Data_.max_len, 
                                 vocab_size=Data_.word2id.__len__()+1, 
                                 class_num= Data_.tag2id.__len__(), 
                                 model_save_path=FLAGS.save_path, 
                                 embed_size=FLAGS.embed_size,  
                                 hs=FLAGS.hidden_size)
    print('Model definition completed!')
    train = BiLSTMTrain(data_train, data_valid, data_test, model)
    train.train()
    print('Model training completed!')

if __name__ == '__main__':
    tf.app.run()
