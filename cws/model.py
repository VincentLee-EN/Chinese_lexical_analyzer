# encoding=utf8
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import crf


class BiLSTMModel(object):
    def __init__(self, max_len=200, vocab_size=None, class_num=None, model_save_path=None, embed_size=256, hs=512):
        self.timestep_size = self.max_len = max_len
        self.vocab_size = vocab_size
        self.input_size = self.embedding_size = embed_size
        self.class_num = class_num
        self.hidden_size = hs
        self.lr = tf.placeholder(tf.float32, [])
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.batch_size = tf.placeholder(tf.int32, [])
        self.model_save_path = model_save_path
        # Embedding vector 
        with tf.variable_scope('embedding'):
            self.embedding = tf.get_variable("embedding", [vocab_size, self.embedding_size], dtype=tf.float32)
        self.train()

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def lstm_cell(self):
        cell = rnn.LSTMCell(self.hidden_size, reuse=tf.get_variable_scope().reuse)
        return rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

    def bi_lstm(self, X_inputs):
        # The actual input parameters and the converted output as follows:
        # X_inputs.shape = [batchsize, timestep_size]  ->  inputs.shape = [batchsize, timestep_size, embedding_size]    
        self.inputs = tf.nn.embedding_lookup(self.embedding, X_inputs)
        # The input sentence is still padding filled data.
        # Calculate the actual length of each sentence, that is, the actual length of the non-zero non-padding portion.
        self.length = tf.reduce_sum(tf.sign(X_inputs), 1)
        self.length = tf.cast(self.length, tf.int32)

        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(self.lstm_cell(), self.lstm_cell(), self.inputs,
                                                                    sequence_length=self.length, dtype=tf.float32)
        
        output = tf.concat([output_fw, output_bw], axis=-1)
        output = tf.reshape(output, [-1, self.hidden_size * 2])
        return output

    def train(self):
        with tf.variable_scope('Inputs'):
            self.X_inputs = tf.placeholder(tf.int32, [None, self.timestep_size], name='X_input')
            self.y_inputs = tf.placeholder(tf.int32, [None, self.timestep_size], name='y_input')
        
        bilstm_output = self.bi_lstm(self.X_inputs)
        
        print('The shape of BiLstm Layer output:',bilstm_output.shape)

        with tf.variable_scope('outputs'):
            softmax_w = self.weight_variable([self.hidden_size * 2, self.class_num])
            softmax_b = self.bias_variable([self.class_num])
            self.y_pred = tf.matmul(bilstm_output,
                                    softmax_w) + softmax_b  # there is no softmax, reduce the amount of calculation.
            
            self.scores = tf.reshape(self.y_pred, [-1, self.timestep_size,
                                                   self.class_num])  # [batchsize, timesteps, num_class]
            print('The shape of Output Layer:',self.scores.shape)
            log_likelihood, self.transition_params = crf.crf_log_likelihood(self.scores, self.y_inputs, self.length)
            self.loss = tf.reduce_mean(-log_likelihood)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(self.loss)
