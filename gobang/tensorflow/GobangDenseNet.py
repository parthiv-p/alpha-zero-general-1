import sys
sys.path.append('..')
from utils import *

import tensorflow as tf


class GobangNNet():
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Renaming functions 
        Relu = tf.nn.relu
        Tanh = tf.nn.tanh
        BatchNormalization = tf.layers.batch_normalization
        Dropout = tf.layers.dropout
        Dense = tf.layers.dense

        # Neural Net
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input_boards = tf.placeholder(tf.float32, shape=[None, self.board_x, self.board_y])    # s: batch_size x board_x x board_y
            self.dropout = tf.placeholder(tf.float32)
            self.isTraining = tf.placeholder(tf.bool, name="is_training")

            x_image = tf.reshape(self.input_boards, [-1, self.board_x, self.board_y, 1])                    # batch_size  x board_x x board_y x 1
            
        
            # conv layer 1
            output= tf.layers.conv2d(inputs= x_image,
                                        filters=24,
                                        kernel_size=[1,1],
                                       padding="same",
                                       activation=tf.nn.relu)
            numblocks= 3
            for i in range(numblocks-1):
                output= self.dense_block(output)
                output= self.transition_layer(output)

            dense= self.dense_block(output)
            

            # fully connected layers
            h_conv4_flat= tf.layers.flatten(dense)
            

            s_fc1 = Dropout(Relu(BatchNormalization(Dense(h_conv4_flat, 1024), axis=1, training=self.isTraining)), rate=self.dropout) # batch_size x 1024
            s_fc2 = Dropout(Relu(BatchNormalization(Dense(s_fc1, 512), axis=1, training=self.isTraining)), rate=self.dropout)         # batch_size x 512
            self.pi = Dense(s_fc2, self.action_size)                                                        # batch_size x self.action_size
            self.prob = tf.nn.softmax(self.pi)
            self.v = Tanh(Dense(s_fc2, 1))                                                               # batch_size x 1

            self.calculate_loss()

    def conv2d(self, x, out_channels, padding):
      return tf.layers.conv2d(x, out_channels, kernel_size=[3,3], padding=padding)

    
    def dense_block(self, features):

        #convolutional layer 2
        _input = features
        batchnorm1 = tf.layers.batch_normalization(features)
        relu1 = tf.nn.relu(batchnorm1)

        conv1= tf.layers.conv2d(inputs= relu1,
                                filters=12*4,
                                kernel_size=[1,1],
                               padding="same")

        
        batchnorm2 = tf.layers.batch_normalization(conv1)
        relu2 = tf.nn.relu(batchnorm2)

        conv2= tf.layers.conv2d(inputs= relu2,
                                filters=12,
                                kernel_size=[3,3],
                               padding="same")

        concat1 = tf.concat([_input, conv2],-1)

        batchnorm1b = tf.layers.batch_normalization(concat1)
        relu1b = tf.nn.relu(batchnorm1b)

        conv1b = tf.layers.conv2d(inputs= relu1b,
                                filters=12*4,
                                kernel_size=[1,1],
                               padding="same")

        
        batchnorm2b  = tf.layers.batch_normalization(conv1b)
        relu2b  = tf.nn.relu(batchnorm2b)

        conv2b = tf.layers.conv2d(inputs= relu2b,
                                filters=12,
                                kernel_size=[3,3],
                               padding="same")

        concat2 = tf.concat([concat1, conv2b],-1)
        
        return concat2

    def transition_layer(self, features):
            # conv layer 1

        conv1= tf.layers.conv2d(inputs= features,
                                filters=int(features.shape[-1]),
                                kernel_size=[1,1],
                               padding="same",
                               activation=tf.nn.relu)

        #pool1= tf.layers.max_pooling2d(inputs= conv1, pool_size=[2,2], strides=2)

        return conv1
    

    def calculate_loss(self):
        self.target_pis = tf.placeholder(tf.float32, shape=[None, self.action_size])
        self.target_vs = tf.placeholder(tf.float32, shape=[None])
        self.loss_pi =  tf.losses.softmax_cross_entropy(self.target_pis, self.pi)
        self.loss_v = tf.losses.mean_squared_error(self.target_vs, tf.reshape(self.v, shape=[-1,]))
        self.total_loss = self.loss_pi + self.loss_v
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(self.args.lr).minimize(self.total_loss)
