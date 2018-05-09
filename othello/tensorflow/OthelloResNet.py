import sys
sys.path.append('..')
from utils import *

import tensorflow as tf

class OthelloNNet():
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
            
            
            inputs= tf.layers.conv2d(inputs= x_image,
                                filters=64,
                                kernel_size=[7,7], strides= 2, padding='same')
            

            inputs= tf.layers.max_pooling2d(inputs= inputs, pool_size=[3,3], strides=2, padding='same')
            inputs= self.res_block(inputs, 64, 1)
            inputs= self.res_block(inputs, 64, 1)

            filters= 64
            for i in range(2,5):
                fltrs= filters*2
                
                inputs= self.res_block(inputs, filters, 1)
                inputs= self.res_block(inputs, 64, 1)
                
            inputs = inputs= tf.layers.batch_normalization(inputs)
            inputs = tf.nn.relu(inputs)     

            # fully connected layers
            h_conv4_flat= tf.contrib.layers.flatten(inputs)
            
            
            s_fc1 = Dropout(Relu(BatchNormalization(Dense(h_conv4_flat, 1024), axis=1, training=self.isTraining)), rate=self.dropout) # batch_size x 1024
            s_fc2 = Dropout(Relu(BatchNormalization(Dense(s_fc1, 512), axis=1, training=self.isTraining)), rate=self.dropout)         # batch_size x 512
            self.pi = Dense(s_fc2, self.action_size)                                                        # batch_size x self.action_size
            self.prob = tf.nn.softmax(self.pi)
            self.v = Tanh(Dense(s_fc2, 1))                                                               # batch_size x 1

            self.calculate_loss()

    def conv2d(self, x, out_channels, padding):
      return tf.layers.conv2d(x, out_channels, kernel_size=[3,3], padding=padding)

    def res_block(self, inputs, filters, strides):

        #convolutional layer 2
        shortcut= inputs
        inputs= tf.layers.batch_normalization(inputs)
        inputs= tf.nn.relu(inputs)
        
        
        inputs= tf.layers.conv2d(inputs= inputs,
                                filters=filters, strides= strides,
                                kernel_size=[3,3],
                               padding="same")
        inputs= tf.layers.batch_normalization(inputs)
        inputs = tf.nn.relu(inputs)

        inputs= tf.layers.conv2d(inputs= inputs,
                                filters=filters,
                                kernel_size=[3,3], strides= 1,
                               padding="same")

        #concatenation   
        concat1= tf.concat([shortcut,inputs],-1)

        return concat1
    

    def calculate_loss(self):
        self.target_pis = tf.placeholder(tf.float32, shape=[None, self.action_size])
        self.target_vs = tf.placeholder(tf.float32, shape=[None])
        self.loss_pi =  tf.losses.softmax_cross_entropy(self.target_pis, self.pi)
        self.loss_v = tf.losses.mean_squared_error(self.target_vs, tf.reshape(self.v, shape=[-1,]))
        self.total_loss = self.loss_pi + self.loss_v
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(self.args.lr).minimize(self.total_loss)