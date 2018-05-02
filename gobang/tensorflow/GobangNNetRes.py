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
            
            conv1= tf.layers.conv2d(inputs= x_image,
                                filters=64,
                                kernel_size=[7,7], strides= 2,
                               activation=tf.nn.relu)
            batchnorm1= tf.layes.batch_normalization(conv1)
            relu1 = tf.nn.relu(batchnorm1, name=None)

            output= tf.layers.max_pooling2d(inputs= relu1, pool_size=[3,3], strides=2)

            fltrs= 64
            for i in range(1,5):
                output= res_block(output, fltrs)
                fltrs= 64*i

                conv1a= outputs
                conv2= tf.layers.conv2d(inputs= conv1a,
                                    filters=fltrs,
                                    kernel_size=[3,3], strides=2)

                batchnorm2= tf.layes.batch_normalization(conv2)
                relu2 = tf.nn.relu(batchnorm2, name=None)

                conv3= tf.layers.conv2d(inputs= relu2,
                                    filters=fltrs,
                                    kernel_size=[3,3],
                                   padding="same")
                batchnorm3= tf.layes.batch_normalization(conv3)
                relu3 = tf.nn.relu(batchnorm3, name=None)

                #concatenation   
                output= tf.concat([conv1a,relu3],-1)

            # conv1a has the output


            # fully connected layers
            h_conv4_flat= tf.contrib.layers.flatten(conv1a)
            

            s_fc1 = Dropout(Relu(BatchNormalization(Dense(h_conv4_flat, 1024), axis=1, training=self.isTraining)), rate=self.dropout) # batch_size x 1024
            s_fc2 = Dropout(Relu(BatchNormalization(Dense(s_fc1, 512), axis=1, training=self.isTraining)), rate=self.dropout)         # batch_size x 512
            self.pi = Dense(s_fc2, self.action_size)                                                        # batch_size x self.action_size
            self.prob = tf.nn.softmax(self.pi)
            self.v = Tanh(Dense(s_fc2, 1))                                                               # batch_size x 1

            self.calculate_loss()

    def conv2d(self, x, out_channels, padding):
      return tf.layers.conv2d(x, out_channels, kernel_size=[3,3], padding=padding)

    

    def res_block(self, features, fltr):

        #convolutional layer 2
        _input= features
        conv1= tf.layers.conv2d(inputs= conv1,
                                filters=fltr,
                                kernel_size=[3,3],
                               padding="same")
        batchnorm1= tf.layes.batch_normalization(conv1)
        relu1 = tf.nn.relu(batchnorm1, name=None)

        conv2= tf.layers.conv2d(inputs= relu1,
                                filters=fltr,
                                kernel_size=[3,3],
                               padding="same")
        batchnorm2= tf.layes.batch_normalization(conv2)
        relu2 = tf.nn.relu(batchnorm2, name=None)
        #concatenation   
        concat1= tf.concat([_input,relu2],-1)

        return concat1

   


    def calculate_loss(self):
        self.target_pis = tf.placeholder(tf.float32, shape=[None, self.action_size])
        self.target_vs = tf.placeholder(tf.float32, shape=[None])
        self.loss_pi =  tf.losses.softmax_cross_entropy_with_logits(self.target_pis, self.pi)
        self.loss_v = tf.losses.mean_squared_error(self.target_vs, tf.reshape(self.v, shape=[-1,]))
        self.total_loss = self.loss_pi + self.loss_v
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(self.args.lr).minimize(self.total_loss)




