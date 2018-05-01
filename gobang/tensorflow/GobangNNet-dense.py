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
            
            x1 = tf.convert_to_tensor(x_image, np.float32)
            # conv layer 1
            output= tf.layers.conv2d(inputs= x1,
                                        filters=24,
                                        kernel_size=[1,1],
                                       padding="same",
                                       activation=tf.nn.relu)
            numblocks= 3
            for i in range(numblocks-1):
                output= dense_block(output)
                output= transition_layer(output)

            dense= dense_block(output)
            pool1= tf.layers.max_pooling2d(inputs= dense,pool_size=[2,2], strides=2)

            # fully connected layers
            h_conv4_flat= tf.reshape(pool1, [-1,int(pool1.shape[-1])*int(pool1.shape[-2])*int(pool1.shape[-3])])
            

            s_fc1 = Dropout(Relu(BatchNormalization(Dense(h_conv4_flat, 1024), axis=1, training=self.isTraining)), rate=self.dropout) # batch_size x 1024
            s_fc2 = Dropout(Relu(BatchNormalization(Dense(s_fc1, 512), axis=1, training=self.isTraining)), rate=self.dropout)         # batch_size x 512
            self.pi = Dense(s_fc2, self.action_size)                                                        # batch_size x self.action_size
            self.prob = tf.nn.softmax(self.pi)
            self.v = Tanh(Dense(s_fc2, 1))                                                               # batch_size x 1

            self.calculate_loss()

    def conv2d(self, x, out_channels, padding):
      return tf.layers.conv2d(x, out_channels, kernel_size=[3,3], padding=padding)

    
    def dense_block(features):

        #convolutional layer 2
        conv1= features
        conv2= tf.layers.conv2d(inputs= conv1,
                                filters=12,
                                kernel_size=[3,3],
                               padding="same",
                               activation=tf.nn.relu)

        #concatenation   
        concat1= tf.concat([conv1,conv2],-1)

        # conv layer 3

        conv3a= tf.layers.conv2d(inputs= conv2, filters=12*4, kernel_size=[1,1],
                             padding="same", activation=tf.nn.relu)

        conv3= tf.layers.conv2d(inputs= conv3a, filters=12, kernel_size=[3,3],
                             padding="same", activation=tf.nn.relu)

        #concatenation
        concat2= tf.concat([concat1, conv3], -1) 

        # conv layer 3

        conv4a= tf.layers.conv2d(inputs= concat2, filters=12*4, kernel_size=[1,1],
                             padding="same", activation=tf.nn.relu)

        conv4= tf.layers.conv2d(inputs= conv4a, filters=12, kernel_size=[3,3],
                             padding="same", activation=tf.nn.relu)

        #concatenation
        concat3= tf.concat([concat2, conv4], -1) 
        return concat3



    def transition_layer(features):
            # conv layer 1

        conv1= tf.layers.conv2d(inputs= features,
                                filters=int(features.shape[-1]),
                                kernel_size=[5,5],
                               padding="same",
                               activation=tf.nn.relu)

        pool1= tf.layers.max_pooling2d(inputs= conv1, pool_size=[2,2], strides=2)

        return pool1
    
    
def dense_block(features):

    #convolutional layer 2
    conv1= features
    conv2= tf.layers.conv2d(inputs= conv1,
                            filters=12,
                            kernel_size=[3,3],
                           padding="same",
                           activation=tf.nn.relu)
    
    #concatenation   
    concat1= tf.concat([conv1,conv2],-1)
    
    # conv layer 3
    
    conv3a= tf.layers.conv2d(inputs= conv2, filters=12*4, kernel_size=[1,1],
                         padding="same", activation=tf.nn.relu)
    
    conv3= tf.layers.conv2d(inputs= conv3a, filters=12, kernel_size=[3,3],
                         padding="same", activation=tf.nn.relu)

    #concatenation
    concat2= tf.concat([concat1, conv3], -1) 

    # conv layer 3
    
    conv4a= tf.layers.conv2d(inputs= concat2, filters=12*4, kernel_size=[1,1],
                         padding="same", activation=tf.nn.relu)
    
    conv4= tf.layers.conv2d(inputs= conv4a, filters=12, kernel_size=[3,3],
                         padding="same", activation=tf.nn.relu)

    #concatenation
    concat3= tf.concat([concat2, conv4], -1) 
    return concat3
    

    
def transition_layer(features):
        # conv layer 1
    
    conv1= tf.layers.conv2d(inputs= features,
                            filters=int(features.shape[-1]),
                            kernel_size=[5,5],
                           padding="same",
                           activation=tf.nn.relu)
    
    pool1= tf.layers.max_pooling2d(inputs= conv1, pool_size=[2,2], strides=2)
    
    return pool1
    
    


    def calculate_loss(self):
        self.target_pis = tf.placeholder(tf.float32, shape=[None, self.action_size])
        self.target_vs = tf.placeholder(tf.float32, shape=[None])
        self.loss_pi =  tf.losses.softmax_cross_entropy_with_logits(self.target_pis, self.pi)
        self.loss_v = tf.losses.mean_squared_error(self.target_vs, tf.reshape(self.v, shape=[-1,]))
        self.total_loss = self.loss_pi + self.loss_v
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(self.args.lr).minimize(self.total_loss)




