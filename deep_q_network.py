#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
#import cv2
import sys
sys.path.append("game/")
import traffic as game
import random
import numpy as np
from collections import deque

GAME = 'traffic' # the name of the game being played for log files
ACTIONS = 3 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 5000. # timesteps to observe before training
EXPLORE = 1000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.001 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def createNetwork():

    W_fc1 = weight_variable([30, 300])
    b_fc1 = bias_variable([300])

    W_fc2 = weight_variable([300, 300])
    b_fc2 = bias_variable([300])


    W_fc3 = weight_variable([300, 300])
    b_fc3 = bias_variable([300])


    W_fc4 = weight_variable([300, 300])
    b_fc4 = bias_variable([300])


    W_fc5 = weight_variable([300, 300])
    b_fc5 = bias_variable([300])

    W_fc5 = weight_variable([300, 300])
    b_fc5 = bias_variable([300])

    W_fc6 = weight_variable([300, 300])
    b_fc6 = bias_variable([300])

    W_fc7 = weight_variable([300, 300])
    b_fc7 = bias_variable([300])

    W_fc8 = weight_variable([300, 300])
    b_fc8 = bias_variable([300])


    W_fc9 = weight_variable([300, 300])
    b_fc9 = bias_variable([300])

    W_fc10 = weight_variable([300, 300])
    b_fc10 = bias_variable([300])
    
    W_fc11 = weight_variable([300, 300])
    b_fc11 = bias_variable([300])

    W_fc12 = weight_variable([300, 300])
    b_fc12 = bias_variable([300])

    W_fc13 = weight_variable([300, 300])
    b_fc13 = bias_variable([300])

    W_fc14 = weight_variable([300, 300])
    b_fc14 = bias_variable([300])

    W_fc15 = weight_variable([300, 300])
    b_fc15 = bias_variable([300])

    W_fc16 = weight_variable([300, 300])
    b_fc16 = bias_variable([300])

    W_fc17 = weight_variable([300, 300])
    b_fc17 = bias_variable([300])

    W_fc18 = weight_variable([300, 300])
    b_fc18 = bias_variable([300])
       
    W_fc19 = weight_variable([300, 300])
    b_fc19 = bias_variable([300])

    W_fc20 = weight_variable([300, ACTIONS])
    b_fc20 = bias_variable([ACTIONS])

 


    # input layer
    s = tf.placeholder("float", [None, 30])


    h_fc1 = tf.nn.sigmoid(tf.matmul(s, W_fc1) + b_fc1)
    h_fc2 = tf.nn.sigmoid(tf.matmul(h_fc1, W_fc2) + b_fc2)
    h_fc3 = tf.nn.sigmoid(tf.matmul(h_fc2, W_fc3) + b_fc3)
    h_fc4 = tf.nn.sigmoid(tf.matmul(h_fc3, W_fc4) + b_fc4)
    h_fc5 = tf.nn.sigmoid(tf.matmul(h_fc4, W_fc5) + b_fc5)
    

    h_fc6 = tf.nn.sigmoid(tf.matmul(h_fc5, W_fc6) + b_fc6)
    h_fc7 = tf.nn.sigmoid(tf.matmul(h_fc6, W_fc7) + b_fc7)
    h_fc8 = tf.nn.sigmoid(tf.matmul(h_fc7, W_fc8) + b_fc8)
    h_fc9 = tf.nn.sigmoid(tf.matmul(h_fc8, W_fc9) + b_fc9)
    h_fc10 = tf.nn.sigmoid(tf.matmul(h_fc9, W_fc10) + b_fc10)


    h_fc11 = tf.nn.sigmoid(tf.matmul(h_fc10, W_fc11) + b_fc11)
    h_fc12 = tf.nn.sigmoid(tf.matmul(h_fc11, W_fc12) + b_fc12)
    h_fc13 = tf.nn.sigmoid(tf.matmul(h_fc12, W_fc13) + b_fc13)
    h_fc14 = tf.nn.sigmoid(tf.matmul(h_fc13, W_fc14) + b_fc14)
    h_fc15 = tf.nn.sigmoid(tf.matmul(h_fc14, W_fc15) + b_fc15)



    h_fc16 = tf.nn.sigmoid(tf.matmul(h_fc15, W_fc16) + b_fc16)
    h_fc17 = tf.nn.sigmoid(tf.matmul(h_fc16, W_fc17) + b_fc17)
    h_fc18 = tf.nn.sigmoid(tf.matmul(h_fc17, W_fc18) + b_fc18)
    h_fc19 = tf.nn.sigmoid(tf.matmul(h_fc18, W_fc19) + b_fc19)

    #h_fc5_drop = tf.nn.dropout(h_fc5, keep_prob)

    # readout layer
    readout = tf.matmul(h_fc19, W_fc20) + b_fc20

    return s, readout 




def trainNetwork(s, readout, sess):
        # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices = 1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)




    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()

    # printing
    a_file = open("logs_" + GAME + "/readout.txt", 'w')
    h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 300x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal ,score = game_state.frame_step(do_nothing, 0)
    s_t = x_t.flatten()

   
    # saving and loading networks
    saver = tf.train.Saver(max_to_keep=0)
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path, file=sys.stderr)
    else:
        print("Could not find old network weights", file=sys.stderr)

    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    while "flappy bird" != "angry bird":
        # choose an action epsilon greedily
        readout_t = readout.eval(feed_dict={s : [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0

        if random.random() <= epsilon:
            print("----------Random Action----------", file=sys.stderr)
            action_index =random.randrange(ACTIONS)
            a_t[random.randrange(ACTIONS)] = 1
        else:
            action_index =np.argmax(readout_t)
            a_t[action_index] = 1


        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        x_t1 , r_t, terminal ,score = game_state.frame_step(a_t,t)
        s_t1 = x_t1.flatten()


        # store the transition in D
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # perform gradient step
            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch}
            )

        # update the old values
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = t )

        

        print( t, "/ E", epsilon, "/ A", action_index, "/ R", r_t, \
            "/ Q %e" % np.max(readout_t) ,"/ readout" ,readout_t[0] ,readout_t[1] ,readout_t[2] ,"/ SCR " ,score )
        # write info to files

        #h_file.write(str(s_t)+"\n")


    
def playGame():
    sess = tf.InteractiveSession()
    s, readout = createNetwork()
    trainNetwork(s, readout,  sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
