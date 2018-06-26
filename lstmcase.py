import os
import random
import numpy as np
import tensorflow as tf

from distribution import *
from readdata import dataproducer
from writer import *
    
def sample(prediction):
    d = sample_distribution(prediction)
    re = []
    re.append(d)
    return re

num_steps = 200
hidden_size = 500
keep_prob = 1.0
lr_decay = 0.5
batch_size = 20
num_layers = 4
max_epoch = 14
maxrange=1000
output_length = 50

model_path = "./writer/tmp/model.ckpt"
sentence_path = "./writer/tmp/sentence.txt"
filename = "./writer/noval001.txt"


x,y,id_to_word = dataproducer(batch_size, num_steps, filename)
vocab_size = len(id_to_word)

size = hidden_size

lstm_cell = tf.contrib.rnn.BasicLSTMCell(size, forget_bias = 0.5)
lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob = keep_prob)
cell = tf.contrib.rnn.MultiRNNCell([lstm_cell], num_layers)

initial_state = cell.zero_state(batch_size, tf.float32)
state =initial_state
embedding = tf.get_variable('embedding', [vocab_size, size])
input_data = x 
targets = y

test_input = tf.placeholder(tf.int32, shape=[1])
test_initial_state = cell.zero_state(1, tf.float32)

inputs = tf.nn.embedding_lookup(embedding, input_data)
test_inputs = tf.nn.embedding_lookup(embedding, test_input)

outputs = []
initializer = tf.random_uniform_initializer(-0.1,0.1)

saver = tf.train.Saver()



with tf.variable_scope("Model", reuse = None, initializer = initializer):
    with tf.variable_scope("r", reuse = None, initializer = initializer):
        softmax_w = tf.get_variable('softmax_w', [size, vocab_size])
        softmax_b = tf.get_variable('softmax_b', [vocab_size])
    with tf.variable_scope("RNN", reuse = None, initializer = initializer):
        for time_step in range(num_steps):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            (cell_output, state) = cell(inputs[:, time_step, :], state,)
            outputs.append(cell_output)
            
        output = tf.reshape(outputs, [-1,size])
        
        logits = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [tf.reshape(targets,[-1])], [tf.ones([batch_size*num_steps])])
        
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(10.0, global_step, 5000, 0.1, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        gradients, v = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
        optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)
        
        cost = tf.reduce_sum(loss) / batch_size
        #predict:
        teststate = test_initial_state
        (celloutput,teststate)= cell(test_inputs, teststate)
        partial_logits = tf.matmul(celloutput, softmax_w) + softmax_b
        partial_logits = tf.nn.softmax(partial_logits)
        
sv = tf.train.Supervisor(logdir=None)
with sv.managed_session() as session:
    costs = 0
    iters = 0

    if os.access(model_path+'.meta', os.W_OK):
        try:
            saver.restore(session, model_path)
            print("Model restored from file: %s" % model_path)
        except Exception as err:  
            print("Model not match, cannot be restored.")

    for i in range(maxrange+1):
        _,l= session.run([optimizer, cost])
        costs += l
        iters +=num_steps
        perplextity = np.exp(costs / iters)
        if i%50 == 0:
            print(perplextity)
        if i%100 == 0:
            p = random_distribution(vocab_size)
            b = sample(p)
            sentence = id_to_word[b[0]]
            for j in range(output_length):
                test_output = session.run(partial_logits, feed_dict={test_input:b})
                b = sample(test_output)
                sentence += id_to_word[b[0]]
            print(sentence)  
            # Save model weights to disk
            save_path = saver.save(session, model_path)
            print("Model saved in file: %s" % save_path)  
            writer(sentence_path, sentence, 'w')
            print("sentence saved in file: %s" % sentence_path) 
