
import numpy as np
from numpy import pi
import tensorflow as tf
import matplotlib.pyplot as pp


# task related parameters --- 2D puddle world
nd = 2         # number of dimensions of state space
goal = np.array([1.0,1.0])
bound_x = [0.0,1.0]
bound_y = [0.0,1.0]
ngridx = 10
ngridy = 10
x_vec = np.linspace(bound_x[0],bound_x[1],ngridx+1)
y_vec = np.linspace(bound_y[0],bound_y[1],ngridy+1)

def build_states_list():
    states_list = []
    for x in x_vec:
        for y in y_vec:
            states_list.append([x,y])
    return states_list

states_list = build_states_list()
nStates = len(states_list)

# list of available actions
# actions = ['UP' , 'DOWN' , 'RIGHT' , 'LEFT']
actions_list = [0 , 1, 2, 3]
nA = len(actions_list)

# kwta network tensorflow graph
k = 0.1 # 10% of hidden  units are active for kWTA


# shape of netwrok
nx = len(x_vec) + len(y_vec) # input size after Gaussian Distribution
nh = round(0.5 * nStates)    # num hidden units
no = nA



# winit = tf.uniform_unit_scaling_initializer(0.1)
# https://www.tensorflow.org/api_guides/python/state_ops#Sharing_Variables
winit = tf.random_normal_initializer(mean=0.0, stddev=0.1, seed=None, dtype=tf.float32)
binit = tf.random_normal_initializer(mean=0.0, stddev=0.1, seed=None, dtype=tf.float32)

# number of winner units
k = round(k*nh) 
def kWTA(a,k):
  # compute top k+1 and top k activated units in hidden layer
  a_kp1,id_kp1 = tf.nn.top_k(a, k=k+1, sorted=True, name=None)
  a_k,id_k = tf.nn.top_k(a, k=k, sorted=True, name=None)

  # now find the kth and (k+1)th activation 
  a_k_k = tf.reduce_min(a_k, reduction_indices=None, keep_dims=False, name=None)
  a_kp1_kp1 = tf.reduce_min(a_kp1, reduction_indices=None, keep_dims=False, name=None)

  # kWTA bias term
  q = 0.25
  bias_kWTA = a_kp1_kp1 + q * (a_k_k - a_kp1_kp1)

  a_kWTA = a - bias_kWTA
  return a_kWTA

def forward_path(x,w):
    a = tf.matmul(x, w[0]) + b[0]
    h = kWTA(a,k)
    y = tf.matmul(h, w[1]) + b[1]
    return y





tf.reset_default_graph()
# tensorflow graph construction 
with tf.Graph().as_default() as graph:
# variable initializers
    w = []
    b = []
    with tf.variable_scope('policyparams'):
        w.append(tf.get_variable('wih', [nx, nh], initializer = winit))
        b.append(tf.get_variable('bih', [1, nh], initializer = binit))    
        w.append(tf.get_variable('who', [nh, no], initializer = winit))
        b.append(tf.get_variable('bho', [1, no], initializer = binit))
        x = tf.placeholder(tf.float32, [1,nx])


    with tf.name_scope('forwardpath'):
        y = forward_path(x,w)




with tf.Session(graph=graph) as sess:
    # Add an op to initialize the variables.
    init = tf.global_variables_initializer()
    sess.run(init)

    
    y = forward_path(x,w)
    feed={x: np.random.rand(1,22)}

    print(sess.run(y,feed))


    print(sess.run(w[0][1][:]))
    #print(sess.run(y , feed))
    


"""
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(y)
    saver.save(sess, 'my-model', global_step=1)
"""



    


