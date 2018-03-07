import numpy as np
from numpy import pi
import tensorflow as tf
import matplotlib.pyplot as plt
from puddleworld import PuddleWorld
from model import Model
from agent import Agent

########################### AGENT #############################################
"""
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
"""
"""
def normpdf(x_vec,mu,sigma):
    # normal probability distribution with mean mu and std sigma
    y = np.exp(- np.square(x_vec - mu) / (2 * sigma * sigma))
    return y

def encoder(state,discrete_state_vec):
    input_vec = []
    i = 0
    for x in state:
        mu = x
        x_vec = discrete_state_vec[dim_id]
        sigma = np.max(np.ediff1d(x_vec))
        dim_id += 1
        input_vec.append(normpdf(x_vec,mu,sigma))
    #input.flatten()
    input_vec = np.array(input)
    input_vec = input.flatten()
    return input_vec
"""


"""
def forward_path(x,w,b):
    a = tf.matmul(x, w[0]) + b[0]
    if kwta_use:
        a = kWTA(a,kwta_num)
    h = tf.sigmoid(a)
    y = tf.matmul(h, w[1]) + b[1]
    return y
"""


"""
def random_init_state(nd):
    return np.random.random(nd)

def select_action(Q):
    # greedy action
    act_index = np.argmax(Q)
    action = np.zeros(nA)
    # explore
    if np.random.random() < epsilon:
      act_index = np.random.randint(nA)
    
    action[act_index] = 1
    return action
"""





##################### Environment #############################################

environment = PuddleWorld()
agent = Agent(nd = environment.nd, nA = environment.nA, 
                discrete_state_vec = environment.discrete_state_vec)

################ Episode ###################################################

class Episode:
    pass



############ task related parameters --- 2D puddle world #####################
nd = 2    # number of dimensions of state space
goal = np.array([1.0,1.0])  # end state
# [min,max] of each dimension
bound = [[0,1] , [0,1]]
bound_x = [0.0,1.0]
bound_y = [0.0,1.0]
bound_state = [bound_x,bound_y]

# list of available actions for puddle world agent
# actions = ['UP' , 'DOWN' , 'RIGHT' , 'LEFT']
nA = 4
action_list = [[0,1],[0,-1],[1,0],[-1,0]]


############ Environment parameters ###########################################
############ Discretization of continouos state space #########################
ngridx = 10
ngridy = 10
ngrid = [ngridx,ngridy]
meshsize = [1/ngridx , 1/ngridy] # uniform meshing here
x_vec = np.linspace(bound_x[0],bound_x[1],ngridx+1)
y_vec = np.linspace(bound_y[0],bound_y[1],ngridy+1)

discrete_state_vec = np.array([x_vec , y_vec])
states_list = environment.states_list # build_states_list(x_vec,y_vec)
nStates = len(states_list)


############### Agent's Brain: netwrok parameters #############################
encoder_use = True

if encoder_use:
    nx = len(x_vec) + len(y_vec) # input size after Gaussian Distribution
else:
    nx = nd

nh = round(0.5 * nStates)    # num hidden units
no = nA # size of output layer

kwta_use = True
kwta_rate = 0.1 # 10% of hidden  units are active for kWTA
kwta_num = round(kwta_rate*nh) # number of winner units

encoder_use = True

############# Simulation parameters ###########################################
lr = 0.005 # learning rate
epsilon = 0.01 # exploration rate
batchsize = 1
# max step in each episode
maxstep = 40 


############# Agent creates TensorFlow Graph -- Neural Network ################
winit = tf.random_normal_initializer(mean=0.0, stddev=0.1, seed=None, dtype=tf.float32)
binit = tf.random_normal_initializer(mean=0.0, stddev=0.1, seed=None, dtype=tf.float32)
tf.reset_default_graph()

with tf.Graph().as_default() as graph:
    ################## Initilization and placeholding #########################
    w = []
    b = []
    with tf.variable_scope('policyparams'):
        w.append(tf.get_variable('wih', [nx, nh], initializer = winit))
        b.append(tf.get_variable('bih', [1, nh], initializer = binit))    
        w.append(tf.get_variable('who', [nh, no], initializer = winit))
        b.append(tf.get_variable('bho', [1, no], initializer = binit))
        x = tf.placeholder(tf.float32, [1,nx])

    ################## Forward path in tf graph ###############################
    with tf.name_scope('forwardpath'):
        x = tf.placeholder(tf.float32, [1,nx])
        y = forward_path(x,w,b)

    ################### Compute gradients for update weights ################## 
    with tf.name_scope('backward_pass'):
        grad_ys = tf.placeholder(tf.float32, [batchsize,no])
        gw = tf.gradients(y, w, grad_ys=grad_ys)
        gb = tf.gradients(y, b, grad_ys=grad_ys)
    update_w = []
    update_b = []
    with tf.name_scope('updates'):
        dw0 = tf.placeholder(tf.float32, [nx,nh])
        dw1 = tf.placeholder(tf.float32, [nh,no])
        db0 = tf.placeholder(tf.float32, [1,nh])
        db1 = tf.placeholder(tf.float32, [1,no])

    update_w.append(tf.assign_add(w[0], dw0))
    update_w.append(tf.assign_add(w[1], dw1))
    
    update_b.append(tf.assign_add(b[0], db0))
    update_b.append(tf.assign_add(b[1], db1))


"""

def train():
    N = 1000 # Number of learning Episodes

    performance = np.zeros([N,1])

    for n in range(N):
    
        # run the agent and environment forward:
        states, actions, means, errors, rewards = episode(agent)
    
        backward_pass(states, actions, means, errors, rewards, network_agent)
    
        performance[n,0] = rewards.sum()
        
    return performance

performance = train()

"""

with tf.Session(graph=graph) as sess:
  # Add an op to initialize the variables.
    init = tf.global_variables_initializer()
    sess.run(init)

    
    y = forward_path(x,w,b)
    feed={x: np.random.rand(1,22)}

    # print(sess.run(y,feed))
    # print(sess.run(w[0][1][:]))




   # Add ops to save and restore all the variables.
    # saver = tf.train.Saver(y)
    # saver.save(sess, 'my-model', global_step=1)



    
    
    