
import numpy as np
from numpy import pi
import tensorflow as tf
import matplotlib.pyplot as plt

###### CLASS puddleworld ######################################
import math
from scipy.spatial.distance import cdist
######
def closest_point(pt, others):
    distances = cdist(pt, others)
    return others[distances.argmin()]
        
def shortest_distance(pt, others):
        distances = cdist(pt, others)
        return np.min(distances)

class puddleworld:    
    def __init__(self):
        puddle = self.CreatePointOutsidePuddle()
        self.puddle = puddle
    
    def CreatePointOutsidePuddle(self):
        puddle = []
        # to find an accurate distance to edge mess is finer
        ngrid = [40, 40]
        x_vec = np.linspace(0,1,ngrid[0])
        y_vec = np.linspace(0,1,ngrid[1])
        for x in x_vec:
            for y in y_vec:
                if ~self.inpuddle([x,y]):
                    puddle.append([x,y])
        # puddle is a closed loop 
        outpuddlepts = np.asarray(puddle)
        return outpuddlepts


    def inpuddle(self, state):
        agentinPuddle = False
        # Horizontal wing of puddle consists of 
        # 1) rectangle area xch1<= x <=xc2 && ych1-radius <= y <=ych2+radius
        # (xchi,ychi) is the center points (h ==> horizantal)
        # x, y = state[0], state[1]
        x, y = state[0], state[1]
        xch1, ych1 = 0.3, 0.75
        xch2, ych2 = 0.65, ych1
        radius = 0.1
        inHorRec = (x>=xch1) and (y>= ych1-radius) and (x<=xch2)  and (y<=ych2+radius)   
        # 2) two half-circle at end edges of rectangle
        inHorCir1 = ( ( (x-xch1)**2 + (y-ych1)**2 <= radius**2 ) and x<xch1 )
        inHorCir2 = ( ((x-xch2)**2 + (y-ych2)**2) <= radius**2 and x>xch2 )
        inHor = inHorRec or inHorCir1 or inHorCir2

        #Vertical wing of puddle consists of 
        # 1) rectangle area xcv1-radius<= x <=xcv2+radius && ycv1 <= y <= ycv2
        # where (xcvi,ycvi) is the center points (v ==> vertical)
        xcv1 = 0.45; ycv1=0.4;
        xcv2 = xcv1; ycv2 = 0.8;

        inVerRec = (x >= xcv1-radius) and (y >= ycv1) and (x <= xcv2+radius) and (y <= ycv2)    
        # % 2) two half-circle at end edges of rectangle
        inVerCir1 = ( ( (x-xcv1)**2 + (y-ycv1)**2 <= radius**2 ) and y<ycv1 )
        inVerCir2 = ( ( (x-xcv2)**2 + (y-ycv2)**2 <= radius**2 ) and y>ycv2 )
        inVer = inVerRec or inVerCir1 or inVerCir2

        agentinPuddle = inHor or inVer

        return agentinPuddle

    def dist2edge(self, state):
        state =np.asarray([state])
        dist2edge = shortest_distance(state , self.puddle)
        return dist2edge
############################ END CLASS puddleworld ############################



########################### AGENT #############################################
def build_states_list(x_vec,y_vec):
    states_list = []
    for x in x_vec:
        for y in y_vec:
            states_list.append([x,y])
    return states_list

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


def normpdf(x_vec,mu,sigma):
    # normal probability distribution with mean mu and std sigma
    y = np.exp(- np.square(x_vec - mu) / (2 * sigma * sigma))
    return y

def encoder(state,discrete_state_vec):
    input = []
    i = 0
    for x in state:
        mu = x
        x_vec = discrete_state_vec[i]
        sigma = np.max(np.ediff1d(x_vec))
        i = i+1
        input.append(normpdf(x_vec,mu,sigma))
    #input.flatten()
    input = np.array(input)
    input = input.flatten()
    return input

def agent():
    return 0

def forward_path(x,w):
    a = tf.matmul(x, w[0]) + b[0]
    h = kWTA(a,k)
    y = tf.matmul(h, w[1]) + b[1]
    return y

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

##################### Environment #############################################
def environment(state, action):
        
    def update_state():
        inc_state = state + np.multiply(meshsize , action_list[np.argmax(action)] )
        state_update = np.minimum(inc_state , [1,1])
        state_update = np.maximum(inc_state, [0,0])
        return state_update

    def bumped2wall():
        inc_state = state + np.multiply(meshsize , action_list[np.argmax(action)] )
        bumped = False
        if (np.min(inc_state) < 0 or np.max(inc_state) > 1) :
            bumped = True
        return bumped


    def success():
        reach2goal = False
        if all(state_update == goal):
            reach2goal = True
        return reach2goal

    def env_reward():
        # cost of any action
        reward = -1
        if success():
            reward = 0
        elif bumped2wall():
            reward = -2
        elif inPuddle(state):
            reward = -400 * dist2Edge(state)
        
        return reward

    state_update = update_state()
    reach2goal = success()
    reward = env_reward()

    return state_update , reward, reach2goal


################ Episode ###################################################
def episode(agent,environment):
    T = maxStep

    states = np.zeros([T,nx])
    actions = np.zeros([T,no])
    means = np.zeros([T,no])
    errors = np.zeros([T,nx])
    rewards = np.zeros([T,1])

    # initialize state
    state = random_init_state(nd)
    # encoded state to feed to NN
    enc_state = encoder(state,discrete_state_vec)

    y = forward_path(x,w)
    feed={x: enc_state}
    # Q(s0,:)
    Q = sess.run(y,feed)
    action = select_action(nA,Q)


    for t in range(T):
        success = (state == goal)
        # collect (st,at)
        states[t,:] = state
        actions[t,:] = action
      




        if success:
            break




    return states, actions, means, errors, rewards


############ task related parameters --- 2D puddle world #####################
nd = 2    # number of dimensions of state space
goal = np.array([1.0,1.0])  # end state
# [min,max] of each dimension
bound = [[0,1] , [0,1]]
bound_x = [0.0,1.0]
bound_y = [0.0,1.0]
bound_state = (bound_x,bound_y)

# list of available actions for puddle world agent
# actions = ['UP' , 'DOWN' , 'RIGHT' , 'LEFT']
nA = 4
action_list = [[0,1],[0,-1],[1,0],[-1,0]]


############ Discretization of continouos state space ########################
ngridx = 10
ngridy = 10
ngrid = [ngridx,ngridy]
meshsize = [1/ngridx , 1/ngridy] # uniform meshing here
x_vec = np.linspace(bound_x[0],bound_x[1],ngridx+1)
y_vec = np.linspace(bound_y[0],bound_y[1],ngridy+1)

discrete_state_vec = (x_vec , y_vec)
states_list = build_states_list(x_vec,y_vec)
nStates = len(states_list)


############### Agent's Brain: netwrok parameters #############################
nx = len(x_vec) + len(y_vec) # input size after Gaussian Distribution
nh = round(0.5 * nStates)    # num hidden units
no = nA # size of output layer

k_rate = 0.1 # 10% of hidden  units are active for kWTA
k = round(k_rate*nh) # number of winner units


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
        y = forward_path(x,w)

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

    
    y = forward_path(x,w)
    feed={x: np.random.rand(1,22)}

    # print(sess.run(y,feed))
    # print(sess.run(w[0][1][:]))




   # Add ops to save and restore all the variables.
    # saver = tf.train.Saver(y)
    # saver.save(sess, 'my-model', global_step=1)



    
    
    