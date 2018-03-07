import tensorflow as tf 
import numpy as np
from numpy import pi

from puddleworld import PuddleWorld 
tf.reset_default_graph()

env = PuddleWorld(nd = 2, 
                goal = np.array([1.0,1.0]),
                state_bound = [[0,1],[0,1]],
                nA = 4,
                action_list = [[0,1],[0,-1],[1,0],[-1,0]],
                ngrid = [20.0,20.0],
                maxStep = 80) 

###############################################################################
################# tensorflow model -- Q function ##############################
###############################################################################

nx = env.input_size
nh = len(env.states_list) // 2
no = env.nA

kwta_rate = 0.1
kwta_num = round(kwta_rate * nh)

def kWTA(a,k):
	shunt = 1
	# compute top k+1 and top k activated units in hidden layer
	a_kp1,id_kp1 = tf.nn.top_k(a, k=k+1, sorted=True, name=None)
	a_k,id_k = tf.nn.top_k(a, k=k, sorted=True, name=None)
	# now find the kth and (k+1)th activation 
	a_k_k = tf.reduce_min(a_k, reduction_indices=None, 
										keep_dims=False, name=None)
	a_kp1_kp1 = tf.reduce_min(a_kp1, reduction_indices=None, 
										keep_dims=False, name=None)
	# kWTA bias term
	q = 0.25
	bias_kWTA = a_kp1_kp1 + q * (a_k_k - a_kp1_kp1)
	a_kWTA = a - bias_kWTA - shunt
	return a_kWTA

x = tf.placeholder(tf.float32,[1,nx])
y_ = tf.placeholder(tf.float32,[1,no])

dim_w = {}
dim_w['0_w_fc'] = [nx,nh]
dim_w['0_b_fc'] = [1,nh]
dim_w['1_w_fc'] = [nh,no]
dim_w['1_b_fc'] = [1,no]

# w_initializer = tf.contrib.layers.xavier_initializer()
w_initializer = tf.random_uniform_initializer(-0.1, 0.1)

w = {}
for key, _ in dim_w.items():
	w[key] = tf.get_variable(key, shape=dim_w[key], 
											initializer=w_initializer)

net_0 = tf.matmul(x, w['0_w_fc']) + w['0_b_fc']
net_after_kwta = kWTA(net_0, kwta_num)
act_after_kwta = tf.sigmoid(net_after_kwta)

act_no_kwta = tf.sigmoid(net_0)

# ignore kwta gradient
# act_0 = act_no_kwta + tf.stop_gradient( act_after_kwta - act_no_kwta )

act_0 = act_after_kwta

y = tf.matmul(act_0, w['1_w_fc']) + w['1_b_fc']

loss = 1/2 * tf.reduce_mean(tf.square(y - y_))
#loss = tf.losses.mean_squared_error(y, y_)
###############################################################################
################# Gradeints -- update weights of Q function ###################
###############################################################################
# I am not sure how this is going to turn out
# since kwta is hightly nonlinear
gw = {}
for layer, _ in w.items():
	gw[layer] = tf.gradients(xs=w[layer], ys=loss)

lr_tf = tf.placeholder(tf.float32)

# method 1 to update weights
# update_w_placeholder = {}
# for layer, _ in w.items():
# 	update_w_placeholder[layer] = tf.placeholder(tf.float32,dim_w[layer])
# update_w = {}
# for layer, _ in w.items():
# 	update_w[layer] = w[layer].assign( update_w_placeholder[layer] )

# method 2 to update weights -- both should work
update_w = {}
for layer, _ in w.items():
	update_w[layer] = w[layer].assign( w[layer] - lr_tf * gw[layer][0] )


# todo
# method 3 to update weights -- ignore kwta gradient
# manual computation of gradient


def random_init_state(nd):
	# s = np.random.random(nd)
	s = np.zeros(nd)
	for i in range(nd):
		s[i] = np.random.choice(env.discrete_state_vec[i]) 
	return s

def select_action(Q,epsilon):
	# greedy action
	nA = Q.size
	act_index = np.argmax(Q)
	action = np.zeros(nA)
	# explore
	if np.random.random() < epsilon:
		act_index = np.random.randint(nA)

	action[act_index] = 1
	return action

def normpdf(x_vec,mu,sigma):
	# normal probability distribution with mean mu and std sigma
	y = np.exp(- np.square(x_vec - mu) / (2 * sigma * sigma))
	# y = y.reshape((1,y.size))
	return y

def network_input(state):
	input_vec = []
	dim_id = 0
	for xi in state:
		mu = xi
		x_vec = env.discrete_state_vec[dim_id]
		sigma = np.max(np.ediff1d(x_vec))
		dim_id += 1
		input_vec.append(normpdf(x_vec,mu,sigma))
	input_vec = np.array(input_vec)
	input_vec = np.reshape(input_vec,[1,input_vec.size])

	return input_vec

def value_function(sess, s):
	feed_dict = {x:network_input(s)}
	return sess.run(y,feed_dict=feed_dict)

def update_weights(sess, s, target, lr):
	
	feed_dict = {x: network_input(s),
				 y_: target,
				 lr_tf: lr}

	# for method 1
	# w_val = sess.run(w)
	# gw_val = sess.run(gw,feed_dict=feed_dict)
	# new_w = {}
	# for layer, _ in w.items():
	# 	new_w[layer] = w_val[layer]-lr * gw_val[layer][0]
	# 	feed_dict.update({update_w_placeholder[layer]: new_w[layer]})

	sess.run(update_w, feed_dict=feed_dict)
	return


T = env.maxStep
nd = env.nd
nA = env.nA

gamma = 0.99
epsilon = 0.1	
lr = 0.005
max_episodes = 4000

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)	
	for e in range(max_episodes):
		states = np.zeros([T,nd])
		actions = np.zeros([T,nA])
		means = np.zeros([T,nA])
		errors = np.zeros([T])
		rewards = np.zeros([T,1])
		done = False

		# initialize state
		s = random_init_state(nd)
		s0 = s
		Q = value_function(sess,s)
		a = select_action(Q,epsilon)

		for t in range(T):			
			states[t,:]  = s
			actions[t,:] = a
			#environment.success
			sp1, r, done = env.update_state_env_reward(s,a)
			rewards[t,0] = r
			Qp1 = value_function(sess,sp1)
			ap1 = select_action(Qp1,epsilon)
			if ~done:
			    delta = r + gamma * Qp1[0,np.argmax(ap1)] - Q[0,np.argmax(a)]
			else:
			    delta = r - Q[0,np.argmax(a)]

			errors[t] = delta
			target = Q
			target[0,np.argmax(a)] += delta
			update_weights(sess, s, target, lr)

			if done:
				print('Episode: {} s0: {} steps: {}' .format(e, s0, t) )
				break

			s, a, Q = sp1, ap1, Qp1
