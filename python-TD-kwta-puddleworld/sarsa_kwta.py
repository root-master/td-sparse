import tensorflow as tf 
import numpy as np
from numpy import pi
import math
import copy
from random import random

import scipy.io as sio

import matplotlib.pyplot as plt


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

def compute_kWTA_bias(net,k):
	shunt = 1
	# compute top k+1 and top k activated units in hidden layer
	a_kp1,id_kp1 = tf.nn.top_k(net, k=k+1, sorted=True, name=None)
	a_k,id_k = tf.nn.top_k(net, k=k, sorted=True, name=None)
	# now find the kth and (k+1)th activation 
	a_k_k = tf.reduce_min(a_k, reduction_indices=None, 
										keep_dims=False, name=None)
	a_kp1_kp1 = tf.reduce_min(a_kp1, reduction_indices=None, 
										keep_dims=False, name=None)
	# kWTA bias term
	q = 0.25
	bias_kWTA = a_kp1_kp1 + q * (a_k_k - a_kp1_kp1)
	return bias_kWTA

def kWTA(net,k):
	shunt = 1
	# compute top k+1 and top k activated units in hidden layer
	a_kp1,id_kp1 = tf.nn.top_k(net, k=k+1, sorted=True, name=None)
	a_k,id_k = tf.nn.top_k(net, k=k, sorted=True, name=None)
	# now find the kth and (k+1)th activation 
	a_k_k = tf.reduce_min(a_k, reduction_indices=None, 
										keep_dims=False, name=None)
	a_kp1_kp1 = tf.reduce_min(a_kp1, reduction_indices=None, 
										keep_dims=False, name=None)
	# kWTA bias term
	q = 0.25
	bias_kWTA = a_kp1_kp1 + q * (a_k_k - a_kp1_kp1)
	# net after kwta
	net_kWTA = net - bias_kWTA - shunt
	return net_kWTA


x = tf.placeholder(tf.float32,[1,nx])
y_ = tf.placeholder(tf.float32,[1,no])

dim_w = {}
dim_w['0_w_fc'] = [nx,nh]
dim_w['0_b_fc'] = [1,nh]
dim_w['1_w_fc'] = [nh,no]
dim_w['1_b_fc'] = [1,no]

def random_np_array(shape_list):
	dim_1 = shape_list[0]
	dim_2 = shape_list[1]
	random_vec = np.zeros((dim_1,dim_2))
	for i in range(dim_1):
		for j in range(dim_2):
			random_vec[i,j] = 0.1 * ( random() - 0.5)
	return random_vec


# w_initializer = tf.contrib.layers.xavier_initializer()
w_initializer = tf.random_uniform_initializer(-0.05, 0.05)

w = {}
for key, _ in dim_w.items():
	w[key] = tf.get_variable(key, shape=dim_w[key], 
											initializer=w_initializer)

w_init_values = {}
for key, _ in dim_w.items():
	#w_init_values[key] = 0.1 * ( np.random.random(tuple(dim_w[key])) - 0.5 )
	w_init_values[key] = random_np_array(dim_w[key])

w_init_manual_place_holder = {}
for key, _ in dim_w.items():
	w_init_manual_place_holder[key] = tf.placeholder(tf.float32,dim_w[key])

w_initializer_manual = {}
for key, _ in dim_w.items():
	w_initializer_manual[key] = w[key].assign(w_init_manual_place_holder[key])


net_0 = tf.matmul(x, w['0_w_fc']) + w['0_b_fc']

# bias_kwta_tf = compute_kWTA_bias(net_0,kwta_num)
# bias_kwta_place_holder = tf.placeholder(tf.float32)
# net_after_kwta = net_0 - bias_kwta_place_holder - 1.0

net_after_kwta = kWTA(net_0, kwta_num)
act_after_kwta = tf.sigmoid(net_after_kwta)

act_no_kwta = tf.sigmoid(net_0)

# ignore kwta gradient
# act_0 = act_no_kwta + tf.stop_gradient( act_after_kwta - act_no_kwta )
# act_0 = act_no_kwta
act_0 = act_after_kwta

y = tf.matmul(act_0, w['1_w_fc']) + w['1_b_fc']

loss = tf.reduce_mean(tf.square(y - y_))
# loss = tf.losses.mean_squared_error(y, y_)r
###############################################################################
################# Gradeints -- update weights of Q function ###################
###############################################################################
# I am not sure how this is going to turn out
# since kwta is hightly nonlinear
gw = {}
for layer, _ in w.items():
	gw[layer] = tf.gradients(xs=w[layer], ys=loss)

lr_tf = tf.placeholder(tf.float32)


# # # method 2 to update weights -- both should work
update_w = {}
for layer, _ in w.items():
	update_w[layer] = w[layer].assign( w[layer] - lr_tf * gw[layer][0] )


# # method 3 to update weights -- ignore kwta gradient
# delta_tf = y_ - y
# error_tf = -delta_tf

# # error_tf = tf.placeholder(tf.float32,[1,no])
# # # manual computation of gradient
# gw = {}
# gw['1_w_fc'] = tf.matmul(act_0,error_tf,
# 				transpose_a=True,transpose_b=False)
# gw['1_b_fc'] = error_tf
# delta_j = tf.multiply( tf.multiply( act_0,(1.0-act_0) ) , 
# 							tf.matmul( error_tf , tf.transpose(w['1_w_fc']) ) )
# gw['0_w_fc'] = tf.matmul(x, delta_j,
# 				transpose_a=True,transpose_b=False)
# gw['0_b_fc'] = delta_j

# update_w = {}
# for layer, _ in w.items():
# 	update_w[layer] = w[layer].assign( w[layer] - lr_tf * gw[layer] )

# method 1 to update weights
# update_w_placeholder = {}
# for layer, _ in w.items():
# 	update_w_placeholder[layer] = tf.placeholder(tf.float32,dim_w[layer])
# update_w = {}
# for layer, _ in w.items():
# 	update_w[layer] = w[layer].assign( update_w_placeholder[layer] )


# # method 4
# trainer = tf.train.GradientDescentOptimizer(learning_rate=lr_tf)
# update_w = trainer.minimize(loss)

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
	if random() < epsilon:
		act_index = np.random.randint(nA)

	action[act_index] = 1
	return action

def normpdf(x_vec,mu,sigma):
	# normal probability distribution with mean mu and std sigma
	return np.exp(- np.square(x_vec - mu) / (2 * sigma * sigma))

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

	# # for method 1
	# w_val = sess.run(w)
	# gw_val = sess.run(gw,feed_dict=feed_dict)
	# new_w = {}
	# for layer, _ in w.items():
	# 	new_w[layer] = w_val[layer]-lr * gw_val[layer][0]
	# 	feed_dict.update({update_w_placeholder[layer]: new_w[layer]})

	sess.run(update_w, feed_dict=feed_dict)
	return

def update_weights_test(sess, s, error_vec, lr):	
	feed_dict = {x: network_input(s),
				 error_tf: -error_vec,
				 lr_tf: lr}

	sess.run(update_w, feed_dict=feed_dict)
	return


def w_init_manually(sess):
	feed_dict = {}
	for key, _ in dim_w.items():
		feed_dict.update({w_init_manual_place_holder[key]: w_init_values[key]})
	sess.run(w_initializer_manual, feed_dict=feed_dict)
	return

def test_if_random_works(epsilon=0.1,max_step=1000):
	count = 0
	for i in range(max_step):
		if random()<epsilon:
			count += 1
	return count/max_step

T = env.maxStep
nd = env.nd
nA = env.nA

gamma = 0.99

epsilon_max = 0.1	
epsilon_min = 0.001
epsilon = epsilon_max

lr_max = 0.005
lr_min = 0.0001
lr = lr_max
max_episodes = 20000
#max_episodes=1
consecutive_successful_episodes = 0
mean_errors = []
successful_episodes_list = []
init = tf.global_variables_initializer()

# with tf.Session() as sess:
# 	sess.run(init)	
# 	#w_init_manually(sess)
# 	for e in range(max_episodes):
# 		states = np.zeros([T,nd])
# 		actions = np.zeros([T,nA])
# 		means = np.zeros([T,nA])
# 		errors = []
# 		rewards = []
# 		done = False

# 		# initialize state
# 		s = random_init_state(nd)
# 		s0 = s
# 		Q = value_function(sess,s)
# 		a = select_action(Q,epsilon)

# 		for t in range(T):			
# 			states[t,:]  = s
# 			actions[t,:] = a
# 			#environment.success
# 			sp1, r, done = env.update_state_env_reward(s,a)
# 			rewards.append(r)
# 			Qp1 = value_function(sess, sp1)
# 			ap1 = select_action(Qp1, epsilon)
# 			if ~done:
# 			    delta = r + gamma * Qp1[0,np.argmax(ap1)] - Q[0,np.argmax(a)]
# 			else:
# 			    delta = r - Q[0,np.argmax(a)]

# 			errors.append(delta)
# 			error_vec = np.zeros(Q.shape)
# 			error_vec[0,np.argmax(a)] = delta

# 			target = np.copy(Q)
# 			target[0,np.argmax(a)] += delta
			
# 			update_weights(sess, s, target, lr)
# 			#update_weights_test(sess, s, error_vec, lr)
# 			if done:
# 				successful_episodes_list.append(e)
# 				print('Episode: {} s0: {} steps: {}' .format(e, s0, t) )			
# 				break
		
# 			s = np.copy(sp1)
# 			a = np.copy(ap1)
# 			Q = np.copy(Qp1)

# 		# modify exploration-vs-exploitation and learning rate
# 		mean_errors.append( np.mean(errors) )
# 		# if done:
# 		# 	epsilon = max(0.01, 0.999 * epsilon)
# 		# 	lr = max(0.0001, 0.999 * lr )
# 		# else:
# 		# 	epsilon = min(0.1, epsilon * 1.001)
# 		# 	lr = min(0.005, 1.001 * lr )

# 		if done and abs( np.mean(mean_errors) ) < 0.2:
# 			epsilon = max(0.001, 0.99 * epsilon)
# 			lr = max(1E-4, 0.99 * lr )
# 		else:
# 			epsilon = min(0.1, epsilon * 1.01)
# 			lr = min(0.001, 1.01 * lr )

# 		# # method 2
# 		# if done:
# 		# 	epsilon = max(0.001, 0.999 * epsilon)
# 		# 	lr = max(1E-6, 0.999 * lr )
# 		# else:
# 		# 	epsilon = min(0.1, epsilon * 1.001)
# 		# 	lr = min(0.001, 1.001 * lr )

# 		# convergence condition
# 		if abs( np.mean(errors)) < 0.05 and done:
# 			consecutive_successful_episodes += 1
# 		else:
# 			consecutive_successful_episodes = 0

# 		convergence = consecutive_successful_episodes > 100
# 		if convergence:
# 			print('maybe convergence')
# 			break

###############################################################################
##################### NUMPY VERSION ###########################################
###############################################################################

def compute_np_bias_kWTA(net,k):
	shunt = 1
	sorted_net = np.flip(np.sort(net),axis=1)
	net_k = sorted_net[0,k-1]
	net_kp1 = sorted_net[0,k]
	# kWTA bias term
	q = 0.25
	bias_kWTA = net_kp1 + q * (net_k - net_kp1)
	return bias_kWTA


def np_kWTA(net,k):
	shunt = 1
	sorted_net = np.flip(np.sort(net),axis=1)
	net_k = sorted_net[0,k-1]
	net_kp1 = sorted_net[0,k]
	# kWTA bias term
	q = 0.25
	bias_kWTA = net_kp1 + q * (net_k - net_kp1)
	# net after kwta
	net_kWTA = net - bias_kWTA - shunt
	return net_kWTA

# load from matlab weighst
mat_weights_dict = sio.loadmat('./weights.mat')

# w_init_values = {}
# w_init_values['1_w_fc'] = mat_weights_dict['Who']
# w_init_values['0_w_fc'] = mat_weights_dict['Wih']
# w_init_values['1_b_fc'] = mat_weights_dict['biasho']
# w_init_values['0_b_fc'] = mat_weights_dict['biasih']


w_np = copy.copy(w_init_values)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def forward_np(s):
	global w_np
	x_np = network_input(s)
	net_0_np = x_np @ w_np['0_w_fc'] + w_np['0_b_fc']
	net_0_kwta = np_kWTA(net_0_np,kwta_num)
	act_0_np = sigmoid(net_0_kwta)
	output_np = act_0_np @ w_np['1_w_fc'] + w_np['1_b_fc']
	return output_np

def update_np(s,delta_vec,lr):
	global w_np
	x_np = network_input(s)
	net_0_np = x_np @ w_np['0_w_fc'] + w_np['0_b_fc']
	net_0_kwta = np_kWTA(net_0_np,kwta_num)
	act_0_np = sigmoid(net_0_kwta)
	output_np = act_0_np @ w_np['1_w_fc'] + w_np['1_b_fc']

	error_vec = -delta_vec
	gw_np = {}
	gw_np['1_w_fc'] = act_0_np.T @ error_vec
	gw_np['1_b_fc'] = error_vec
	delta_j_np = ( error_vec @ w_np['1_w_fc'].T ) * act_0_np * (1.0-act_0_np)
	gw_np['0_b_fc'] = delta_j_np
	gw_np['0_w_fc'] = x_np.T @ delta_j_np
	
	for key, matrix in w_np.items():
		w_np[key] = w_np[key] - lr * gw_np[key]
convergence = False
e = 0
while (e < max_episodes and not convergence):
	states = np.zeros([T,nd])
	actions = np.zeros([T,nA])
	means = np.zeros([T,nA])
	errors = []
	rewards = []
	done = False

	# initialize state
	s = random_init_state(nd)
	s0 = s
	Q = forward_np(s)
	a = select_action(Q,epsilon)
	t = 0	
	while(t < T and not env.success(s) ):
		# states[t,:]  = s
		# actions[t,:] = a
		#environment.success
		sp1, r, done = env.update_state_env_reward(s,a)
		
		rewards.append(r)
		Qp1 = forward_np(sp1)
		ap1 = select_action(Qp1, epsilon)
		if ~done:
			delta = r + gamma * Qp1[0,np.argmax(ap1)] - Q[0,np.argmax(a)]
			errors.append(delta)
			delta_vec = np.zeros_like(Q)
			delta_vec[0,np.argmax(a)] = delta
			update_np(s,delta_vec,lr)
		else:
			delta = r - Q[0,np.argmax(a)]
			delta_vec = np.zeros_like(Q)
			delta_vec[0,np.argmax(a)] = delta
			update_np(s,delta_vec,lr)

		# target = np.copy(Q)
		# target[0,np.argmax(a)] += delta
		
		
		if done:
			successful_episodes_list.append(e)
			print('Episode: {} s0: {} steps: {}' .format(e, s0, t) )			
		t += 1
	
		s = copy.copy(sp1)
		a = copy.copy(ap1)
		Q = copy.copy(Qp1)

	# modify exploration-vs-exploitation and learning rate
	mean_errors.append( np.mean(errors) )

	if done and abs( mean_errors[-1] ) < 0.2:
		epsilon = max(epsilon_min, 0.999 * epsilon)
		#lr = max(lr_min, 0.999 * lr )
	else:
		epsilon = min(epsilon_max, epsilon * 1.01)
		#lr = min(lr_max, 1.0001 * lr )

	# convergence condition
	if abs( np.mean(mean_errors)) < 0.05 and done:
		consecutive_successful_episodes += 1
	else:
		consecutive_successful_episodes = 0

	convergence = consecutive_successful_episodes > 100
	if convergence:
		print('maybe convergence')

	e += 1



# with tf.Session() as sess:
# 	sess.run(init)	
# 	w_init_manually(sess)

# 	# initialize state
# 	s = random_init_state(nd)
# 	net0_val = sess.run(net_0,feed_dict={x:network_input(s)})
# 	kwta_bias_val_np = compute_np_bias_kWTA(net0_val,kwta_num)
# 	kwta_bias_val_tf = sess.run( compute_kWTA_bias(net_0,kwta_num), 
# 				feed_dict= {x:network_input(s)})
# 	num_winner = (net0_val>kwta_bias_val_np).sum()

		
		
		
