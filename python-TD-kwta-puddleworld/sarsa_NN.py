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
nh_1 = len(env.states_list) // 2
nh_2 = len(env.states_list) // 2
no = env.nA

x = tf.placeholder(tf.float32,[1,nx])
y_ = tf.placeholder(tf.float32,[1,no])

dim_w = {}
dim_w['0_w_fc'] = [nx,nh_1]
dim_w['0_b_fc'] = [1,nh_1]
dim_w['1_w_fc'] = [nh_1,nh_2]
dim_w['1_b_fc'] = [1,nh_2]
dim_w['2_w_fc'] = [nh_2,no]
dim_w['2_b_fc'] = [1,no]

# w_initializer = tf.contrib.layers.xavier_initializer()
w_initializer = tf.random_uniform_initializer(-0.05, 0.05)

w = {}
for key, _ in dim_w.items():
	w[key] = tf.get_variable(key, shape=dim_w[key], 
											initializer=w_initializer)

w_init_values = {}
for key, _ in dim_w.items():
	w_init_values[key] = 0.1 * ( np.random.random(tuple(dim_w[key])) - 0.5 )

w_init_manual_place_holder = {}
for key, _ in dim_w.items():
	w_init_manual_place_holder[key] = tf.placeholder(tf.float32,dim_w[key])

w_initializer_manual = {}
for key, _ in dim_w.items():
	w_initializer_manual[key] = w[key].assign(w_init_manual_place_holder[key])

net_0 = tf.matmul(x, w['0_w_fc']) + w['0_b_fc']
act_0 = tf.sigmoid(net_0)

net_1 = tf.matmul(act_0, w['1_w_fc']) + w['1_b_fc']
act_1 = tf.sigmoid(net_1)

y = tf.matmul(act_1, w['2_w_fc']) + w['2_b_fc']

loss = tf.reduce_mean(tf.square(y - y_))
# loss = tf.losses.mean_squared_error(y, y_)
###############################################################################
################# Gradeints -- update weights of Q function ###################
###############################################################################
# I am not sure how this is going to turn out
# since kwta is hightly nonlinear
# gw = {}
# for layer, _ in w.items():
# 	gw[layer] = tf.gradients(xs=w[layer], ys=loss)

lr_tf = tf.placeholder(tf.float32)


# # method 2 to update weights -- both should work
# update_w = {}
# for layer, _ in w.items():
# 	update_w[layer] = w[layer].assign( w[layer] - lr_tf * gw[layer][0] )


# # method 3 to update weights -- ignore kwta gradient
# delta_tf = y - y_
# # # manual computation of gradient
# gw = {}
# gw['1_w_fc'] = tf.matmul(act_0,delta_tf,
# 				transpose_a=True,transpose_b=False)
# gw['1_b_fc'] = delta_tf
# delta_j = tf.multiply( tf.multiply( act_0,(1.0-act_0) ) , 
# 							tf.matmul( delta_tf , tf.transpose(w['1_w_fc']) ) )
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
trainer = tf.train.GradientDescentOptimizer(learning_rate=lr_tf)
update_w = trainer.minimize(loss)

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

	# # for method 1
	# w_val = sess.run(w)
	# gw_val = sess.run(gw,feed_dict=feed_dict)
	# new_w = {}
	# for layer, _ in w.items():
	# 	new_w[layer] = w_val[layer]-lr * gw_val[layer][0]
	# 	feed_dict.update({update_w_placeholder[layer]: new_w[layer]})

	sess.run(update_w, feed_dict=feed_dict)
	return

def w_init_manually(sess):
	feed_dict = {}
	for key, _ in dim_w.items():
		feed_dict.update({w_init_manual_place_holder[key]: w_init_values[key]})
	sess.run(w_initializer_manual, feed_dict=feed_dict)
	return


T = env.maxStep
nd = env.nd
nA = env.nA

gamma = 0.99
epsilon = 0.1	
lr = 0.005
max_episodes = 20000
consecutive_successful_episodes = 0
mean_errors = []
successful_episodes_list = []

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)	
	w_init_manually(sess)
	for e in range(max_episodes):
		states = np.zeros([T,nd])
		actions = np.zeros([T,nA])
		means = np.zeros([T,nA])
		errors = []
		rewards = []
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
			rewards.append(r)
			Qp1 = value_function(sess, sp1)
			ap1 = select_action(Qp1, epsilon)
			if ~done:
			    delta = r + gamma * Qp1[0,np.argmax(ap1)] - Q[0,np.argmax(a)]
			else:
			    delta = r - Q[0,np.argmax(a)]

			errors.append(delta)
			target = Q
			target[0,np.argmax(a)] += delta
			update_weights(sess, s, target, lr)

			if done:
				successful_episodes_list.append(e)
				print('Episode: {} s0: {} steps: {}' .format(e, s0, t) )			
				break
		
			s, a, Q = sp1, ap1, Qp1

		# modify exploration-vs-exploitation and learning rate
		mean_errors.append( np.mean(errors) )
		# if done:
		# 	epsilon = max(0.01, 0.999 * epsilon)
		# 	lr = max(0.0001, 0.999 * lr )
		# else:
		# 	epsilon = min(0.1, epsilon * 1.001)
		# 	lr = min(0.005, 1.001 * lr )

		if done and abs( np.mean(errors) ) < 0.2:
			epsilon = max(0.001, 0.999 * epsilon)
			# lr = max(1E-6, 0.99 * lr )
		else:
			epsilon = min(0.1, epsilon * 1.001)
			# lr = min(0.001, 1.01 * lr )

		# # method 2
		# if done:
		# 	epsilon = max(0.001, 0.999 * epsilon)
		# 	lr = max(1E-6, 0.999 * lr )
		# else:
		# 	epsilon = min(0.1, epsilon * 1.001)
		# 	lr = min(0.001, 1.001 * lr )

		# convergence condition
		if abs( np.mean(errors)) < 0.05 and done:
			consecutive_successful_episodes += 1
		else:
			consecutive_successful_episodes = 0

		convergence = consecutive_successful_episodes > 100
		if convergence:
			print('maybe convergence')
			break


		
		
		
