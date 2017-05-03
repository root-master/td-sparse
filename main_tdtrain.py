import numpy as np
import tensorflow as tf
from puddleworld import PuddleWorld
from agent import Agent
from sarsaepisode import SarsaEpisode
from timeit import default_timer as timer

# fix random seed for reproducibility
seed = 43
np.random.seed(seed)

def kWTA(a,k):
	"""
	k-winner-take-all function
	args: 
		a: tensor
		k: number of neurons that should remain hight than threshold 

	returns: 
		a_kWTA: tensor that only "kwta_num" winner neurons have high 
		activity and the rest of neurons are losers (having less activities)
	"""
	
	# compute top "k+1" activated units in hidden layer
	a_kp1,id_kp1 = tf.nn.top_k(a, k=k+1, sorted=True, name=None)	
	# compute top k+1 and top k activated units in hidden layer
	a_k,id_k = tf.nn.top_k(a, k=k, sorted=True, name=None)
	# now find the kth and (k+1)th activation 
	a_k_k = tf.reduce_min(a_k, reduction_indices=None, keep_dims=False, name=None)
	a_kp1_kp1 = tf.reduce_min(a_kp1, reduction_indices=None, keep_dims=False, name=None)
	# a coefficient
	q = 0.25
	# kWTA bias term
	bias_kWTA = a_kp1_kp1 + q * (a_k_k - a_kp1_kp1)	
	# a constant to even make activitiy of neurons weaker
	shunt = 1
	a_kWTA = a - bias_kWTA - shunt
	return a_kWTA


environment = PuddleWorld()
agent = Agent(nd = environment.nd, nA = environment.nA, 
                discrete_state_vec = environment.discrete_state_vec)
# size of network input
nx = 42
# size of hidden layer
nh = round( environment.nStates / 2) + 1
# size of output
no = environment.nA
# k parameter of k-Winner-Take-All mechanism
k = 0.1
# the number of Winner neurons
kwta_num = round(k*nh)

# discount factor for SARSA delta
gamma = 0.99

# exloration rate
epsilon = 0.1

# learning rate
lr = 0.005

# maxEpisode to quit simulation if convergence doesn't occur
maxEpisodes = 55000
       
tf.reset_default_graph()
graph = tf.Graph()

with graph.as_default():
	with tf.name_scope('forward_pass'):
		Wih = tf.Variable(tf.truncated_normal([nx,nh],mean=0.0,stddev=0.1))
		bih = tf.Variable(tf.truncated_normal([1,nh],mean=0.0,stddev=0.1))
		Who = tf.Variable(tf.truncated_normal([nh,no],mean=0.0,stddev=0.1))
		bho = tf.Variable(tf.truncated_normal([1,no],mean=0.0,stddev=0.1))

		x_tf = tf.placeholder(shape=[1,nx],dtype=tf.float32)
		net = tf.matmul(x_tf, Wih) + bih
		net = kWTA(net,k=kwta_num)
		h = tf.sigmoid(net)
		Q_tf = tf.matmul(h, Who) + bho

	with tf.name_scope('backward-path-manual'):
		
		error_tf = tf.placeholder(tf.float32, [1,no])
		delta_j = tf.multiply( 
			tf.multiply( h,(1.0-h) ) , 
			tf.matmul( error_tf , tf.transpose(Who) ) )
		
		# delta of weights and biases
		delta_Who = tf.matmul(tf.transpose(h), error_tf)
		delta_Wih = tf.matmul(tf.transpose(x_tf) , delta_j )
		delta_bho = error_tf
		delta_bih = delta_j

		# update the weights
		update=[tf.assign(Who, tf.add(Who , delta_Who)),
				tf.assign(Wih, tf.add(Wih , delta_Wih)),
				tf.assign(bih, tf.add(bih , delta_bih)),
				tf.assign(bho, tf.add(bho , delta_bho))]
		
	init = tf.global_variables_initializer()


with tf.Session(graph=graph) as sess:
	sess.run(init)
	mean_error = []
	nGoodEpisodes = 0
	convergence = False
	####### EPISODE LOOP #############################################################
	for e in range(maxEpisodes):
		states = []
		actions = []
		errors = []
		rewards = []
		# initialize state
		s = agent.random_init_state(environment.nd)
		feed_dict = {x_tf: agent.network_input(s)}
		# value fucntion Q(s,a)
		Q = sess.run(Q_tf,feed_dict)
		# select an epsilon-greedy action "a"
		a = agent.select_action(Q,epsilon)
		#### STEPS ##################################################
		for t in range(environment.maxStep):
			states.append(s)
			actions.append(a)
			if environment.success(s):
				break
			
			# update state to sp1, get reward r, and check if agent reached to the goal
			sp1, r, done = environment.update_state_env_reward(s,a)
			rewards.append(r)
			# dictionary input to the network
			feed_dict = {x_tf: agent.network_input(sp1)}
			# value fucntion Q(s',a')
			Qp1 = sess.run(Q_tf,feed_dict)
			# select an epsilon-greedy action "a'"
			ap1 = agent.select_action(Qp1,epsilon)
			# compute Temporal Difference error delta
			if ~done:
			    delta = r + gamma * Qp1[0,np.argmax(ap1)] - Q[0,np.argmax(a)]
			else:
			    delta = r - Q[0,np.argmax(a)]

			errors.append(delta)
			
			# compute error signal to network
			agent_network_error = np.zeros([1,4])
			agent_network_error[0,np.argmax(a)] = delta
			agent_network_error = agent_network_error * lr
			
			# update weights for Q(s,a)			
			feed_dict={x_tf: agent.network_input(s), error_tf: agent_network_error }
			update_v = sess.run(update,feed_dict)

			# update state, action, value						
			s, a, Q = sp1, ap1, Qp1
		####### END EPISODE LOOP ########################################################

		if len(errors) > 0:
			mean_error.append(np.mean(errors))

		if done:
			print('Episode number {} success was {} and mean error is {}' .format(e,done,np.mean(mean_error)))

		# decrease exploration if agent is succeeding 0.001<= epilson<= 0.1
		if done:
			epsilon = epsilon * 0.999
			epsilon = np.min( [epsilon , 0.1] )
			epsilon = np.max( [epsilon , 0.001] )
		else: # increase exploration if agnet fails
			epsilon = epsilon * 1.001
			epsilon = np.min( [epsilon, 0.1] )
			epsilon = np.max( [epsilon , 0.001] )

		
		# count good episodes with small error    
		if ( np.mean(mean_error)<0.1 and done ):
			nGoodEpisodes += 1
		else:
			nGoodEpisodes = 0

		# convergence conditions. break if convergence happens 
		if ( np.mean(mean_error)<0.05 and nGoodEpisodes > environment.nStates ):
			convergence = True
			break

	print('Convergence of netwrok: {}' .format(convergence))
			

		






