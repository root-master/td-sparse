import tensorflow as tf
import numpy as np
from numpy import pi

class Model:
	
	def __init__(self,
				nx=42,
				nh=220,
				no=4,
				kwta_use = False,
				kwta_rate = 0.1,
				learning_rate=0.005):
		self.nx = nx
		self.nh = nh
		self.no = no
		self.learning_rate = learning_rate

		self.kwta_use = kwta_use
		self.kwta_num =round(nh * kwta_rate)
		
		tf.reset_default_graph()	
		self.graph = tf.Graph()
		self.make_tensorflow_graph(nx,nh,no)
		
		# initilize weights and bias to evaluate numerically
		self.sess = tf.Session(graph=self.graph)
		with self.sess.as_default():
				self.sess.run(self.init_forward)
			
		print('model init: done')

		# self.make_tensorflow_graph(nx,nh,no)

	def make_tensorflow_graph(self,nx,nh,no):

		############# Agent creates TensorFlow Graph -- Neural Network ################
		winit = tf.random_normal_initializer(mean=0.0, stddev=0.1, seed=None, dtype=tf.float32)			
		binit = tf.random_normal_initializer(mean=0.0, stddev=0.0, seed=None, dtype=tf.float32)

		with self.graph.as_default():
			################## Initilization and placeholding #########################
			w = []
			b = []
			with tf.variable_scope('policyparams'):
				w.append(tf.get_variable('wih', [nx, nh], dtype=tf.float32, initializer = winit))
				b.append(tf.get_variable('bih', [1, nh], dtype=tf.float32, initializer = binit))    
				w.append(tf.get_variable('who', [nh, no], dtype=tf.float32, initializer = winit))
				b.append(tf.get_variable('bho', [1, no], dtype=tf.float32, initializer = binit))

			self.w = w
			self.b = b

			################## Forward path in tf graph ###############################
			with tf.name_scope('forwardpath'):
				x = tf.placeholder(tf.float32,[1,nx])
				a = tf.matmul(x, w[0]) + b[0]
				if self.kwta_use:n
					activity = self.kWTA(a,self.kwta_num)
				else:
					activity = a
				h = tf.sigmoid(activity)
				y = tf.matmul(h, w[1]) + b[1]

				# y = self.forward_path(x)

				self.x = x
				self.h = h
				self.y = y

			# Add an op to initialize the variables.
			self.init_forward = tf.global_variables_initializer()
			################### Compute gradients for update weights ################## 
			with tf.name_scope('backward_pass'):
				learning_rate_tf = tf.placeholder(tf.float32, shape=[])
				grad_ys = tf.placeholder(tf.float32, [1,no])
				gw_tf = tf.gradients(y, w, grad_ys=grad_ys)
				gb_tf = tf.gradients(y, b, grad_ys=grad_ys)

				delta = tf.placeholder(tf.float32, [1,no])
				self.delta = delta
				self.learning_rate_tf = learning_rate_tf
				error = tf.scalar_mul(learning_rate_tf, delta)

				gw_manual = []
				bw_manual = []

				
				grad_y_who = h
				grad_y_wih = []


				delta_j = tf.multiply( 
					tf.multiply( h,(1.0-h) ) , 
					tf.matmul( error , tf.transpose(w[1]) ) )
				
				# Update wheights
				delta_who = tf.matmul(tf.transpose(h), error)
				delta_wih = tf.matmul(tf.transpose(x) , delta_j )


				update_w = []
				update_b = []
			with tf.name_scope('updates'):
				dw0 = tf.placeholder(tf.float32, [nx,nh])
				dw1 = tf.placeholder(tf.float32, [nh,no])
				db0 = tf.placeholder(tf.float32, [1,nh])
				db1 = tf.placeholder(tf.float32, [1,no])

				dw0 = delta_wih
				dw1 = delta_who
				db0 = delta_j
				db1 = error

				update_w.append(tf.assign_add(w[0], dw0))
				update_w.append(tf.assign_add(w[1], dw1))
				update_b.append(tf.assign_add(b[0], db0))
				update_b.append(tf.assign_add(b[1], db1))

				self.update_w = update_w
				self.update_b = update_b

				
	def kWTA(self,a,k):
		with self.graph.as_default():
			shunt = 1
			# compute top k+1 and top k activated units in hidden layer
			a_kp1,id_kp1 = tf.nn.top_k(a, k=k+1, sorted=True, name=None)
			a_k,id_k = tf.nn.top_k(a, k=k, sorted=True, name=None)
			# now find the kth and (k+1)th activation 
			a_k_k = tf.reduce_min(a_k, reduction_indices=None, keep_dims=False, name=None)
			a_kp1_kp1 = tf.reduce_min(a_kp1, reduction_indices=None, keep_dims=False, name=None)
			# kWTA bias term
			q = 0.25
			bias_kWTA = a_kp1_kp1 + q * (a_k_k - a_kp1_kp1)
			a_kWTA = a - bias_kWTA - 1
		return a_kWTA
	"""
	def forward_path(self,x):
		a = tf.matmul(x, self.weights[0]) + self.bias[0]
		if self.kwta_use:
			activity = self.kWTA(a,self.kwta_num)
		else:
			activity = a
		h = tf.sigmoid(activity)
		y = tf.matmul(h, self.weights[1]) + self.bias[1]
		return y
	"""
	def eval_output(self,input_vec):
		
		#with self.sess.as_default():
		feed_dict = {self.x: input_vec}
		Q = self.sess.run(self.y, feed_dict = feed_dict)
		return Q

	def update_network(self,input_vec,lr,delta):
		#with self.sess.as_default():
			# update the weights
		feed_dict = {self.x: input_vec,
					self.learning_rate_tf:lr,
					self.delta:delta}
		update_weights_values = self.sess.run(self.update_w, feed_dict)
		update_bias_values = self.sess.run(self.update_b, feed_dict)

		self.w[0].assign(update_weights_values[0])
		self.w[1].assign(update_weights_values[1])
		self.b[0].assign(update_bias_values[0])
		self.b[1].assign(update_bias_values[1])

n = Model()
x = np.random.random([1,n.nx])

print(n.eval_output(x)) 
lr = 0.01
delta = np.random.random([1,n.no])
print(n.sess.run(n.w[0][1]))
n.update_network(x,lr,delta)
print(n.sess.run(n.w[0][1]))

