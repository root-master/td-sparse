import numpy as np
import tensorflow as tf
from puddleworld import PuddleWorld
from agent import Agent
from sarsaepisode import SarsaEpisode
from timeit import default_timer as timer

def kWTA(a,k):
	shunt = 0
	# compute top k+1 and top k activated units in hidden layer
	a_kp1,id_kp1 = tf.nn.top_k(a, k=k+1, sorted=True, name=None)
	a_k,id_k = tf.nn.top_k(a, k=k, sorted=True, name=None)
	# now find the kth and (k+1)th activation 
	a_k_k = tf.reduce_min(a_k, reduction_indices=None, keep_dims=False, name=None)
	a_kp1_kp1 = tf.reduce_min(a_kp1, reduction_indices=None, keep_dims=False, name=None)
	# kWTA bias term
	q = 0.25
	bias_kWTA = a_kp1_kp1 + q * (a_k_k - a_kp1_kp1)
	a_kWTA = a - bias_kWTA - shunt
	return a_kWTA

tf.reset_default_graph()
graph = tf.Graph()

with graph.as_default():
	x = tf.placeholder(shape=[7],dtype=tf.float32)
	#h = tf.multiply(x,(1-x))
	h = kWTA(x,2)
	h = tf.sigmoid(h)

with tf.Session(graph=graph) as sess:
	feed_dict = {x: np.array([1,2,3,4,5,6,7])}
	aa = sess.run(h,feed_dict)
	print(aa)