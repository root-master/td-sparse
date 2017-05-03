import numpy as np
from model import Model
from puddleworld import PuddleWorld


class Agent:
	
	def __init__(self,
					nd,
					nA,
					discrete_state_vec,
					encoder_use = True):
		# initilize agent critic network
		#self.model = Model(nx=42,nh=220,no=4,kwta_use = True,kwta_rate = 0.1,learning_rate=0.005)
		self.nd = nd
		self.nA = nA
		self.discrete_state_vec = discrete_state_vec
		self.encoder_use = encoder_use
		print('agent init: done.')

	def random_init_state(self,nd):
		 # s = np.random.random(nd)
		 s = np.zeros(nd)
		 for i in range(nd):
		 	s[i] = np.random.choice(self.discrete_state_vec[i]) 
		 return s

	def select_action(self,Q,epsilon):
		# greedy action
		act_index = np.argmax(Q)
		action = np.zeros(self.nA)
		# explore
		if np.random.random() < epsilon:
			act_index = np.random.randint(self.nA)

		action[act_index] = 1
		return action

	def normpdf(self,x_vec,mu,sigma):
		# normal probability distribution with mean mu and std sigma
		y = np.exp( - np.square(x_vec - mu) / (2 * sigma * sigma) )
		# y = y.reshape((1,y.size))

		return y

	def network_input(self,state):
		input_vec = []
		dim_id = 0
		for x in state:
		    mu = x		    
		    x_vec = self.discrete_state_vec[dim_id]
		    sigma = np.max(np.ediff1d(x_vec))
		    dim_id += 1
		    input_vec.append(self.normpdf(x_vec,mu,sigma))
		input_vec = np.array(input_vec)
		input_vec = np.reshape(input_vec,[1,input_vec.size])

		return input_vec


	def valueFunction(self,state):
		input_vec = self.network_input(state)
		Q = self.model.eval_output(input_vec)
		return Q
"""
p = PuddleWorld()
discrete_state_vec = p.discrete_state_vec
agent = Agent(nd=2,nA=4,discrete_state_vec=discrete_state_vec)
state = np.array([0,1])
print(state.shape)
print(agent.valueFunction(state).shape)


state = agent.random_init_state(2)
print(state.shape)
Q = agent.valueFunction(state)
action = agent.select_action(Q,0.1)
"""






