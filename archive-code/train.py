import numpy as np
from puddleworld import PuddleWorld
from agent import Agent
from sarsaepisode import SarsaEpisode

environment = PuddleWorld()
agent = Agent(nd = environment.nd, nA = environment.nA, 
                discrete_state_vec = environment.discrete_state_vec)


maxEpisodes = 20000

for e in range(maxEpisodes):
	print('Episode number: %d' % e)
	sarsa = SarsaEpisode(agent,environment,epsilon=0.1,learning_rate=0.005)
	if sarsa.done:
		print(sarsa.states)
