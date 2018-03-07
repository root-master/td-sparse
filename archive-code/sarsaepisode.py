import math
from scipy.spatial.distance import cdist
import numpy as np
from numpy import pi
from puddleworld import PuddleWorld
from agent import Agent


class SarsaEpisode:

    def __init__(self,
                agent,
                environment,
                epsilon,
                learning_rate):
        
        

        states, actions, means, errors, rewards, done \
        = self.runEpisode(agent,environment,epsilon,learning_rate)

        self.states = states
        self.actions = actions
        self.means = means
        self.errors = errors
        self.rewards = rewards
        self.done = done


    def runEpisode(self,agent,environment,epsilon,learning_rate):
        T = environment.maxStep
        nd = environment.nd
        nA = environment.nA

        states = np.zeros([T,nd])
        actions = np.zeros([T,nA])
        means = np.zeros([T,nA])
        errors = np.zeros([T,nA])
        rewards = np.zeros([T,1])
        done = False


        gamma = 0.99
        # initialize state
        s = agent.random_init_state(nd)
        Q = agent.valueFunction(s)
        a = agent.select_action(Q,epsilon)

        for t in range(environment.maxStep):
            states[t,:]  = s
            actions[t,:] = a
            #environment.success
            sp1, r, done = environment.update_state_env_reward(s,a)
            rewards[t,0] = r
            Qp1 = agent.valueFunction(sp1)
            ap1 = agent.select_action(Qp1,epsilon)
            if ~done:
                delta = r + gamma * Qp1[0,np.argmax(ap1)] - Q[0,np.argmax(a)]
            else:
                delta = r - Q[0,np.argmax(a)]

            errors[t,np.argmax(a)] = delta
            agent_network_error = np.zeros([1,4])
            agent_network_error[0,np.argmax(a)] = delta
            agent.model.update_network(agent.network_input(s),learning_rate,agent_network_error)

            if done:
                print('Successful Episode')
                break
            
            s, a, Q = sp1, ap1, Qp1
        return states, actions, means, errors, rewards, done

# sarsa = SarsaEpisode(agent,environment,epsilon=0.01,learning_rate=0.005)

