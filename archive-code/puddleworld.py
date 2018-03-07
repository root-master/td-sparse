###### CLASS puddleworld ######################################
import math
from scipy.spatial.distance import cdist
import numpy as np
from numpy import pi
######
def closest_point(pt, others):
    distances = cdist(pt, others)
    return others[distances.argmin()]
        
def shortest_distance(pt, others):
        distances = cdist(pt, others)
        return np.min(distances)

# each row is corresponding discretized state
def discretizatize_state_vector(state_bound, ngrid):
    discrete_vec = []
    dim_id = 0
    for bound in state_bound:
        x_vec = np.linspace(state_bound[dim_id][0],state_bound[dim_id][1],ngrid[dim_id]+1)
        dim_id += 1
        discrete_vec.append(x_vec)
    return discrete_vec

# I don't know how to write this function to generate any list for any 
# nd dimension
def build_states_list(discrete_vec):
    states_list = []
    x_vec = discrete_vec[0]
    y_vec = discrete_vec[1]
    for x in x_vec:
        for y in y_vec:
            states_list.append([x,y])
    return np.array(states_list)


class PuddleWorld:    

    def __init__(self, 
                nd = 2, 
                goal = np.array([1.0,1.0]),
                state_bound = [[0,1],[0,1]],
                nA = 4,
                action_list = [[0,1],[0,-1],[1,0],[-1,0]],
<<<<<<< HEAD:archive-code/puddleworld.py
                ngrid = [10.0,10.0],
                maxStep = 40):
        """
=======
                ngrid = [20.0,20.0],
                maxStep = 80):
>>>>>>> 6bc271447fe7a11d457d422a8629e4802fc0c18a:puddleworld.py
        # Parameters should be in a parameter list or sth?
        nd = 2 # DOF
        goal = np.array([1.0,1.0])  # end state
        # [min,max] of each dimension
        state_bound = [[0,1] , [0,1]]
        # list of available actions for puddle world agent
        # actions = ['UP' , 'DOWN' , 'RIGHT' , 'LEFT']
        nA = 4
        action_list = [[0,1],[0,-1],[1,0],[-1,0]]
        ############ Discretization of continouos state space ########################
        meshsize = [1/ngrid[0] , 1/ngrid[1]] # uniform meshing here
        discrete_state_vec = discretizatize_state_vector(state_bound, ngrid)
        states_list =  build_states_list(discrete_state_vec)
        
        self.EnvironmentName = 'Puddle World' 
        self.nd = nd
        self.nA = nA
        self.goal = goal
        self.action_list = action_list
        self.state_bound = state_bound
        self.discrete_state_vec = discrete_state_vec
        self.states_list = states_list
        self.ngrid = ngrid
        self.meshsize = meshsize
        self.maxStep = maxStep
        self.nStates = states_list.shape[0]

        puddle = self.createPointsOutsidePuddle()
        self.puddle = puddle
        
        print('Environment: ' + self.EnvironmentName)
        
    
    # I don't know how to extract the coordinate of the boundary of puddle
    # if you have time create a function:
    # def boundary(list_of_points):
    #    ...
    # return boundary_points

    # return array of points outside of puddle -- these are not necessarily state
    # actually finer grid used to find accuare distance of agent to edge of puddle
    def createPointsOutsidePuddle(self):
        puddle = []
        # to find an accurate distance to edge mess is finer
        """
        ngrid = [40, 40]
        x_vec = np.linspace(0,1,ngrid[0])
        y_vec = np.linspace(0,1,ngrid[1])
        for x in x_vec:
            for y in y_vec:
                if ~self.inPuddle([x,y]):
                    puddle.append([x,y])
        # puddle is a closed loop 
        outpuddlepts = np.asarray(puddle)
        """


        # Horizontal wing of puddle consists of 
        # 1) rectangle area xch1<= x <=xc2 && ych1-radius <= y <=ych2+radius
        # (xchi,ychi) is the center points (h ==> horizantal)
        # x, y = state[0], state[1]
        xch1, ych1 = 0.3, 0.7
        xch2, ych2 = 0.65, ych1
        radius = 0.1


        #Vertical wing of puddle consists of 
        # 1) rectangle area xcv1-radius<= x <=xcv2+radius && ycv1 <= y <= ycv2
        # where (xcvi,ycvi) is the center points (v ==> vertical)
        xcv1 = 0.45; ycv1=0.4;
        xcv2 = xcv1; ycv2 = 0.8;

        # % 2) two half-circle at end edges of rectangle
        
        # POINTS ON HORIZANTAL LINES OF PUDDLE BOUNDARY
        for x in np.arange(xch1,xcv1-radius,self.meshsize[0]/2):
            puddle.append([x,ych1-radius])
        puddle.append([xcv1-radius,ych1-radius])
        
        for x in np.arange(xcv1+radius,xch2,self.meshsize[0]/2):
            puddle.append([x,ych1-radius])
        
        for x in np.arange(xch1,xcv1-radius,self.meshsize[0]/2):
            puddle.append([x,ych1+radius])
        
        puddle.append([xcv1-radius,ych1+radius])


        for x in np.arange(xcv1+radius,xch2,self.meshsize[0]/2):
            puddle.append([x,ych1+radius])

        # POINTS ON VERTICAL LINES OF PUDDLE BOUNDARY
        for y in np.arange(ycv1,ych1-radius,self.meshsize[1]/2):
            puddle.append([xcv1-radius,y])
        
        for y in np.arange(ycv1,ych1-radius,self.meshsize[1]/2):
            puddle.append([xcv1+radius,y])
        """
        for y in np.arrange():
            puddle.append([])
        
        for y in np.arrange():
            puddle.append([])
        """

        # HALF CIRCLES
        ngridTheta = 10
        thetaVec = np.linspace(0,pi,ngridTheta)

        for t in thetaVec:
            puddle.append([xch1+radius*np.cos(pi/2+t),ych1+radius*np.sin(pi/2+t)])

        for t in thetaVec:
            puddle.append([xch2+radius*np.cos(-pi/2+t),ych2+radius*np.sin(-pi/2+t)])

        for t in thetaVec:
            puddle.append([xcv1+radius*np.cos(pi+t),ycv1+radius*np.sin(pi+t)])

        for t in thetaVec:
            puddle.append([xcv2+radius*np.cos(t),ycv2+radius*np.sin(t)])

        
        outpuddlepts = np.asarray(puddle)
        return outpuddlepts

    # return true if agent is in puddle
    def inPuddle(self, state):
        agentinPuddle = False
        # Horizontal wing of puddle consists of 
        # 1) rectangle area xch1<= x <=xc2 && ych1-radius <= y <=ych2+radius
        # (xchi,ychi) is the center points (h ==> horizantal)
        # x, y = state[0], state[1]
        x, y = state[0], state[1]
        xch1, ych1 = 0.3, 0.7
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

    def update_state_env_reward(self, state, action):
        inc_state = state
        inc_state = state + np.multiply(self.meshsize , self.action_list[np.argmax(action)] )
        state_bound = self.state_bound
        state_update = np.minimum(inc_state , [1,1])
        state_update = np.maximum(state_update, [0,0])

        done = False
        # cost of any action
        reward = -1
        if self.success(state_update):
            reward = 0
            done = True
        elif self.bumped2wall(state, action):
            reward = -2
        elif self.inPuddle(state_update):
            reward = np.min([-400 * self.dist2edge(state_update),-1])
        
        return state_update, reward, done


    def bumped2wall(self, state, action):
        inc_state = state + np.multiply(self.meshsize , self.action_list[np.argmax(action)] )
        bumped = False
        if (np.min(inc_state) < 0 or np.max(inc_state) > 1) :
            bumped = True
        return bumped

    def success(self,state):
        reach2goal = False
        if (state == self.goal).all(): 
            reach2goal = True
        return reach2goal

        

<<<<<<< HEAD:archive-code/puddleworld.py
p = PuddleWorld()
=======
# p = PuddleWorld()
>>>>>>> 6bc271447fe7a11d457d422a8629e4802fc0c18a:puddleworld.py

############################ END CLASS puddleworld ############################
