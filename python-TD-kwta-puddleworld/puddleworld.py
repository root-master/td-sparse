###### CLASS puddleworld ######################################
import math
from scipy.spatial.distance import cdist
import numpy as np
from numpy import pi
from copy import copy
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
                ngrid = [10.0,10.0],
                maxStep = 40):
        """
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
        ngridx = 10
        ngridy = 10
        ngrid = [ngridx,ngridy]
        """
        meshsize = [1/ngrid[0] , 1/ngrid[1]] # uniform meshing here
        puddle = self.createPointsOutsidePuddle()
        discrete_state_vec = discretizatize_state_vector(state_bound, ngrid)
        states_list =  build_states_list(discrete_state_vec)
        
        self.EnvironmentName = 'Puddle World' 
        self.nd = nd
        self.nA = nA
        self.goal = goal
        self.puddle = puddle
        self.action_list = action_list
        self.state_bound = state_bound
        self.discrete_state_vec = discrete_state_vec
        self.states_list = states_list
        self.ngrid = ngrid
        self.meshsize = meshsize
        self.maxStep = maxStep
        self.input_size = len(discrete_state_vec[0]) + len(discrete_state_vec[1])
        
        puddle_boundary_points = self.create_puddle_boundary()
        self.puddle_boundary_points = puddle_boundary_points
        print('Environment: ' + self.EnvironmentName)
        
    
    # I don't know how to extract the coordinate of the boundary of puddle
    # if you have time create a function:
    # def boundary(list_of_points):
    #    ...
    # return boundary_points

    # return array of points outside of puddle -- 
    # these are not necessarily state actually finer grid used to find accuare
    # distance of agent to edge of puddle
    def createPointsOutsidePuddle(self):
        puddle = []
        # to find an accurate distance to edge mess is finer
        ngrid = [20, 20]
        x_vec = np.linspace(0,1,ngrid[0])
        y_vec = np.linspace(0,1,ngrid[1])
        for x in x_vec:
            for y in y_vec:
                if ~self.inPuddle([x,y]):
                    puddle.append([x,y])
        # puddle is a closed loop 
        outpuddlepts = np.asarray(puddle)
        return outpuddlepts

    def create_puddle_boundary(self): 
        xch1, ych1 = 0.3, 0.7
        xch2, ych2 = 0.65, ych1
 
        xcv1, ycv1 = 0.45, 0.4
        xcv2, ycv2 = xcv1, 0.8

        num_grids = 20
        radius = 0.1
        theta = np.linspace(0.0,pi,num=num_grids).reshape(1,-1).T  
        
        # bottom horizantal line 
        num_grids_l1 = round( (xch2-xch1) / self.meshsize[0] ) + 1
        l1_x = np.linspace(xch1,xch2,num=num_grids_l1).reshape(1,-1).T
        l1_y = (ych1-radius) * np.ones_like(l1_x)
        l1 = np.concatenate((l1_x,l1_y),axis=1)
     
        # top horizantal line 
        num_grids_l2 = round( (xch2-xch1) / self.meshsize[0] ) + 1
        l2_x = np.linspace(xch1,xch2,num=num_grids_l2).reshape(1,-1).T
        l2_y = (ych1+radius) * np.ones_like(l2_x)
        l2 = np.concatenate((l2_x,l2_y),axis=1)

        # left vertical line 
        num_grids_l3 = round( (ycv2-ycv1) / self.meshsize[1] ) + 1
        l3_y = np.linspace(ycv1,ycv2,num=num_grids_l3).reshape(1,-1).T
        l3_x = (xcv1-radius) * np.ones_like(l3_y)
        l3 = np.concatenate((l3_x,l3_y),axis=1)

        # right vertical line 
        num_grids_l4 = round( (ycv2-ycv1) / self.meshsize[1] ) + 1
        l4_y = np.linspace(ycv1,ycv2,num=num_grids_l4).reshape(1,-1).T
        l4_x = (xcv1+radius) * np.ones_like(l4_y)
        l4 = np.concatenate((l4_x,l4_y),axis=1)

        # bottom semi-circle
        cv1_x = xcv1 + radius * np.cos(theta+pi)
        cv1_y = ycv1 + radius * np.sin(theta+pi)
        cv1 = np.concatenate((cv1_x,cv1_y),axis=1)        

        # top semi-circle
        cv2_x = xcv2 + radius * np.cos(theta)
        cv2_y = ycv2 + radius * np.sin(theta)
        cv2 = np.concatenate((cv2_x,cv2_y),axis=1)        

        # left semi-circle
        ch1_x = xch1 + radius * np.cos(theta+pi/2)
        ch1_y = ych1 + radius * np.sin(theta+pi/2)
        ch1 = np.concatenate((ch1_x,ch1_y),axis=1)        

        # left semi-circle
        ch2_x = xch2 + radius * np.cos(theta-pi/2)
        ch2_y = ych2 + radius * np.sin(theta-pi/2)
        ch2 = np.concatenate((ch2_x,ch2_y),axis=1)

        puddle_boundary = np.concatenate((l1,l2,l3,l4,ch1,ch2,cv1,cv2),axis=0)        
        return puddle_boundary


    def set_goal(self, goal):
        self.goal = goal

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
        inHorRec = (x>=xch1) and (y>= ych1-radius) and (x<=xch2)  and \
                                                            (y<=ych2+radius)   
        # 2) two half-circle at end edges of rectangle
        inHorCir1 = ( ( (x-xch1)**2 + (y-ych1)**2 <= radius**2 ) and x<xch1 )

        inHorCir2 = ( ((x-xch2)**2 + (y-ych2)**2) <= radius**2 and x>xch2 )
        inHor = inHorRec or inHorCir1 or inHorCir2

        #Vertical wing of puddle consists of 
        # 1) rectangle area xcv1-radius<= x <=xcv2+radius && ycv1 <= y <= ycv2
        # where (xcvi,ycvi) is the center points (v ==> vertical)
        xcv1, ycv1 = 0.45, 0.4
        xcv2, ycv2 = xcv1, 0.8

        inVerRec = (x >= xcv1-radius) and (y >= ycv1) and (x <= xcv2+radius) \
                                                                and (y <= ycv2)    
        # % 2) two half-circle at end edges of rectangle
        inVerCir1 = ( ( (x-xcv1)**2 + (y-ycv1)**2 <= radius**2 ) and y<ycv1 )
        inVerCir2 = ( ( (x-xcv2)**2 + (y-ycv2)**2 <= radius**2 ) and y>ycv2 )
        inVer = inVerRec or inVerCir1 or inVerCir2

        agentinPuddle = inHor or inVer

        return agentinPuddle

    def dist2edge(self, state):
        agentinPuddle = False
        closestDist = 0.0
        # Horizontal wing of puddle consists of 
        # 1) rectangle area xch1<= x <=xc2 && ych1-radius <= y <=ych2+radius
        # (xchi,ychi) is the center points (h ==> horizantal)
        # x, y = state[0], state[1]
        x, y = state[0], state[1]
        xch1, ych1 = 0.3, 0.7
        xch2, ych2 = 0.65, ych1
        radius = 0.1
        inHorRec = (x>=xch1) and (y>= ych1-radius) and (x<=xch2) \
                                                        and (y<=ych2+radius)   
        # 2) two half-circle at end edges of rectangle
        inHorCir1 = ( ( (x-xch1)**2 + (y-ych1)**2 <= radius**2 ) and x<xch1 )

        inHorCir2 = ( ((x-xch2)**2 + (y-ych2)**2) <= radius**2 and x>xch2 )
        inHor = inHorRec or inHorCir1 or inHorCir2

        #Vertical wing of puddle consists of 
        # 1) rectangle area xcv1-radius<= x <=xcv2+radius && ycv1 <= y <= ycv2
        # where (xcvi,ycvi) is the center points (v ==> vertical)
        xcv1 = 0.45; ycv1=0.4;
        xcv2 = xcv1; ycv2 = 0.8;

        inVerRec = (x >= xcv1-radius) and (y >= ycv1) and \
                                    (x <= xcv2+radius) and (y <= ycv2)    
        # % 2) two half-circle at end edges of rectangle
        inVerCir1 = ( ( (x-xcv1)**2 + (y-ycv1)**2 <= radius**2 ) and y<ycv1 )
        inVerCir2 = ( ( (x-xcv2)**2 + (y-ycv2)**2 <= radius**2 ) and y>ycv2 )
        inVer = inVerRec or inVerCir1 or inVerCir2

        agentinPuddle = inHor or inVer
        closestDist = 0
        num_grids = 20
        theta = np.linspace(0.0,pi,num=num_grids).reshape(1,-1).T
        if inHorCir1:
            xp = xch1 + radius * np.cos(theta+pi/2)
            yp = ych1 + radius * np.sin(theta+pi/2)
            points = np.concatenate((xp,yp),axis=1)
            closestDistHorCircle = shortest_distance(np.asarray([state]),points)

        if inHorCir2:
            xp = xch2 + radius * np.cos(theta-pi/2)
            yp = ych2 + radius * np.sin(theta-pi/2)
            points = np.concatenate((xp,yp),axis=1)
            closestDistHorCircle = shortest_distance(np.asarray([state]),points)

        if inVerCir1:
            xp = xcv1 + radius * np.cos(theta+pi)
            yp = ycv1 + radius * np.sin(theta+pi)
            points = np.concatenate((xp,yp),axis=1)
            closestDistVerCircle = shortest_distance(np.asarray([state]),points)

        if inVerCir2:
            xp = xcv2 + radius * np.cos(theta)
            yp = ycv2 + radius * np.sin(theta)
            points = np.concatenate((xp,yp),axis=1)
            closestDistVerCircle = shortest_distance(np.asarray([state]),points)

        if inHor and not inVer:
            if inHorRec:
                points = np.array([[x, ych1+radius], [x, ych1-radius]])
                closestDist = shortest_distance(np.asarray([state]), points)
            else:
                closestDist = closestDistHorCircle
        elif inHorRec and inVerRec:
            points = np.array(  [[xcv1-radius, ych1-radius], 
                                [xcv1-radius,ych1+radius],
                                [xcv1+radius,ych1-radius],
                                [xcv1+radius,ych2+radius]])
            closestDist = shortest_distance(np.asarray([state]), points)
        elif inHorRec and (inVerCir1 or inVerCir2):
            points = np.array(  [[x, ych1+radius], 
                                [x, ych1-radius]] )
            closestDist = shortest_distance(np.asarray([state]), points)
            closestDist = np.max(closestDist,closestDistVerCircle)

        if inVer and not inHor:
            if inVerRec:
                points = np.array(  [[xcv1-radius,y], 
                                    [xcv1+radius,y]] ) 
                closestDist = shortest_distance(np.asarray([state]), points)
            else:
                closestDist = closestDistVerCircle
        elif inVerRec and (inHorCir1 or inHorCir2):
            points = np.array(  [[xcv1-radius,y], 
                                [xcv1+radius,y]] )
            closestDist = shortest_distance(np.asarray([state]), points)
            closestDist = np.max(closestDist,closestDistHorCircle)
        elif (inVerCir1 or inVerCir2) and (inHorCir1 or inHorCir2):
            closestDist = np.max(closestDistHorCircle,closestDistVerCircle)

        return closestDist


    # def dist2edge(self, state):
    #     state = np.asarray([state])
    #     closestDist = shortest_distance(state , self.puddle)
    #     return closestDist

    # def dist2edge(self, state):
    #     state = np.asarray([state])
    #     closestDist = shortest_distance(state , self.puddle_boundary_points)
    #     return closestDist

    def closest_puddle_point_to_state(self,state):
        state = np.asarray([state])
        return closest_point(state, self.puddle_boundary_points)


    def update_state_env_reward(self, state, action):
        inc_state = copy(state)
        inc_state = state + np.multiply(self.meshsize , 
                            self.action_list[np.argmax(action)] )
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
            reward = -400 * self.dist2edge(state_update)
            reward = min(-1, reward)
        
        return state_update, reward, done


    def bumped2wall(self, state, action):
        inc_state = state + np.multiply(self.meshsize , 
                            self.action_list[np.argmax(action)] )
        bumped = False
        if (np.min(inc_state) < 0 or np.max(inc_state) > 1) :
            bumped = True
        return bumped

    def success(self,state):
        reach2goal = False
        if (state == self.goal).all(): 
            reach2goal = True
        return reach2goal

############################ END CLASS puddleworld ############################
