import numpy as np
from numpy import pi
import math
from scipy.spatial.distance import cdist

def closest_point(pt, others):
    distances = cdist(pt, others)
    return others[distances.argmin()]
        
def shortest_distance(pt, others):
    distances = cdist(pt, others)
    return np.min(distances)

class puddleworld:    
    def __init__(self):
        puddle = self.CreatePuddleOutsidePoints()
        self.puddle = puddle
    
    def CreatePuddleOutsidePoints(self):
        puddle = []
        # to find an accurate distance to edge mess is finer
        ngrid = [40, 40]
        x_vec = np.linspace(0,1,ngrid[0])
        y_vec = np.linspace(0,1,ngrid[1])
        for x in x_vec:
            for y in y_vec:
                if ~self.inpuddle([x,y]):
                    puddle.append([x,y])
        # puddle is a closed loop 
        outpuddlepts = np.asarray(puddle)
        return outpuddlepts


    def inpuddle(self, state):
        agentinPuddle = False
        # Horizontal wing of puddle consists of 
        # 1) rectangle area xch1<= x <=xc2 && ych1-radius <= y <=ych2+radius
        # (xchi,ychi) is the center points (h ==> horizantal)
        # x, y = state[0], state[1]

        x, y = state[0], state[1]
        xch1, ych1 = 0.3, 0.75
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


