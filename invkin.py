import math
import numpy as np

def div(a,b):
        return a/b if b is not 0 else math.inf if a>0 else -math.inf

class Arm :
    def __init__(self,arm_lens):
        self.arm_lens = arm_lens
        
    def get_position(self,angles):
        x = self.arm_lens[0]*np.cos(angles[0]) + self.arm_lens[1]* np.cos(angles[0]+angles[1])
        y = self.arm_lens[0]*np.sin(angles[0]) + self.arm_lens[1]* np.sin(angles[0]+angles[1])
        return x,y

    def inv_kin(self,final_coords):
        D = div( (final_coords[0]**2 + final_coords[1]**2 - self.arm_lens[0]**2 - self.arm_lens[1]**2), 2*self.arm_lens[0]*self.arm_lens[1] )
        angles = [0,0]
        temp = math.atan2( (1-(D**2))**(1/2),D  )
        angles[1] = temp if temp>=0 else -temp
        angles[0] = math.atan2(final_coords[1],final_coords[0]) - math.atan2( self.arm_lens[1]*np.sin(angles[1]), self.arm_lens[0]+ self.arm_lens[1]*np.cos(angles[1]) )
        return np.array(angles)

    def time_series(self,coordinate_series):
        angle_series=[]
        for i in range(len(coordinate_series)):
            angle_series.append(self.inv_kin(coordinate_series[i]))
        return angle_series
def test():

    initialangles = [.349 , .349]
    lengths = [4,4]
    initialPos = [0,0]
    Arm1 = Arm(lengths)

    series = [[2,2],[3,3],[4,4]]
    print(Arm1.time_series(series))


