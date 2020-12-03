import numpy as np
import scipy.optimize


class Arm3Link:

    def __init__(self,Len=None):

        self.angles = [.3, .3, 0] 
        self.default = np.array([np.pi/4, np.pi/4, np.pi/4]) 
        self.Len = np.array([1, 1, 1]) if Len is None else Len

    def inv_kin(self, xy):

        def distance_to_default(q, *args):
            weight = [1, 1, 1.3]
            return np.sqrt(np.sum([(qi - q0i)**2 * wi for qi, q0i, wi in zip(q, self.default, weight)]))
            

        def x_constraint(q, xy):
            x = (self.Len[0]*np.cos(q[0]) + self.Len[1]*np.cos(q[0]+q[1]) +self.Len[2]*np.cos(np.sum(q))) - xy[0]
            return x

        def y_constraint(q, xy):
            y = (self.Len[0]*np.sin(q[0]) + self.Len[1]*np.sin(q[0]+q[1]) +self.Len[2]*np.sin(np.sum(q))) - xy[1]
            return y


        return scipy.optimize.fmin_slsqp(func=distance_to_default,x0=self.angles,eqcons=[x_constraint,y_constraint],args=(xy,),iprint=0) 

    def time_series(self,coordinate_series):
        angle_series=[]
        self.angles = self.inv_kin(coordinate_series[0])

        for i in range(len(coordinate_series)):
            angle_series.append(self.inv_kin(coordinate_series[i]))
            self.angles = self.inv_kin(coordinate_series[i])

        return angle_series
