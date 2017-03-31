# An object to apply dependent GP prediciton given vector input. The algorithm
# follows the examples in **Multiple Output Gaussian Process Regression** by
# Phillip Boyle and Marcus Frean.
#
# See gen_dgp_figures.py for usage.
#
# Author:   Shih-Yuan Liu <shihyuan.liu@gmail.com>

import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import copy

class dgp(object):
    @staticmethod
    def DgpParam():
        return namedtuple('DgpParam',['A_1','A_2','v_1','v_2','B_1','B_2','w_1','w_2','sig_sq_1','sig_sq_2','mu','p'])

    def __init__(self,dgp_param):
        self.set_param(dgp_param)

    def set_param(self,dgp_param):
        self.param = copy.deepcopy(dgp_param)
        self._gen_C_Y()

    def _Sigma(self,A):
        A_sum_inv = np.linalg.inv(A[0]+A[1])
        return A[0].dot(A_sum_inv).dot(A[1])

    def _C_ij(self,d,A,v,mu,p):
        A_i, A_j = A[0], A[1]
        v_i, v_j = v[0], v[1]
        scale = (((2*np.pi)**(p/2.))*v_i*v_j)/np.sqrt(np.linalg.det(A_i+A_j))
        exp_order = -0.5*(np.atleast_1d(d-mu).dot(self._Sigma(A))).dot(np.atleast_1d(d-mu))
        return scale*np.exp(exp_order)

    def _gen_C_Y(self):
        C_U_11 = lambda d: self._C_ij(d,(self.param.A_1,self.param.A_1), (self.param.v_1,self.param.v_1),0,self.param.p)
        C_U_12 = lambda d: self._C_ij(d,(self.param.A_1,self.param.A_2), (self.param.v_1,self.param.v_2),self.param.mu,self.param.p)
        C_U_21 = lambda d: self._C_ij(d,(self.param.A_1,self.param.A_2), (self.param.v_1,self.param.v_2),-self.param.mu,self.param.p)
        C_U_22 = lambda d: self._C_ij(d,(self.param.A_2,self.param.A_2), (self.param.v_2,self.param.v_2),0,self.param.p)
        C_V_11 = lambda d: self._C_ij(d,(self.param.B_1,self.param.B_1), (self.param.w_1,self.param.w_1),0,self.param.p)
        C_V_22 = lambda d: self._C_ij(d,(self.param.B_2,self.param.B_2), (self.param.w_2,self.param.w_2),0,self.param.p)

        self.C_Y_11 = lambda d: C_U_11(d) + C_V_11(d) + float((d==0)*self.param.sig_sq_1)
        self.C_Y_22 = lambda d: C_U_22(d) + C_V_22(d) + float((d==0)*self.param.sig_sq_2)
        
        self.C_Y_12 = lambda d: C_U_12(d)
        self.C_Y_21 = lambda d: C_U_21(d)

    def get_covar_matrix(self,X_1,X_2):
        C_11 = np.atleast_2d([[ self.C_Y_11(np.atleast_1d(x_j - x_i)) for x_i in X_1]for x_j in X_1])
        C_22 = np.atleast_2d([[ self.C_Y_22(np.atleast_1d(x_j - x_i)) for x_i in X_2]for x_j in X_2])
        C_12 = np.atleast_2d([[ self.C_Y_12(np.atleast_1d(x_j - x_i)) for x_i in X_2]for x_j in X_1])
        C_21 = np.atleast_2d([[ self.C_Y_21(np.atleast_1d(x_j - x_i)) for x_i in X_1]for x_j in X_2])

        if C_22.size == 0:
            return C_11
        else:
            return np.vstack([np.hstack([C_11,C_12]),np.hstack([C_21,C_22])])

    def _predict(self,x_query, X_1, X_2, Y_1, Y_2,C_inv):
        # C = self.get_covar_matrix(X_1,X_2)
        # C_inv = np.linalg.inv(C)
        kappa_1, kappa_2 = self.C_Y_11(np.atleast_1d(0)), self.C_Y_22(np.atleast_1d(0))

        k_11 = np.array([ self.C_Y_11(x_query - x_1) for x_1 in X_1])
        k_12 = np.array([ self.C_Y_12(x_query - x_2) for x_2 in X_2])
        k_1 = np.hstack([k_11,k_12])

        k_21 = np.array([ self.C_Y_21(x_query - x_1) for x_1 in X_1])
        k_22 = np.array([ self.C_Y_22(x_query - x_2) for x_2 in X_2])
        k_2 = np.hstack([k_21,k_22])

        Y = np.hstack([np.array(Y_1),np.array(Y_2)])
        y_pred_1 = k_1.dot(C_inv).dot(Y)
        y_pred_2 = k_2.dot(C_inv).dot(Y)

        sigma_sq_1 = kappa_1 - k_1.dot(C_inv).dot(k_1)
        sigma_sq_2 = kappa_2 - k_2.dot(C_inv).dot(k_2)
        return y_pred_1, y_pred_2, sigma_sq_1, sigma_sq_2

    def predict(self,x_list,X_1,X_2,Y_1,Y_2):
        C = self.get_covar_matrix(X_1,X_2)
        C_inv = np.linalg.inv(C)
        pred_data = [self._predict(x, X_1,X_2,Y_1,Y_2, C_inv) for x in x_list]
        y_1,y_2,sig_1,sig_2 = np.transpose(pred_data)
        return y_1, y_2, sig_1, sig_2

    def get_likelihood(self,X_1,X_2,Y_1,Y_2):
        C = self.get_covar_matrix(X_1,X_2)
        C_inv = np.linalg.inv(C)
        Y = np.hstack([Y_1,Y_2])
        return -0.5*np.log(np.linalg.det(C)) -0.5*Y.dot(C_inv).dot(Y) - ((len(X_1)+len(X_2))/2.0)*np.log(2*np.pi)