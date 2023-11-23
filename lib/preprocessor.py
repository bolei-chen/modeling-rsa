import numpy as np
import pandas as pd 
import torch 
import math 
 
  
columns = ["x","y","u_mean","v_mean","w_mean","p_mean","dissipation_mean","vorticity_mean", "uu","vv","ww","uv","uw","vw","pp"]
i2c = dict(zip(range(0, 15), columns)) 
c2i = dict(zip(columns, range(0, 15))) 
 
class Preprocessor:
 
    def __init__(self, corpus):
        self.corpus = corpus 
        self.y_coords = [sample[1] for sample in corpus if sample[0] == 0] 
        self.x_coords = [sample[0] for sample in corpus if sample[1] == 0] 
        self.y_val2index = dict(zip(self.y_coords, range(0, len(self.y_coords)))) 
        self.x_val2index = dict(zip(self.x_coords, range(0, len(self.x_coords)))) 
         
        p_means = np.reshape(np.array(self.corpus[:, c2i['p_mean']]), (len(self.y_coords), len(self.x_coords))) 
        u_means = np.reshape(np.array(self.corpus[:, c2i['u_mean']]), (len(self.y_coords), len(self.x_coords))) 
        v_means = np.reshape(np.array(self.corpus[:, c2i['v_mean']]), (len(self.y_coords), len(self.x_coords))) 
        gamma1s = np.reshape(np.array(self.corpus[:, c2i['u_mean']] / np.sqrt(self.corpus[:, c2i['u_mean']] ** 2 + self.corpus[:, c2i['v_mean']] ** 2)), (len(self.y_coords), len(self.x_coords))) 
        gamma2s = np.reshape(np.array(self.corpus[:, c2i['v_mean']] / np.sqrt(self.corpus[:, c2i['v_mean']] ** 2 + self.corpus[:, c2i['v_mean']] ** 2)), (len(self.y_coords), len(self.x_coords))) 
        u_mean_squares = np.reshape(np.array([u_mean ** 2 for u_mean in np.array(self.corpus[:, c2i['u_mean']])]), (len(self.y_coords), len(self.x_coords)))
        v_mean_squares = np.reshape(np.array([v_mean ** 2 for v_mean in np.array(self.corpus[:, c2i['v_mean']])]), (len(self.y_coords), len(self.x_coords)))
         
        self.dp_dxs, self.dp_dys = np.gradient(p_means, self.y_coords, self.x_coords) 
        self.du_dxs, self.du_dys = np.gradient(u_means, self.y_coords, self.x_coords) 
        self.dv_dxs, self.dv_dys = np.gradient(v_means, self.y_coords, self.x_coords) 
        self.dus_dxs, self.dus_dys = np.gradient(u_mean_squares, self.y_coords, self.x_coords) 
        self.dvs_dxs, self.dvs_dys = np.gradient(v_mean_squares, self.y_coords, self.x_coords) 
        self.dg1_dxs, self.dg1_dys = np.gradient(gamma1s, self.y_coords, self.x_coords) 
        self.dg2_dxs, self.dg2_dys = np.gradient(gamma2s, self.y_coords, self.x_coords) 
             
    ''' 
    inut:
        t: a 2 by 2 tensor 
    outut:
        the matrix norm of t 
    ''' 
    def _matrix_norm(self, t):
        s = 0 
        for i in range(0, 2):
            for j in range(0, 2):
                s += t[i][j] ** 2 
        return math.sqrt(s) 


    ''' 
    inut: 
        i: the row number selected from the dataframe 
    ouut:
        a tensor exressing delta v 
    ''' 
    def _corpus2deltav(self, i):
        s = self.corpus[i]
        y_i = self.y_val2index[float(s[c2i['y']])] 
        x_i = self.x_val2index[float(s[c2i['x']])] 
        t = torch.DoubleTensor(
            [
                [self.du_dxs[y_i][x_i], self.du_dys[y_i][x_i]], 
                [self.dv_dxs[y_i][x_i], self.dv_dys[y_i][x_i]]
            ]
        ) 
        return t 

    ''' 
    inut:
        i: the row number selected from the dataframe 
    outut:
        comuted q_1 value 
    ''' 
    def q1(self, i):
        L = self._corpus2deltav(i) 
        E = 0.5 * (L - np.transpose(L)) 
        W = 0.5 * (L + np.transpose(L)) 
        q_1_hat = 0.5 * (self._matrix_norm(E) ** 2 - self._matrix_norm(W) ** 2) 
        q_1_star = self._matrix_norm(W) ** 2
        if float(np.abs(q_1_hat) + np.abs(q_1_star)) == 0:
            return 0
        q_1 = q_1_hat / float(np.abs(q_1_hat) + np.abs(q_1_star))
        return float(q_1)

    ''' 
    inut:
        i: the row number selected from the dataframe 
    outut:
        comuted q_4 value 
    ''' 
    def q4(self, i): 
        s = self.corpus[i]
        x_i = self.x_val2index[float(s[c2i['x']])] 
        y_i = self.y_val2index[float(s[c2i['y']])] 
        U = [s[c2i['u_mean']], s[c2i['v_mean']]] 
        partial_dp = [self.dp_dxs[y_i][x_i], self.dp_dys[y_i][x_i]] 
         
        q_4_hat = sum([U[i] * partial_dp[i] for i in range(0, 2)]) 
         
        q_4_star = np.sqrt(sum([partial_dp[j] ** 2 * U[i] ** 2 
                                for i in range(0, 2) 
                                for j in range(0, 2)]))

        if float(np.abs(q_4_hat) + np.abs(q_4_star)) == 0:
            return 0
        q_4 = q_4_hat / float(np.abs(q_4_hat) + np.abs(q_4_star))
        return float(q_4)
         
    ''' 
    inut:
        i: the row number selected from the dataframe 
    outut:
        comuted q_6 value 
    ''' 
    def q6(self, i):
        s = self.corpus[i]
        x_i = self.x_val2index[s[c2i['x']]] 
        y_i = self.y_val2index[s[c2i['y']]] 
        partial_dp = [self.dp_dxs[y_i][x_i], self.dp_dys[y_i][x_i]] 

        q_6_hat = np.sqrt(sum([dp ** 2 for dp in partial_dp])) 
        rho = 1
        dU_square_dxs = [self.dus_dxs[y_i][x_i], self.dvs_dys[y_i][x_i]] 
        q_6_star = 0.5 * rho * sum(dU_square_dxs)
         
        if float(np.abs(q_6_hat) + np.abs(q_6_star)) == 0:
            return 0
        q_6 = q_6_hat / float(np.abs(q_6_hat) + np.abs(q_6_star))
        return float(q_6)
         
    '''
    inut:
        i: the row number selected from the dataframe 
    outut:
        comuted q_7 value 
    ''' 
    def q7(self, i):
        s = self.corpus[i] 
        x_i = self.x_val2index[s[c2i['x']]] 
        y_i = self.y_val2index[s[c2i['y']]] 
        U = [s[c2i['u_mean']], s[c2i['v_mean']]] 
        dU_dxs = [[self.du_dxs[y_i][x_i], self.du_dys[y_i][x_i]],
                 [self.dv_dxs[y_i][x_i], self.dv_dys[y_i][x_i]]] 

        q_7_hat = np.abs(sum([U[i] * U[j] * dU_dxs[i][j] 
                        for i in range(0, 2) 
                        for j in range(0, 2)]))
        q_7_star = np.sqrt(sum([U[l] ** 2 * U[i] * dU_dxs[i][j] * U[k] * dU_dxs[k][j]
                        for i in range(0, 2)
                        for j in range(0, 2)
                        for k in range(0, 2)
                        for l in range(0, 2)]))
         
        if float(np.abs(q_7_hat) + np.abs(q_7_star)) == 0:
            return 0
        q_7 = q_7_hat / float(np.abs(q_7_hat) + np.abs(q_7_star))
        return float(q_7)
             
             
    def q10(self, i):
        s = self.corpus[i]
        x_i = self.x_val2index[s[c2i['x']]] 
        y_i = self.y_val2index[s[c2i['y']]] 
        U = [s[c2i['u_mean']], s[c2i['v_mean']]] 
        dg_dxs = [[self.dg1_dys[y_i][x_i], self.dg1_dys[y_i][x_i]],
                [self.dg2_dxs[y_i][x_i], self.dg2_dys[y_i][x_i]]]
        U_mod = math.sqrt(sum([mean ** 2 for mean in U])) 
        if U_mod == 0:
            return 0 
        q_10_hat = np.abs(sum([U[j] * dg_dxs[i][j] 
                    for i in range(0, 2)
                    for j in range(0, 2)])) / U_mod
        q_10_star = 1 
         
        if float(np.abs(q_10_hat) + np.abs(q_10_star)) == 0:
            return 0
        q_10 = q_10_hat / float(np.abs(q_10_hat) + np.abs(q_10_star))
        return float(q_10)
         
    def vorticity(self, i):
        s = self.corpus[i]
        x_i = self.x_val2index[float(s[c2i['x']])] 
        y_i = self.y_val2index[float(s[c2i['y']])] 
        return self.dv_dxs[y_i][x_i] - self.du_dys[y_i][x_i] 

    def liutex(self, i):
        omega = self.vorticity(i)
        delta_v = self._corpus2deltav(i) 
        omega_s = 1 if omega > 0 else -1
        evalues, _ = np.linalg.eig(delta_v)
        if not all(np.iscomplex(evalues)): 
            return 0 
        lambda_cr, lambda_ci = evalues[0].real, evalues[0].imag
        return omega_s * (np.abs(omega) - np.sqrt(omega ** 2 - 4 * lambda_ci ** 2)) 

    def sheer(self, i):
        return self.vorticity(i) - self.liutex(i) 
         
             
         