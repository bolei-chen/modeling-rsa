import numpy as np
from matplotlib import pyplot as plt 
import torch
import math 

class Plots:
     
    def __init__(self):
        pass 
     
    # input: a pandas dataframe and the row number of the dataframe 
    # ouput: a torch tensor generated from the row data 
    def __construct_tensor(self, df, index):
        delta = torch.DoubleTensor([[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]]) 
        s = df.iloc[[index]] 
        t = torch.DoubleTensor([list(s.uu) + list(s.uv) + list(s.uw),
                                list(s.uv) + list(s.vv) + list(s.vw), 
                                list(s.uw) + list(s.vw) + list(s.ww)])
         
        k = sum(map(lambda i : t[i][i], range(0, 3))) / 2
         
        normalized_t = torch.zeros((3, 3))
         
        for i in range(0, 3):
            for j in range(0, 3):
                normalized_t[i][j] = t[i][j] / (2 * k) - delta[i][j] / 3
                 
        return normalized_t 

    # input: the lambda values of a tensor
    # ouput: the c values needed for the barycentric plot 
    def __get_cs(self, ls):
        c1 = ls[0] - ls[1]
        c2 = 2 * (ls[1] - ls[2]) 
        c3 = 3 * ls[2] + 1 
        return [c1, c2, c3] 
        
    # input: a torch tensor
    # ouput: a torch tensor which is the anisotropy of the input 
    def __get_anisotropy(self, t):
        k = 0
        for i in range(0, 3):
            k += t[i][i]
        k /= 2 
        for i in range(0, 3):
            t[i][i] -= 2 * k / 3 
        return t 

    # input: the lambda values of a tensor
    # ouput: the princial values needed for the lumley plot 
    def __get_principals(self, ls):
        p2 = ls[0] ** 2 + ls[0] * ls[1] + ls[1] ** 2
        p3 = -ls[0] * ls[1] * (ls[0] + ls[1]) 
        return [p2, p3] 
     
    # input: a torch tensor 
    # output: the xi and eta values 
    def __get_xi_and_eta(self, t):
        d_cube_sum = sum(list(map(lambda i : t[i][i] ** 3, range(0, 3))))
        d_square_sum = sum(list(map(lambda i : t[i][i] ** 2, range(0, 3))))  
        xi = (d_cube_sum / 6) ** (1 / 3) 
        eta = math.sqrt(d_square_sum / 6) 
        return [xi, eta] 
     
    # input: a torch tensor
    # output: a sorted list of lambda values 
    def __get_ls(self, t):
        ls, _ = torch.linalg.eig(torch.nan_to_num(t))
        ls = np.array(ls) 
        return np.sort(ls)[::-1] 

    # input: a list of c values 
    # output: a barycentric coordinate
    def __get_barycentric_coords(self, cs):
        coords = np.array([[1, 0], [0, 0], [0.5, math.sqrt(3) / 2]])
        coords = list(map(lambda i : coords[i] * cs[i], range(0, len(coords)))) 
        return coords[0] + coords[1] + coords[2] 
         
    # input: a list of principal values 
    # output: a lumley coordinate
    def __get_lumley_coords(self, ps):
        return ps[::-1] 

    # input: a pair of xi and eta values 
    # output: a turbulence coordinate
    def __get_turbulence_coords(self, ps):
        return ps[::-1]

    # input: a list of lambda values
    # output: a eigen coordinate 
    def __get_eigen_coords(self, ls):
        return ls[:2] 
     
    ''' 
    input:
        df: a pandas dataframe
        marker: style of a sample point, default is set to s
        c: color of sample point, default is set to pink
        edgecolors: edge color of a sample point, default is set to green
        s: size of a sample point, default is set to 5 
        d: density of sample points in the plot, the higher the value, the higher the number of samples ploted, default is set to 3
    output:
        a barycentric plot of the dataframe 
    ''' 
    def plot_barcentric_map(self, df, marker='s', c='pink', edgecolors='green', s=5, d=3):
        corners = [[1, 0], [0, 0], [0.5, math.sqrt(3) / 2]] 
         
        # draw the triangle 
        triangle = plt.Polygon(corners, fill=False, color='green')
        plt.gca().add_patch(triangle) 
         
        # get the coordinates and plot the dots 
        coords = list(map(lambda i : 
        list(
            self.__get_barycentric_coords(
                self.__get_cs(
                    self.__get_ls(
                        self.__get_anisotropy(
                            self.__construct_tensor(df, i)
                        )
                    )
                )
            )
        ), range(0, len(df), 100 // d))) 
        x, y = zip(*coords)
        plt.scatter(x, y, marker=marker, c=c, edgecolors=edgecolors, s=s) 
        plt.title("barycentric triangle map") 
        plt.show() 
         
    ''' 
    input:
        df: a pandas dataframe
        marker: style of a sample point, default is set to s
        c: color of sample point, default is set to pink
        edgecolors: edge color of a sample point, default is set to green
        s: size of a sample point, default is set to 5 
        d: density of sample points in the plot, the higher the value, the higher the number of samples ploted, default is set to 3
    output:
        a lumley plot of the dataframe 
    ''' 
    def plot_lumley_triangle_map(self, df, marker='s', c='pink', edgecolors='green', s=5, d=3):
        coords = list(map(lambda i : 
        list(
            self.__get_lumley_coords(
                self.__get_principals(
                    self.__get_ls(
                        self.__get_anisotropy(
                            self.__construct_tensor(df, i)
                        )
                    )
                )
            )
        ), range(0, len(df), 100 // d))) 
        x, y = zip(*coords) 
        plt.scatter(x, y, marker=marker, c=c, edgecolors=edgecolors, s=s) 
        plt.xlabel("principal 3") 
        plt.ylabel("principal 2") 
        plt.title("lumley triangle map") 
        plt.show() 

 
    ''' 
    input:
        df: a pandas dataframe
        marker: style of a sample point, default is set to s
        c: color of sample point, default is set to pink
        edgecolors: edge color of a sample point, default is set to green
        s: size of a sample point, default is set to 5 
        d: density of sample points in the plot, the higher the value, the higher the number of samples ploted, default is set to 3 
    output:
        a turbulence plot of the dataframe 
    ''' 
    def plot_turbulence_triangle_map(self, df, marker='s', c='pink', edgecolors='green', s=5, d=3):
        coords = list(map(lambda i : 
        list(
            self.__get_turbulence_coords(
                self.__get_xi_and_eta(
                    self.__get_anisotropy(
                        self.__construct_tensor(df, i)
                    )
                )
            )
        ), range(0, len(df), 100 // d))) 
        x, y = zip(*coords) 
        plt.scatter(x, y, marker=marker, c=c, edgecolors=edgecolors, s=s) 
        plt.xlabel("eta") 
        plt.ylabel("xi") 
        plt.title("turbulence triangle map") 
        plt.show() 
     
      
    ''' 
    input:
        df: a pandas dataframe
        marker: style of a sample point, default is set to s
        c: color of sample point, default is set to pink
        edgecolors: edge color of a sample point, default is set to green
        s: size of a sample point, default is set to 5 
        d: density of sample points in the plot, the higher the value, the higher the number of samples ploted, default is set to 3
    output:
        a eigen value plot of the dataframe 
    ''' 
    def plot_eigen_value_map(self, df, marker='s', c='pink', edgecolors='green', s=5, d=3):
        coords = list(map(lambda i : 
        list(
            self.__get_eigen_coords(
                self.__get_ls(
                    self.__get_anisotropy(
                        self.__construct_tensor(df, i)
                    )
                )
            )
        ), range(0, len(df), 100 // d)))
        x, y = zip(*coords) 
        plt.scatter(x, y, marker=marker, c=c, edgecolors=edgecolors, s=s) 
        plt.xlabel("lambda2") 
        plt.ylabel("lambda1") 
        plt.title("eigen value map") 
        plt.show() 