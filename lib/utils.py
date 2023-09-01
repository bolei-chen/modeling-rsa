import torch 
import numpy as np 
import warnings
warnings.simplefilter('ignore') 
  
columns = ["x","y","u_mean","v_mean","w_mean","p_mean","dissipation_mean","vorticity_mean", "uu","vv","ww","uv","uw","vw","pp"]
i2c = dict(zip(range(0, 15), columns)) 
c2i = dict(zip(columns, range(0, 15))) 
 
def load(file_path): 
    data = np.loadtxt(file_path, skiprows=20)
    return data
 
def get_cs(corpus, i):
    delta = torch.DoubleTensor([[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]]) 
    s = corpus[i] 
    meanings = [['uu', 'uv', 'uw'],
                ['uv', 'vv', 'vw'],
                ['uw', 'vw', 'ww']] 
    t = torch.FloatTensor([[s[c2i[meanings[i][j]]] for j in range(0, 3)] for i in range(0, 3)]) 
     
    k = sum([t[i][i] for i in range(0, 3)]) / 2
    if 2 * k == 0:
        t = - delta / 3 
    else:
        t = t / (2 * k) - delta / 3 
         
    for i in range(0, 3):
        t[i][i] -= 2 * k / 3 
         
    [lambda1, lambda2, lambda3] = torch.sort(torch.linalg.eigvals(torch.nan_to_num(t)).real, descending=True)[0]
    c1 = lambda1 - lambda2 
    c2 = 2 * (lambda2 - lambda3) 
    c3 = 3 * lambda3 + 1 
    return np.array([c1, c2, c3])

def cs2coords(cvalues):
    corners = np.array([[1, 0], [0, 0], [0.5, np.sqrt(3) / 2]])
    coords = [(corners[0] * cvalue[0] + corners[1] * cvalue[1] + corners[2] * cvalue[2]).real for cvalue in cvalues] 
    return coords
     
def select(corpus, c, x):
    m = c2i[c] 
    x_nearest = 0 
    diff = np.inf
    for sample in corpus: 
        curr_diff = np.abs(x - sample[m]) 
        if (curr_diff < diff):
            x_nearest = sample[m] 
            diff = curr_diff 
    selected_is = [i for i in range(0, len(corpus)) if corpus[i, m] == x_nearest] 
    return selected_is 