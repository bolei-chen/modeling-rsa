import numpy as np
from matplotlib import pyplot as plt 
from utils import * 
 
def plot_barcentric_old(corpus, x, marker='s', c='pink', edgecolors='green', s=5, d=3):
    corners = [[1, 0], [0, 0], [0.5, np.sqrt(3) / 2]] 
    
    selected_is = select(corpus, 'x', x) 
    cs = [to_cs(to_lambdas(to_rsa(to_rst(corpus, i)))) for i in range(0, len(corpus)) if i in selected_is]
     
    triangle = plt.Polygon(corners, fill=False, color='green')
    plt.gca().add_patch(triangle) 
     
    coords = [to_barcoords(c) for c in cs] 
    x, y = zip(*coords)
    plt.scatter(x, y, marker=marker, c=c, edgecolors=edgecolors, s=s) 
    plt.title("barycentric triangle map") 
    plt.show() 
    

def evaluate(corpus, x, cs_pred, ms=['s', 's'], cs=['pink', 'blue'], edgecolors=["green", "red"], s=5):
    corners = np.array([[1, 0], [0, 0], [0.5, np.sqrt(3) / 2]])
    triangle = plt.Polygon(corners, fill=False, color='green')
    plt.gca().add_patch(triangle) 
     
    selected_is = select(corpus, 'x', x) 
    cs_real = [to_cs(to_lambdas(to_rsa(to_rst(corpus, i)))) for i in range(0, len(corpus)) if i in selected_is]
         
    coords_real = np.array([to_barcoords(c_real) for c_real in cs_real]) 
    xs_real, ys_real = zip(*coords_real) 
     
    coords_pred = np.array([to_barcoords(c_pred) for c_pred in cs_pred])
    xs_pred, ys_pred = zip(*coords_pred) 

    plt.scatter(xs_real, ys_real, marker='s', c='pink', edgecolors='green', s=5)
    plt.scatter(xs_pred, ys_pred, marker='s', c='blue', edgecolors='red', s=5)
    plt.title('barycentric evaludation map') 
    plt.show() 

         