# A script to generate figures for dependent and independent GP given the same
# set of vector data.
# Author:   Shih-Yuan Liu <shihyuan.liu@gmail.com>

import numpy as np
import matplotlib.pyplot as plt
from dgp import dgp

# Generate Observation Data
X_1 = np.atleast_1d([-2.,-1.,0.])
Y_1 = np.atleast_1d([-1.,0.,1.])
X_2 = np.atleast_1d([0.])
Y_2 = np.atleast_1d([0.5])
x_pred_list = np.linspace(-5,5,100)

def gen_figure(title,filename,y_1,y_2,sig_1,sig_2):
    # Set size of the figure
    plt.figure(figsize=(4,3))
    # Plot GP covariance shade
    plt.fill_between(x_pred_list,y_1 - sig_1, y_1 + sig_1, alpha=0.2, color='b')
    plt.fill_between(x_pred_list,y_2 - sig_2, y_2 + sig_2, alpha=0.2, color='r')
    # Plot GP mean prediciton
    plt.plot(x_pred_list,y_1,'b',label="$f(t)$",linewidth=1.5)
    plt.plot(x_pred_list,y_2,'r',label="$g(t)$",linewidth=1.5)
    # Plot data points 
    plt.plot(X_1,Y_1,'bo')
    plt.plot(X_2,Y_2,'ro')
    # Set x axis limits
    plt.xlim((min(x_pred_list),max(x_pred_list)))
    # Set y axis limits
    ylim = [-2.0,2.0]
    plt.ylim(ylim)
    # Plot legend
    plt.legend(loc=4)
    # Set title
    plt.title(title)
    # Save figure as pdf files
    plt.savefig(filename)
    print("Figure saved to %s" %(filename))

# Define default DGP parameters and create dgp obj
param = dgp.DgpParam()
param.A_1 = np.atleast_2d(1.)
param.A_2 = np.atleast_2d(1.) 
param.B_1 = np.atleast_2d(1.)
param.B_2 = np.atleast_2d(1.)
param.sig_sq_1 = 0.0
param.sig_sq_2 = 0.0
param.mu = 0.0
param.p = 1.0

# Generate Dependent GP Figure
# Set parameter for dependent GP
param.v_1 = 0.5
param.v_2 = 0.5
param.w_1 = 0.5
param.w_2 = 0.5
dgp_a = dgp(param)
y_1,y_2,sig_1,sig_2 = dgp_a.predict(x_pred_list,X_1,X_2,Y_1,Y_2)
gen_figure("Dependent GP","fig_dgp.pdf",y_1,y_2,sig_1,sig_2)

# Generate Dependent GP Figure
# Set parameter for independent GP
param.v_1 = 0.
param.v_2 = 0.
param.w_1 = 1.
param.w_2 = 1.
dgp_b = dgp(param)
y_1,y_2,sig_1,sig_2 = dgp_b.predict(x_pred_list,X_1,X_2,Y_1,Y_2)
gen_figure("Independent GP","fig_igp.pdf",y_1,y_2,sig_1,sig_2)