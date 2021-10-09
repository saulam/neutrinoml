import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale, PolynomialFeatures

def plot_event(df, event_n):
    event = df.iloc[event_n]
    pid = event['TruePID']
    momentum = event['TrueMomentum']
    X = event['NodePosX']
    Y = event['NodePosY']
    Z = event['NodePosZ']
    
    # ranges of the detector
    min_x = -985.92
    max_x = 985.92
    min_y = -257.56
    max_y = 317.56
    min_z = -2888.78
    max_z = -999.1
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X, Z, Y)
    ax.set_xlabel('X (mm)', labelpad=8)
    ax.set_xticks([-900, -600, -300, 0, 300, 600, 900])
    ax.set_ylabel('Z (mm)', labelpad=10)
    ax.set_yticks([-2750, -2250, -1750, -1250])
    ax.set_zlabel('Y (mm)', labelpad=5)
    ax.set_zticks([-300, -150, 0, 150, 300])
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_z, max_z)
    ax.set_zlim(min_y, max_y)
    ax.auto_scale_xyz([min_x, max_x], [min_z, max_z], [min_y, max_y])
    #ax.set_box_aspect((abs(max_x-min_x), abs(max_z-min_z), abs(max_y-min_y)))
    
    plt.title('Event {0}. PID: {1}, momentum: {2:.3f} MeV'.format(event_n, pid, momentum))
    plt.show()

def plot_parameters(X, y, names):
    n_params = len(names)-1
    rows = math.floor(math.sqrt(n_params))
    cols = math.ceil(math.sqrt(n_params))
    
    fig = plt.figure(figsize=(cols*5, rows*3))

    for i in range(n_params):
        ax1 = plt.subplot(rows,cols,i+1)
        ax1.scatter(X[:,i], y, s=1)
        ax1.set_xlabel(names[i+1])
        ax1.set_ylabel(names[0])

    plt.tight_layout()
    plt.show()

def plot_regression(X, y, reg, names, degree=None):
    min_x = np.min(X)
    max_x = np.max(X)
    line = np.linspace(min_x, max_x).reshape(-1,1)

    plt.figure(figsize=(5, 3))
    plt.xlabel(names[1])
    plt.ylabel(names[0])
    plt.scatter(X, y, s=1)
    if degree==None:
        plt.plot(line, reg.intercept_ + reg.coef_[0] * line, c='red', label="Regression line")
    else:
        xs = PolynomialFeatures(degree=degree, include_bias=True).fit_transform(line)
        ys = reg.predict(xs)
        plt.plot(line, ys, c='red', label="Regression line")
    plt.legend()
    plt.show()

def plot_distributions(X, y, y_pred):
    x_min = min(y.min(), X.min())
    x_max = max(y.max(), y.max())

    fig = plt.figure(figsize=(10, 7))

    ax1 = plt.subplot(321)
    ax1.hist(y, bins=100, range=(x_min,x_max))
    ax1.title.set_text('Original')
    ax1.set_xlabel('momentum (MeV)')
    ax1.set_ylabel('frequency')

    ax2 = plt.subplot(322)
    ax2.hist(y, bins=100, range=(x_min,x_max))
    ax2.set_yscale('log')
    ax2.title.set_text('Original (log-scale)')
    ax2.set_xlabel('momentum (MeV)')
    ax2.set_ylabel('frequency')

    ax3 = plt.subplot(323, sharex=ax1, sharey=ax1)
    ax3.hist(y_pred, bins=100)
    ax3.title.set_text('Predicted')
    ax3.set_xlabel('momentum (MeV)')
    ax3.set_ylabel('frequency')

    ax4 = plt.subplot(324, sharex=ax2, sharey=ax2)
    ax4.hist(y_pred, bins=100)
    ax4.set_yscale('log')
    ax4.title.set_text('Predicted (log-scale)')
    ax4.set_xlabel('momentum (MeV)')
    ax4.set_ylabel('frequency')

    difference = y-y_pred

    ax5 = plt.subplot(325)
    ax5.hist(difference, bins=100)#, range=(x_min,x_max))
    ax5.title.set_text('Difference: original-predicted')
    ax5.text(0.8, 0.8, f'mean:{np.mean(difference):.2f}\nstd:{np.std(difference):.2f}', transform=ax5.transAxes)

    ax6 = plt.subplot(326)
    ax6.hist(difference, bins=100)#, range=(x_min,x_max))
    ax6.set_yscale('log')
    ax6.title.set_text('Difference: original-predicted (log-scale)')
    ax6.text(0.8, 0.8, f'mean:{np.mean(difference):.2f}\nstd:{np.std(difference):.2f}', transform=ax6.transAxes)

    plt.tight_layout()
    plt.show()
