import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale, PolynomialFeatures

def map_value(y,z):
    min_y = -257.56
    max_y = 317.56
    min_z = -2888.78
    max_z = -999.1

    y = int((y-min_y)//10)
    z = int((z-min_z)//10)

    return y, z

def plot_event(df, event_n):
    # Extracting data for the event
    event = df.iloc[event_n]
    pid = event['TruePID']
    momentum = event['TrueMomentum']
    X = event['NodePosX']
    Y = event['NodePosY']
    Z = event['NodePosZ']
    
    # Ranges of the detector
    min_x, max_x = -985.92, 985.92
    min_y, max_y = -257.56, 317.56
    min_z, max_z = -2888.78, -999.1
    
    # Create a figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot
    ax.scatter(X, Z, Y, marker='o', s=20, c='b', label='Data Points')
    
    # Set axis labels and ticks
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_zlabel('Y (mm)')
    
    # Set axis limits
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_z, max_z)
    ax.set_zlim(min_y, max_y)
    
    # Set axis ticks
    ax.set_xticks([-900, -600, -300, 0, 300, 600, 900])
    ax.set_yticks([-2750, -2250, -1750, -1250])
    ax.set_zticks([-300, -150, 0, 150, 300])
    
    # Auto scale the axis
    ax.auto_scale_xyz([min_x, max_x], [min_z, max_z], [min_y, max_y])
    
    # Set plot title
    plt.title('Event {0}. PID: {1}, momentum: {2:.3f} MeV'.format(event_n, pid, momentum))
    
    # Add a legend
    ax.legend()
    
    # Show the plot
    plt.show()

def plot_parameters(X, y, param_names, y_names, mode="reg"):
    if mode=="reg":
        n_params = len(param_names)-1
        rows = math.floor(math.sqrt(n_params))
        cols = math.ceil(math.sqrt(n_params))
    
        fig = plt.figure(figsize=(cols*5, rows*3))

        for i in range(n_params):
            ax1 = plt.subplot(rows,cols,i+1)
            ax1.scatter(X[:,i], y, s=1)
            ax1.set_xlabel(param_names[i+1])
            ax1.set_ylabel(param_names[0])
    else:
        n_params = len(param_names)
        rows = math.floor(math.sqrt(n_params))
        cols = math.ceil(math.sqrt(n_params))

        fig = plt.figure(figsize=(cols*5, rows*3))

        for i in range(n_params):
            X0 = X[y==0,i]
            X1 = X[y==1,i]
            X0 = X0[X0!=-1]
            X1 = X1[X1!=-1]
            ax1 = plt.subplot(rows,cols,i+1)
            ax1.hist(X0, bins=50, histtype='step', label=y_names[0])
            ax1.hist(X1, bins=50, histtype='step', label=y_names[1])
            if len(y_names)>2:
                X2 = X[y==2,i]
                X2 = X2[X2!=-1]
                ax1.hist(X2, bins=50, histtype='step', label=y_names[2])
            ax1.set_xlabel(param_names[i])
            ax1.set_ylabel("frequency")
            plt.legend()

    plt.tight_layout()
    plt.show()

def plot_params_pid(X, y, param_names, y_names):
    fig = plt.figure(figsize=(5, 3))
    
    ax1 = plt.subplot(1,1,1)
    ax1.scatter(X[y==0,0], X[y==0,1], c="b", s=1, label=y_names[0])
    ax1.scatter(X[y==1,0], X[y==1,1], c="r", s=1, label=y_names[1])
    ax1.set_xlabel(param_names[0])
    ax1.set_ylabel(param_names[1])
    plt.legend()

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
    ax1.set_xlabel('momentum [MeV/c]')
    ax1.set_ylabel('frequency')

    ax2 = plt.subplot(322)
    ax2.hist(y, bins=100, range=(x_min,x_max))
    ax2.set_yscale('log')
    ax2.title.set_text('Original (log-scale)')
    ax2.set_xlabel('momentum [MeV/c]')
    ax2.set_ylabel('frequency')

    ax3 = plt.subplot(323, sharex=ax1, sharey=ax1)
    ax3.hist(y_pred, bins=100)
    ax3.title.set_text('Predicted')
    ax3.set_xlabel('momentum [MeV/c]')
    ax3.set_ylabel('frequency')

    ax4 = plt.subplot(324, sharex=ax2, sharey=ax2)
    ax4.hist(y_pred, bins=100)
    ax4.set_yscale('log')
    ax4.title.set_text('Predicted (log-scale)')
    ax4.set_xlabel('momentum [MeV/c]')
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

def plot_logistic_regression(log_reg, X, y, param_names, y_names):
    min_x = np.amin(X, axis=0)
    max_x = np.amax(X, axis=0)

    fig = plt.figure(figsize=(10, 3))

    ax1 = plt.subplot(1,2,1)
    ax1.scatter(X[y==0,0], X[y==0,1], c="b", s=1, label=y_names[0])
    ax1.scatter(X[y==1,0], X[y==1,1], c="r", s=1, label=y_names[1])
    line = np.linspace(min_x[0], max_x[0])
    ax1.plot(line, -(line * log_reg.coef_[0][0] + log_reg.intercept_) / log_reg.coef_[0][1], c='lime', linewidth=5, label="separation")
    ax1.set_xlabel(param_names[0])
    ax1.set_ylabel(param_names[1])
    ax1.set_xlim(min_x[0], max_x[0])
    ax1.set_ylim(min_x[1], max_x[1])

    plt.legend()

    ax2 = plt.subplot(1,2,2)
    xx, yy = np.meshgrid(np.linspace(min_x[0], max_x[0]), np.linspace(min_x[1], max_x[1]))
    Z = log_reg.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,0]
    Z = Z.reshape(xx.shape)
    ax2.pcolormesh(xx, yy, Z, cmap=plt.cm.bwr_r)
    ax2.scatter(X[:,0], X[:,1], c='w', marker='x')
    ax2.set_xlabel(param_names[0])
    ax2.set_ylabel(param_names[1])
    ax2.set_xlim(min_x[0], max_x[0])
    ax2.set_ylim(min_x[1], max_x[1])

    plt.tight_layout()
    plt.show()

def plot_sigmoid():
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    x = np.linspace(-10,10,100)
    y = sigmoid(x)

    fig = plt.figure(figsize=(5, 3))
    plt.plot(x,y)
    plt.xlabel('z')
    plt.ylabel('sigmoid(z)')
    plt.show()

def print_conf(conf, target_names):
    assert len(target_names) == conf.shape[0]

    for i in range(len(target_names)+1):
        for j in range(len(target_names)+1):
            if i==0:
                if j==0:
                    print("\t\t", end = '')
                elif j<=len(target_names)-1:
                    print("True {}\t".format(target_names[j-1]), end = '')
                else:
                    print("True {}".format(target_names[j-1]))
            else:
                if j==0:
                    print("Pred {}\t".format(target_names[i-1]), end = '')
                elif j<=len(target_names)-1:
                    print("{:>{x}}\t".format(conf[i-1,j-1], x=5+len(target_names[j-1])), end = '')
                else:
                    print("{:>{x}}".format(conf[i-1,j-1], x=5+len(target_names[j-1])))

def plot_projection(df, event_n, Xf):
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
    
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(X, Z, Y)
    ax1.set_xlabel('X [mm]', labelpad=8)
    ax1.set_xticks([-900, -600, -300, 0, 300, 600, 900])
    ax1.set_ylabel('Z [mm]', labelpad=10)
    ax1.set_yticks([-2750, -2250, -1750, -1250])
    ax1.set_zlabel('Y [mm]', labelpad=5)
    ax1.set_zticks([-300, -150, 0, 150, 300])
    ax1.set_xlim(min_x, max_x)
    ax1.set_ylim(min_z, max_z)
    ax1.set_zlim(min_y, max_y)
    ax1.auto_scale_xyz([min_x, max_x], [min_z, max_z], [min_y, max_y])
    ax1.view_init(elev=10, azim=-30)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(X, Z, Y)
    ax2.set_xlabel('X [mm]', labelpad=8)
    ax2.set_xticks([-900, -600, -300, 0, 300, 600, 900])
    ax2.set_ylabel('Z [mm]', labelpad=10)
    ax2.set_yticks([-2750, -2250, -1750, -1250])
    ax2.set_zlabel('Y [mm]', labelpad=5)
    ax2.set_zticks([-300, -150, 0, 150, 300])
    ax2.set_xlim(min_x, max_x)
    ax2.set_ylim(min_z, max_z)
    ax2.set_zlim(min_y, max_y)
    ax2.auto_scale_xyz([min_x, max_x], [min_z, max_z], [min_y, max_y])
    ax2.view_init(elev=10, azim=30)
  
    ax1.set_title('Event {0}. PID: {1}, momentum: {2:.3f} [MeV/c]'.format(event_n, pid, momentum), loc='center')
    plt.show()     

    cmap = plt.cm.get_cmap("Reds")
    cmap.set_bad(color='white')

    # mask some 'bad' data, in your case you would have: data == 0
    cand_image = np.ma.masked_where(Xf[event_n] == 0.0, Xf[event_n])

    plt.matshow(cand_image.reshape(58,189), cmap=cmap)
    plt.xlabel("Z [cm]")
    plt.ylabel("Y [cm]")

    #plt.title(r'{0} sample: N={1}, $\mu={2:.2f}$, $\sigma={3:.2f}$'.format(prefix, len(distances), np.mean(distances), np.std(distances)))
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('dedx',fontsize=10, rotation=90)
    plt.title("YZ projection", y=1.15)
    plt.gca().invert_yaxis()
    plt.show()
