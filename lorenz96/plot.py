import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_timeseries(times, states, fileName):
    x = times
    y = states[0]
    
    fig = plt.figure()
    ax = fig.add_subplot(311)
    ay = fig.add_subplot(312)
    az = fig.add_subplot(313)

    ax.plot(times, states[:,0])
    ay.plot(times, states[:,1])
    az.plot(times, states[:,2])

    plt.savefig(fileName + ".png", format="png", dpi=300)
    plt.savefig(fileName + ".eps", format="eps")
    plt.show()
    plt.close()

def plot_trajectory(states, fileName):
    fig = plt.figure() 
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel(r"$m_x$")
    ax.set_ylabel(r"$m_y$")
    ax.set_zlabel(r"$m_z$")

    ax.plot(states[:,0], states[:,1], states[:,2])

    plt.savefig(fileName + ".png", format="png", dpi=300)
    plt.savefig(fileName + ".eps", format="eps")
    plt.show()
    plt.close()


def plot_quantity(times, quantity, fileName):
    fig = plt.figure()
    #plt.title("Conserved values")
    ax = fig.add_subplot(111)
    ax.set_ylabel(r"$||{\bf m}||^2$")
    ax.set_xlabel("Time")

    ax.plot(times, quantity)

    plt.savefig(fileName + ".png", format="png", dpi=300)
    plt.savefig(fileName + ".eps", format="eps")
    plt.close()

def plot_multiTimeseries(times, states, fileName):    
    fig = plt.figure()
    ax = fig.add_subplot(311)
    ay = fig.add_subplot(312)
    az = fig.add_subplot(313)

    for i in range(states.shape[1]):
        ax.plot(times, states[:,i,0], lw=0.1)
        ay.plot(times, states[:,i,1], lw=0.1)
        az.plot(times, states[:,i,2], lw=0.1)

    plt.savefig(fileName + ".png", format="png", dpi=300)
    plt.savefig(fileName + ".eps", format="eps")
    plt.close()

def plot_multiTrajectory(states, fileName):
    for i in range(0):
        fig = plt.figure() 
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel(r"$m_x$")
        ax.set_ylabel(r"$m_y$")
        ax.set_zlabel(r"$m_z$")

        ax.plot(states[:, i, 0], states[:, i, 1], states[:, i, 2], lw=0.1)

        plt.savefig(fileName + "_" + str(i) + ".png", format="png", dpi=300)
        plt.savefig(fileName + "_" + str(i) + ".eps", format="eps")
        plt.close()

def plot_multiQuantity(times, quantity, fileName):
#conservative
    fig = plt.figure()
#plt.title("Conserved values")
    ax = fig.add_subplot(111)
    ax.set_ylabel(r"$||{\bf m}||^2$")
    ax.set_xlabel("Time")

    for i in range(quantity.shape[1]):
        ax.plot(times, quantity[:,i])

    plt.savefig(fileName + ".png", format="png", dpi=300)
    plt.savefig(fileName + ".eps", format="eps")
    plt.close()

def plot_timeseriesColorMap(states, fileName, region):     
    plt.ioff()
    plt.imshow(states.T, cmap=plt.cm.jet, interpolation="nearest", aspect=1/100.0,
               extent=(region[0], region[1], region[2], region[3]), vmin=-1, vmax=3) 
    #plt.imshow(states.T, cmap=plt.cm.jet, interpolation="nearest", aspect=6) 

    cbar = plt.colorbar(shrink=0.5)
    cbar.ax.tick_params(labelsize=18)
    plt.xticks(fontsize=20)

    plt.xlabel("")
    plt.ylabel("")
    plt.yticks([])

    plt.savefig(fileName + ".png", format="png", dpi=600)
    plt.savefig(fileName + ".eps", format="eps")
    plt.close()


def plot_accuracy(iteration, accuracy, fileName):
    x = iteration
    y = accuracy
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel("Iteration length")
    ax.set_ylabel("Accuracy")

    ax.plot(x, y)

    plt.savefig(fileName + ".png", format="png", dpi=300)
    plt.savefig(fileName + ".eps", format="eps")


# 混同行列をヒートマップで表示
def plot_mnistTable(y_test, y_pred, fileName):
    plt.ioff()
    plt.rcParams["font.size"] = 12
    plt.rcParams["font.family"] = "arial"
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm,  annot=True, fmt='d', cmap='gnuplot',linewidths=.5,robust=True,cbar=False,square=True)  # annotでセルに値を表示, fmt='d'で整数表示

    plt.xlabel("Test label")
    plt.ylabel("Prediction label")

    plt.savefig(fileName + ".png", format="png", dpi=300)
    plt.savefig(fileName + ".eps", format="eps")
    plt.close()

# # PCA scatter plot
# def plot_scatter(pca_df, fileName):
#     plt.ioff()
#     plt.rcParams["font.size"] = 12
#     plt.rcParams["font.family"] = "arial"
#     sns.FacetGrid(pca_df, hue="label").map(plt.scatter, '1st_principal', '2nd_principal')
    
#     plt.savefig(fileName + ".png", format="png", dpi=300)
#     plt.savefig(fileName + ".eps", format="eps")
#     plt.close()

# PCA scatter plot
def plot_scatter(features, labels, fileName):
    plt.ioff()
    plt.rcParams["font.size"] = 12
    plt.rcParams["font.family"] = "arial"
    #plt.set_aspect('equal')
    fig = plt.figure()
    ax = fig.add_subplot(111)

    #plt.gca().set_aspect('equal')
   
    for i in range(10):
        plt.scatter(features[labels == i,0], features[labels == i,1], alpha=0.8, s = 0.5)

    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')


    #plt.scatter(features[:,0], features[:,1], c=labels, cmap=cm, alpha=0.5, size=0.1)
    #sns.FacetGrid(pca_df, hue="label").map(plt.scatter, '1st_principal', '2nd_principal')
    
    plt.savefig(fileName + ".png", format="png", dpi=600)
    plt.savefig(fileName + ".eps", format="eps")
    plt.close()