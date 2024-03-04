import numpy as np
import matplotlib.pyplot as plt

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
#conservative
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