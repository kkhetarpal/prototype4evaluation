import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import json

HEIGHT = 5
WIDTH = 10

def single_plot_data(data_x, data_y, xlabel, ylabel):
    """ data, names are lists of vectors """
    plt.plot(data_x, data_y, '-', markersize=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def multi_plot_data(data_x, data_y, names, xlabel, ylabel):
    """ data, names are lists of vectors """
    plt.figure(figsize=(WIDTH,HEIGHT))
    for i, y in enumerate(data_x):
        plt.plot(data_x[i], data_y[i], '-', markersize=1, label=names[i])
    plt.legend(prop={'size': 12}, numpoints=3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def plot_data_smooth(data_x, data_y, names, xlabel, ylabel, location,\
                                                            smoothing_window):
    """
    Plot mean curve on top of shaded high variance curve, tensorboard style
    Args:
        data_x: x-axis data, (list of np arrays)
        data_y: list of np arrays, y-axis
        names: your curve labels (list of strings)
        xlabel: x-axis title
        ylabel: y-axis title
        location: legend location
        smoothing_window: mean over how many data points
    """
    for i, y in enumerate(data_x):
        p = plt.plot(data_x[i], data_y[i], '-', alpha=0.25, markersize=1,label='_nolegend_')
        c = p[0].get_color()
        df = pd.DataFrame(data_y[i])
        smoothed = df[0].rolling(smoothing_window).mean()
        plt.plot(data_x[i], smoothed, '-',markersize=0.5, label=names[i],color=c)
    plt.legend(loc=location, prop={'size': 12}, numpoints=3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def csv_to_numpy(path):
    df=pd.read_csv(path, sep=',',header=0)
    val = df.values
    x, y = val[:,1], val[:,2]
    return x, y

def json_to_numpy(path):
    with open(path, "r") as read_file:
        data = json.load(read_file)
    data = data["data"]["rewards"]

    # make list of np arrays
    for i in range(len(data)):
        data[i] = np.array(data[i])
    return data

def funky_csv_to_numpy(path):
    data = []
    with open(path) as f:
        for line in f:
            l = line.rstrip('\n').split(",")
            reward = float(l[4].split(":")[1])
            data.append(reward)
    return np.array(data)

def plot_data(args):
    # Load rewards
    if args.repeated_headers:
        data_y = funky_csv_to_numpy(args.file)
    elif args.type == "csv":
        data_y = csv_to_numpy(args.file)
    else:
        data_y = json_to_numpy(args.file)

    # Plot
    plt.clf()
    plt.figure(figsize=(WIDTH,HEIGHT))
    xlabel="time steps"
    ylabel="average return"

    # Single plot
    if type(data_y) != list:
        data_x = np.arange(0,len(data_y))
        single_plot_data(data_x, data_y, xlabel, ylabel)
    # Plot multiple
    else:
        names = []
        data_x = []
        for i in range(len(data_y)):
            names.append(i)
            x = np.arange(0,len(data_y[i]))
            data_x.append(x)
        multi_plot_data(data_x, data_y,names,"time steps", "average return")

    # Save figure
    plt.savefig(args.save_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("file", help="file to plot")

    types = ["csv", "json"]
    parser.add_argument("--type", help="csv or json", choices=types)

    parser.add_argument("--repeated_headers", action='store_true',
        default=False, help="headers appear at each row")

    parser.add_argument("--save_name", default="figure.png")

    args = parser.parse_args()
    plot_data(args)



