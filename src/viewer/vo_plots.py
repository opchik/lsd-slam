import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.models import App


def extract_data_from_data(path: str):
    data = pd.read_table(path, header=None, sep=r"\s+")
    return data


def create_real_data_plot(x, y): #, z: list[float]):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y, linestyle="--")
    plt.show()


def create_plots(app: App):
    # data = extract_data_from_data(app.real_data)
    # print(data.iloc[:, 1:])
    # create_real_data_plot(data.iloc[:, 1], data.iloc[:, 2])
    fig = plt.figure(figsize=(15, 15))
    ax_3d = fig.add_subplot(projection='3d')
    ax_3d.scatter(app.points["x"], app.points["y"], app.points["z"], s=0.1)
    # ax_3d.scatter(1, 1, 1, s=0.1)
    app.points = dict()
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.set_title('Облако точек (алгоритм ORB)')
    ax_3d.set_xlim([-2000, 1000])
    ax_3d.set_ylim([-1000, 1000])
    ax_3d.set_zlim([-6000, -500])
     
    plt.show()