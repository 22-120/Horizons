import os
import matplotlib.pyplot as plt
import math
import numpy as np
from datetime import datetime

# PyQt5 Tab Window -------------------------------------------------------------------------

import matplotlib
matplotlib.use('qt5agg')

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvas as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import matplotlib.pyplot as plt

class plotWindow():
    def __init__(self, parent=None):
        self.app = QApplication(sys.argv)
        self.MainWindow = QMainWindow()
        self.MainWindow.__init__()
        self.MainWindow.setWindowTitle("plot window")
        self.canvases = []
        self.figure_handles = []
        self.toolbar_handles = []
        self.tab_handles = []
        self.current_window = -1
        self.tabs = QTabWidget()
        self.MainWindow.setCentralWidget(self.tabs)
        self.MainWindow.resize(1280, 900)
        self.MainWindow.show()

    def addPlot(self, title, figure):
        new_tab = QWidget()
        layout = QVBoxLayout()
        new_tab.setLayout(layout)

        figure.subplots_adjust(left=0.05, right=0.99, bottom=0.05, top=0.91, wspace=0.2, hspace=0.2)
        new_canvas = FigureCanvas(figure)
        new_toolbar = NavigationToolbar(new_canvas, new_tab)

        layout.addWidget(new_canvas)
        layout.addWidget(new_toolbar)
        self.tabs.addTab(new_tab, title)

        self.toolbar_handles.append(new_toolbar)
        self.canvases.append(new_canvas)
        self.figure_handles.append(figure)
        self.tab_handles.append(new_tab)

    def show(self):
        self.app.exec_()

# Text Parsing -----------------------------------------------------------------------------

CURRENT = os.getcwd()
DIR = os.path.join(CURRENT, 'Preprocessed_results')

file_list = []
for curDir, dirs, files in os.walk(DIR):
    for file in files:
        #print(os.path.join(curDir, file))
        file_list.append(os.path.join(curDir, file))

planetary_positions = {}
planetary_physical_data = {}
for file in file_list:
    with open(file, 'r') as txt:
        txt_name = os.path.basename(file)
        planetary_positions[txt_name] = []
        planetary_physical_data[txt_name] = {}
        lines = txt.readlines() # List of each lines in txt file
        for line in lines:
            if line[0] == "*":
                data = line[1:].strip().split(" ")
                if data[2] == 'float':
                    planetary_physical_data[txt_name][data[0]] = float(data[1])
                elif data[2] == 'string':
                    planetary_physical_data[txt_name][data[0]] = data[1]
            else:
                data = line.strip().split(" ")

                position = {}
                position["date"] = data[0]
                position["time"] = data[1]
                position["R.A."] = list(map(float, data[2:5]))
                position["DEC"] = list(map(float, data[5:8])) + [data[5][0]]
                planetary_positions[txt_name].append(position)
print(planetary_physical_data)

# 3D Plotting ------------------------------------------------------------------------------

def equatorial_position(ra, dec, pos, radius):
    x = pos[0] + radius * math.cos(ra) * math.cos(dec)
    y = pos[1] + radius * math.sin(ra) * math.cos(dec)
    z = pos[2] + radius * math.sin(dec)
    return np.array([x, y, z])

def earth_position(date, offset=0):
    vernal_equinox = '2023-03-20'
    theta = 2 * math.pi * (elapsed_days(vernal_equinox, date) + offset) / 365.2
    inclination = 23.5 * math.pi / 180
    x = math.cos(theta)
    y = math.sin(theta) * math.cos(inclination)
    z = -math.sin(theta) * math.sin(inclination)
    return np.array([x, y, z])

def init_graph(i, ax, title=''):
    fontlabel = {"fontsize":"large", "color":"gray", "fontweight":"bold"}
    ax.set_title(title)
    ax.set_xlabel("X{}".format(i), fontdict=fontlabel, labelpad=16)
    ax.set_ylabel("Y{}".format(i), fontdict=fontlabel, labelpad=16)
    ax.set_zlabel("Z{}".format(i), fontdict=fontlabel, labelpad=16)
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    ax.view_init(elev=30*i, azim=120*i) 

def calc_positions_sun(txt, radius, pos = np.array([0, 0, 0])):
    # Basic data
    planetary_data = planetary_positions[txt]
    start_date = planetary_physical_data[txt]['start-date']

    # Planetary position transform
    equatorial = list(map(lambda x : (calc_ra(x["R.A."]) + math.pi, calc_dec(x["DEC"][:-1], sign=x["DEC"][-1])), planetary_data))
    earths = [earth_position(start_date, i) for i in range(len(equatorial))]
    positions = list(map(lambda x : raycast_inside_sphere(earth_position(start_date, x[0]), equatorial_position(x[1][0], x[1][1], pos, 1), radius), enumerate(equatorial)))

    # Split position for each axis
    xdata = list(map(lambda x : x[0], positions))
    zdata = list(map(lambda x : x[2], positions))
    ydata = list(map(lambda x : x[1], positions))

    ex = list(map(lambda x : x[0], earths))
    ey = list(map(lambda x : x[1], earths))
    ez = list(map(lambda x : x[2], earths))

    return xdata, ydata, zdata, ex, ey, ez

def calc_positions_earth(planetary_data, pos = np.array([0, 0, 0]), radius = 1):
    # Planetary position transform
    #original = list(map(lambda x : (x["R.A."], x["DEC"]), planetary_data))
    equatorial = list(map(lambda x : (calc_ra(x["R.A."]), calc_dec(x["DEC"][:-1], sign=x["DEC"][-1])), planetary_data))
    positions = list(map(lambda x : equatorial_position(x[0], x[1], pos, radius), equatorial))

    # Split position for each axis
    xdata = list(map(lambda x : x[0], positions))
    ydata = list(map(lambda x : x[1], positions))
    zdata = list(map(lambda x : x[2], positions))

    return xdata, ydata, zdata

def calc_ra(angles:list, start=0):
    ret = 0
    for i, angle in enumerate(angles):
        ret += abs(angle) / 60**(i + start)
    ret *= 15 * math.pi / 180
    return ret if angles[0] >= 0 else -ret

def calc_dec(angles:list, sign='+', start=0): 
    ret = 0
    for i, angle in enumerate(angles):
        ret += abs(angle) / 60**(i + start)
    ret *= math.pi / 180
    return ret if sign == '+' else -ret

def scatter_position_sun(ax, txt):

    # Draw Target's position and Earth's position
    orbit_radius = planetary_physical_data[txt]['semi-major-axis']
    x, y, z, ex, ey, ez = calc_positions_sun(txt, orbit_radius)
    ax.scatter(x, y, z, c=z, cmap='inferno', s=5, alpha=0.5)
    ax.scatter(ex, ey, ez, c=z, cmap='inferno', s=5, alpha=0.5)

    # Draw Plane approximated from Target's position
    plane_target = fit_to_plane(x, y, z)
    plane_earth = fit_to_plane(ex, ey, ez)

    normal_target = np.array([-plane_target[0,0], -plane_target[1,0], 1])
    normal_target /= np.linalg.norm(normal_target)
    normal_earth = np.array([-plane_earth[0,0], -plane_earth[1,0], 1])
    normal_earth /= np.linalg.norm(normal_earth)

    print(txt.split(".")[0], math.acos(np.dot(normal_target, normal_earth)) * 180 / math.pi)

    #plot_plane(ax, plane_target)
    #plot_plane(ax, plane_earth)

    #print("{}'s normal : {}, Earth's normal : {}".format(txt, normal_target, normal_earth))


def scatter_position_earth(ax, planetary_data, pos = np.array([0, 0, 0]), radius = 1):
    x, y, z = calc_positions_earth(planetary_data, pos, radius)
    ax.scatter(x, y, z, c=z, cmap='inferno', s=5, alpha=0.5)

def raycast_inside_sphere(origin, ray, radius): # np.array
    dot_p_dir = np.dot(origin, ray)
    p_square = np.linalg.norm(origin)**2
    l = -dot_p_dir + math.sqrt(dot_p_dir**2 - p_square + radius**2)
    return origin + l * ray / np.linalg.norm(ray)

def elapsed_days(start_date, end_date):
    d1 = datetime.strptime(start_date, "%Y-%m-%d")
    d2 = datetime.strptime(end_date, "%Y-%m-%d")

    delta = d2 - d1
    return delta.days

def fit_to_plane(x, y, z):
    tmp_A = []; tmp_b = []
    for i in range(len(x)):
        tmp_A.append([x[i], y[i], 1])
        tmp_b.append(z[i])
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)

    fit = (A.T * A).I * A.T * b
    return fit

def plot_plane(ax, fit):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
                    np.arange(ylim[0], ylim[1]))
    Z = np.zeros(X.shape)
    for r in range(X.shape[0]):
        for c in range(X.shape[1]):
            Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]
    ax.plot_wireframe(X,Y,Z, color='k')

#fig, axs = plt.subplots(ncols=len(planetary_positions), figsize=(10, 3), subplot_kw={'projection':'3d'})
#
#for a, txt in zip(enumerate(axs if hasattr(axs, '__iter__') else [axs]), planetary_positions):
#    i = a[0]; ax = a[1]
#    init_graph(i, ax, txt.split(".")[0])
#    scatter_position(ax, planetary_positions[txt])

pw = plotWindow()

for i, txt in enumerate(planetary_positions):
    fig = plt.figure(figsize=(10, 3))
    title = txt.split(".")[0]
    ax1 = fig.add_subplot(121, projection='3d')
    init_graph(i, ax1, title)
    scatter_position_sun(ax1, txt)

    ax2 = fig.add_subplot(122, projection='3d')
    init_graph(i, ax2, title)
    scatter_position_earth(ax2, planetary_positions[txt])

    pw.addPlot(title, fig)

pw.show()


