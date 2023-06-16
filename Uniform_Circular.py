import sys
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtWidgets import QLabel, QLineEdit, QPushButton, QWidget, QSlider
from PyQt6.QtWidgets import QGridLayout, QVBoxLayout, QHBoxLayout
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvas as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import Slider

import numpy as np
import time
import math

class Example(QMainWindow):
    def __init__(self):
        super().__init__()

        # Pyplot Figure Canvas
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)

        layout = QHBoxLayout(self.main_widget)
        #self.addToolBar(NavigationToolbar(dynamic_graph, self))

        dynamic_canvas = FigureCanvas(Figure(figsize=(6, 6)))
        #dynamic_canvas.figure.tight_layout()
        layout.addWidget(dynamic_canvas)

        self.simulation_2d = dynamic_canvas.figure.add_axes([0.13, 0.3, 0.7, 0.58])
        self.timer1 = dynamic_canvas.new_timer(100, [(self.update_simulation, (), {})])
        self.timer1.start()

        slider_r1_axis = dynamic_canvas.figure.add_axes([0.25, 0.17, 0.5, 0.03])
        #slider_r1_axis.axis([0.25, 0.4, 0.65, 0.03])
        self.slider_r1 = Slider(slider_r1_axis, label="inner radius", valmin=0.2, valmax=3, valinit=1.49)
        self.slider_r1.on_changed(self.set_r1)

        slider_r2_axis = dynamic_canvas.figure.add_axes([0.25, 0.1, 0.5, 0.03])
        self.slider_r2 = Slider(slider_r2_axis, label="outer radius", valmin=0.2, valmax=3, valinit=2.25)
        self.slider_r2.on_changed(self.set_r2)

        dynamic_graph = FigureCanvas(Figure(figsize=(6, 6)))
        layout.addWidget(dynamic_graph)

        self.calc_theta = dynamic_graph.figure.subplots()
        self.timer2 = dynamic_graph.new_timer(100, [(self.update_theta, (), {})])
        self.timer2.start()

        self.interval = 400 #days
        self.theta = []

        self.earth = 0
        self.mars = 0

        self.t1 = 365.3
        self.t2 = 687

        self.w1 = 2 * math.pi * 1/self.t1
        self.w2 = 2 * math.pi * 1/self.t2

        self.sa1 = 0
        self.sa2 = 0

        self.r1 = 1.49
        self.r2 = 2.25

        self.t = 0
        self.start_time = time.time()

        # Text Input
        vsublayout = QVBoxLayout(self.main_widget)
        layout.addLayout(vsublayout)
        sublayout1 = QHBoxLayout(self.main_widget)
        sublayout2 = QHBoxLayout(self.main_widget)
        vsublayout.addLayout(sublayout1)
        vsublayout.addLayout(sublayout2)
        self.input_start_angle1 = QLineEdit()
        sublayout1.addWidget(self.input_start_angle1, alignment=Qt.AlignmentFlag.AlignCenter)

        self.button_set_angle1 = QPushButton("SET")
        self.button_set_angle1.clicked.connect(self.set_start_angle1)
        sublayout1.addWidget(self.button_set_angle1)

        self.input_start_angle2 = QLineEdit()
        sublayout1.addWidget(self.input_start_angle2, alignment=Qt.AlignmentFlag.AlignCenter)

        self.button_set_angle2 = QPushButton("SET")
        self.button_set_angle2.clicked.connect(self.set_start_angle2)
        sublayout1.addWidget(self.button_set_angle2)

        self.button_restart = QPushButton("RESTART")
        self.button_restart.clicked.connect(self.restart)
        vsublayout.addWidget(self.button_restart)

        self.input_t1 = QLineEdit()
        sublayout2.addWidget(self.input_t1, alignment=Qt.AlignmentFlag.AlignCenter)

        self.button_set_t1 = QPushButton("SET")
        self.button_set_t1.clicked.connect(self.set_t1)
        sublayout2.addWidget(self.button_set_t1)

        self.input_t2 = QLineEdit()
        sublayout2.addWidget(self.input_t2, alignment=Qt.AlignmentFlag.AlignCenter)

        self.button_set_t2 = QPushButton("SET")
        self.button_set_t2.clicked.connect(self.set_t2)
        sublayout2.addWidget(self.button_set_t2)

        self.setGeometry(100, 30, 1500, 530)
        self.setWindowTitle("Uniform Circular Approximation")
        self.show()

    def set_start_angle1(self):
        sa1 = float(self.input_start_angle1.text())
        self.sa1 = sa1 * math.pi / 180

    def set_start_angle2(self):
        sa2 = float(self.input_start_angle2.text())
        self.sa2 = sa2 * math.pi / 180

    def set_r1(self, value):
        self.r1 = value
    
    def set_r2(self, value):
        self.r2 = value

    def set_t1(self):
        self.t1 = float(self.input_t1.text())
        self.w1 = 2 * math.pi / self.t1

    def set_t2(self):
        self.t2 = float(self.input_t2.text())
        self.w2 = 2 * math.pi / self.t2

    def update_simulation(self):
        self.simulation_2d.clear()
        self.simulation_2d.set_xlim([-3, 3])
        self.simulation_2d.set_ylim([-3, 3])

        self.t = 10 * (time.time() - self.start_time)

        # Draw circular orbits
        n = 120
        orbit1 = [[self.r1 * math.cos(2 * math.pi / n * i) for i in range(n+1)], [self.r1 * math.sin(2 * math.pi / n * i) for i in range(n+1)]]
        orbit2 = [[self.r2 * math.cos(2 * math.pi / n * i) for i in range(n+1)], [self.r2 * math.sin(2 * math.pi / n * i) for i in range(n+1)]]
        self.simulation_2d.plot(orbit1[1], orbit1[0])
        self.simulation_2d.plot(orbit2[1], orbit2[0])

        # Draw Planets
        self.earth = np.array([self.r1 * math.cos(self.sa1 + self.w1 * self.t), self.r1 * math.sin(self.sa1 + self.w1 * self.t)])
        self.mars = np.array([self.r2 * math.cos(self.sa2 + self.w2 * self.t), self.r2 * math.sin(self.sa2 + self.w2 * self.t)])
        self.simulation_2d.plot([self.earth[0], self.mars[0]], [self.earth[1], self.mars[1]], '.')

        self.simulation_2d.figure.canvas.draw()

    def update_theta(self):
        self.calc_theta.clear()
        self.calc_theta.set_ylim([-190, 190])

        _dir = self.mars - self.earth
        _dir = _dir / np.linalg.norm(_dir)

        arccosine = math.acos(_dir[0])
        arcsine = math.asin(_dir[1])

        if _dir[1] >= 0 and _dir[0] < 0:
            theta = arccosine
        if _dir[1] >= 0 and _dir[0] >= 0:
            theta = arcsine
        if _dir[1] < 0 and _dir[0] >= 0:
            theta = arcsine
        if _dir[1] < 0 and _dir[0] < 0:
            theta = -arccosine

        time = np.linspace(max(0, self.t - self.interval), self.t, min(self.interval, len(self.theta)+1))
        self.theta.append(theta * 180 / math.pi)

        self.calc_theta.plot(time, self.theta[-self.interval:])
        self.calc_theta.figure.canvas.draw()

    def restart(self):
        self.start_time = time.time()
        self.t = 0
        self.theta = []

        self.timer1.start()
        self.timer2.start()

app = QApplication(sys.argv)
ex1 = Example()
sys.exit(app.exec())