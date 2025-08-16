# From this stackoverflow page and modified: https://stackoverflow.com/questions/44985966/managing-dynamic-plotting-in-matplotlib-animation-module/44989063#44989063
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.axes_grid1
import matplotlib.widgets


class Player(FuncAnimation):
    def __init__(
        self,
        fig,
        func,
        frames=None,
        init_func=None,
        fargs=None,
        save_count=None,
        mini=0,
        maxi=100,
        pos=(0.125, 0.92),
        **kwargs,
    ):
        self.i = 0
        self.min = mini
        self.max = maxi
        self.runs = True
        self.forwards = True
        self.fig = fig
        self.func = func
        self.setup(pos)
        FuncAnimation.__init__(
            self,
            self.fig,
            self.func,
            frames=self.play(),
            init_func=init_func,
            fargs=fargs,
            save_count=save_count,
            **kwargs,
        )

    def play(self):
        while self.runs:
            self.i = self.i + self.forwards - (not self.forwards)
            if self.i > self.min and self.i < self.max:
                yield self.i
            # elif self.i == self.max and self.forwards:
            #     self.i = self.min
            # elif self.i == self.min and not self.forwards:
            #     self.i = self.max
            else:
                self.stop()
                yield self.i

    def start(self):
        if (
            self.forwards
            and self.i < self.max
            or not self.forwards
            and self.i > self.min
        ):
            self.runs = True
            self.event_source.start()

    def stop(self, event=None):
        self.runs = False
        self.event_source.stop()

    def forward(self, event=None):
        self.forwards = True
        self.start()

    def backward(self, event=None):
        self.forwards = False
        self.start()

    def oneforward(self, event=None):
        self.forwards = True
        self.onestep()

    def onebackward(self, event=None):
        self.forwards = False
        self.onestep()

    def onestep(self):
        if self.i > self.min and self.i < self.max:
            self.i = self.i + self.forwards - (not self.forwards)
        elif self.i == self.min and self.forwards:
            self.i += 1
        elif self.i == self.max and not self.forwards:
            self.i -= 1
        elif self.i == self.max and self.forwards:
            self.i = self.min
        elif self.i == self.min and not self.forwards:
            self.i = self.max
        self.func(self.i)
        self.fig.canvas.draw_idle()

    def setup(self, pos):
        playerax = self.fig.add_axes([pos[0], pos[1], 0.22, 0.04])
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(playerax)
        bax = divider.append_axes("right", size="80%", pad=0.05)
        sax = divider.append_axes("right", size="80%", pad=0.05)
        fax = divider.append_axes("right", size="80%", pad=0.05)
        ofax = divider.append_axes("right", size="100%", pad=0.05)
        self.button_oneback = matplotlib.widgets.Button(playerax, label="$\u29cf$")
        self.button_back = matplotlib.widgets.Button(bax, label="$\u25c0$")
        self.button_stop = matplotlib.widgets.Button(sax, label="$\u25a0$")
        self.button_forward = matplotlib.widgets.Button(fax, label="$\u25b6$")
        self.button_oneforward = matplotlib.widgets.Button(ofax, label="$\u29d0$")
        self.button_oneback.on_clicked(self.onebackward)
        self.button_back.on_clicked(self.backward)
        self.button_stop.on_clicked(self.stop)
        self.button_forward.on_clicked(self.forward)
        self.button_oneforward.on_clicked(self.oneforward)
