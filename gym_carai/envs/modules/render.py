import pyglet
import numpy as np


class Rect:
    def __init__(self, x, y, rotation, w, h, batch, color=(255, 0, 0)):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.rotation = rotation
        self.batch = batch
        self.deleted = None

        if len(color) == 3:
            self.color = color * 4
        elif len(color) == 6:
            self.color = color * 2
        elif len(color) == 12:
            self.color = color
        else:
            raise Exception("Color vector wrong size!")

        # X,Y are centre
        self.vertices = (0, 0)*4
        self.v = self.batch.add(4, pyglet.gl.GL_QUADS, None,
                                ('v2f', self.vertices),
                                ('c3B', self.color)
                                )
        self.update_vertices_c()

    def update_pos(self, xc=None, yc=None, rotation=None, w=None, h=None):
        if w is not None:
            self.w = w
        if h is not None:
            self.h = h
        if xc is not None:
            self.x = xc
        if yc is not None:
            self.y = yc
        if rotation is not None:
            self.rotation = rotation
        self.update_vertices_c()

    def update_color(self, color):
        if len(color) == 3:
            self.color = color * 4
        elif len(color) == 6:
            self.color = color * 2
        elif len(color) == 12:
            self.color = color
        else:
            raise Exception("Color vector wrong size! Got {}, expected multiple of 3".format(len(color)))
        self.v.color[:] = self.color

    def update_vertices_c(self):
        if self.deleted:
            self.delete()
            return
        theta_rad = np.deg2rad(self.rotation)
        cr = np.cos(theta_rad)
        sr = np.sin(theta_rad)
        # upper left
        x1 = self.x - cr * self.w / 2 + sr * self.h / 2
        y1 = self.y + cr * self.h / 2 + sr * self.w / 2
        # upper right
        x2 = self.x + cr * self.w / 2 + sr * self.h / 2
        y2 = self.y + cr * self.h / 2 - sr * self.w / 2
        # lower right
        x3 = self.x + cr * self.w / 2 - sr * self.h / 2
        y3 = self.y - cr * self.h / 2 - sr * self.w / 2
        # lower left
        x4 = self.x - cr * self.w / 2 - sr * self.h / 2
        y4 = self.y - cr * self.h / 2 + sr * self.w / 2
        self.vertices = (x1, y1, x2, y2, x3, y3, x4, y4)
        self.v.vertices[:] = self.vertices

    def delete(self):
        # TODO: Improve
        self.v.vertices[:] = (-50, -45)*4

class Line:
    def __init__(self, batch, x1, y1, x2, y2, thickness=None, color=(0, 255, 0)):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.deleted = None
        if self.x1 != self.x2:
            self.rotation = np.rad2deg(np.arctan((y2 - y1) / (x2 - x1)))
        else:
            self.rotation = 90.0
        self.length = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        self.batch = batch
        self.vertices = (self.x1, self.y1,
                         self.x2, self.y2)
        if thickness:
            pyglet.gl.glLineWidth(thickness)

        if len(color) == 3:
            self.color = color * 2
        elif len(color) == 6:
            self.color = color
        else:
            raise Exception("Color vector wrong size!")

        self.v = self.batch.add(2, pyglet.gl.GL_LINES, None,
                                ('v2f', self.vertices),
                                ('c3B', self.color)
                                )

    def update_position(self, x1, y1, x2, y2):
        if self.deleted:
            self.delete()
            return
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.length = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        self.vertices = (self.x1, self.y1,
                         self.x2, self.y2)
        self.v.vertices[:] = self.vertices

    def update_position_rot(self, theta, xc=None, yc=None):
        if xc is None:
            xc = (self.x1 + self.x2) / 2
        if yc is None:
            yc = (self.y1 + self.y2) / 2
        self.rotation = theta
        theta_rad = np.deg2rad(theta)
        cr = np.cos(theta_rad)
        sr = np.sin(theta_rad)
        self.x1 = xc + cr * self.length / 2
        self.x2 = xc - cr * self.length / 2
        self.y1 = yc + sr * self.length / 2
        self.y2 = yc - sr * self.length / 2
        self.vertices = (self.x1, self.y1,
                         self.x2, self.y2)
        self.v.vertices[:] = self.vertices

    def update_color(self, color):
        if len(color) == 3:
            self.color = color * 2
        elif len(color) == 6:
            self.color = color
        else:
            raise Exception("Color vector wrong size! Got {}, expected multiple of 3".format(len(color)))
        self.v.color[:] = self.color

    def delete(self):
        self.deleted = True
        self.v.vertices[:] = (-50,-50,-50,-50)