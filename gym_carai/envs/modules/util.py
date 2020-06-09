import numpy as np
import pyglet
from pyglet.gl import *


def center_image(image):
    """ Sets an image's anchor point to the center """
    image.anchor_x = image.width // 2
    image.anchor_y = image.height // 2


def vector_length(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)


def line_overlapping(line1, line2):
    """ Where each line(segment) is a list containing a start and end position
    e.g. line1 = [x1 y1 x2 y2]
         line1 = [x3 y3 x4 y4]
         get_distance determines whether or not to return the distance, = 0 reduces unnecessary computations
         get_distance = 1 returns the distance between the center of line 1 and its intersection with line 2"""
    p = line1[0:2]
    r = line1[2:4] - p

    q = line2[0:2]
    s = line2[2:4] - q

    rs = np.cross(r, s)
    if rs == 0:
        return False, 0, 0, 0, 0, 0, 0
    qpr = np.cross((q-p), r)

    t = np.cross((q-p), s)/rs

    u = qpr/np.cross(r, s)

    if rs == 0 and qpr == 0:
        # collinear
        t0 = qpr / np.cross(r, r)
        t1 = t0 + np.cross(s, np.cross(r, r))
        if np.dot(s, r) > 0:
            # check interval t0,t1
            a = np.array([t0, t1])
            b = np.array([0, 1])
        else:
            # check interval t1, t0
            a = np.array([t1, t0])
            b = np.array([0, 1])
        if b[0] > a[1] or a[0] > b[0]:
            # print(a, b, 'False')
            return False, p, q, t, r, u, s
        else:
            # print(a, b, 'True')
            return True, t, r, u, s
    elif rs == 0 and qpr != 0:
        return False, p, q, t, r, u, s
    elif rs != 0 and 0 < t < 1 and 0 < u < 1:
        # meet at p+t*r, q+u*s
        return True, p, q, t, r, u, s
    else:
        # not parallel, but do not intersect
        return False, p, q, t, r, u, s


class Line:
     # TODO: Somehow render with openGL instead of using a png.
    def __init__(self, pos, height=4):
        # Load Pos Data
        self.x1 = pos[0]
        self.y1 = pos[1]
        self.x2 = pos[2]
        self.y2 = pos[3]

        # Generate Geometrical data of line
        self.x = (self.x1 + self.x2)/2
        self.y = (self.y1 + self.y2)/2
        self.width = round(np.sqrt((self.x1-self.x2)**2+(self.y1-self.y2)**2))
        self.height = height
        if self.x1 != self.x2:
            self.rotation = -np.rad2deg(np.arctan((self.y2-self.y1)/(self.x2-self.x1)))
        else:
            self.rotation = 90.0
        self.rotation_rad = np.deg2rad(self.rotation)
        self.image = None
        self.sprite = None

    def line(self):
        return np.array([self.x1, self.y1, self.x2, self.y2])

    def set_image(self, img, center=1):
        self.image = pyglet.resource.image(img)
        if center == 1:
            center_image(self.image)
        self.sprite = pyglet.sprite.Sprite(self.image, x=self.x, y=self.y)
        self.sprite.image = self.image
        self.sprite.rotation = self.rotation
        self.sprite.scale_x = (self.width + self.height) / self.image.width
        self.sprite.scale_y = self.height / self.image.height

    # update position based upon x,y,rotation
    def update_position(self, pos):
        # pos is either 3 long: x,y,theta, or 4 long, x1,y1,x2,y2
        if len(pos) == 3:
            self.x = pos[0]
            self.y = pos[1]
            self.rotation = pos[2]
            self.rotation_rad = np.deg2rad(self.rotation)

            self.sprite.x = self.x
            self.sprite.y = self.y
            self.sprite.rotation = self.rotation

            self.x1 = self.x + np.cos(self.rotation_rad)*-self.width/2
            self.x2 = self.x + np.cos(self.rotation_rad)*self.width/2
            self.y1 = self.y + np.sin(self.rotation_rad)*self.width/2
            self.y2 = self.y - np.sin(self.rotation_rad)*self.width/2

        if len(pos) == 4:
            self.x1 = pos[0]
            self.y1 = pos[1]
            self.x2 = pos[2]
            self.y2 = pos[3]
            self.x = (self.x1 + self.x2)/2
            self.y = (self.y1 + self.y2)/2
            self.width = round(np.sqrt((self.x1-self.x2)**2+(self.y1-self.y2)**2))
            if self.x1 != self.x2:
                self.rotation = -np.rad2deg(np.arctan((self.y2-self.y1)/(self.x2-self.x1)))
            else:
                self.rotation = 90.0
            self.rotation_rad = np.deg2rad(self.rotation)
            self.sprite.x = self.x
            self.sprite.y = self.y
            self.sprite.rotation = self.rotation
            self.sprite.scale_x = (self.width + self.height) / self.image.width
            self.sprite.scale_y = self.height / self.image.height

    def update_position_x1y1(self, pos):
        self.x1 = pos[0]
        self.y1 = pos[1]
        self.rotation = pos[2]
        self.rotation_rad = np.deg2rad(pos[2])

        self.x = self.x1 - np.cos(self.rotation_rad) * -self.width / 2
        self.x2 = self.x + np.cos(self.rotation_rad) * self.width / 2
        self.y = self.y1 - np.sin(self.rotation_rad) * self.width / 2
        self.y2 = self.y - np.sin(self.rotation_rad) * self.width / 2

        self.sprite.x = self.x
        self.sprite.y = self.y
        self.sprite.rotation = self.rotation


class Circle:
    def __init__(self, r, points, cent_pos, color, batch):
        self.r = r
        self.points = points
        vertices = []
        for i in range(points):
            angle = np.deg2rad(i/points * 360.0)
            x = r * np.cos(angle) + cent_pos[0]
            y = r * np.sin(angle) + cent_pos[1]
            vertices += [x, y]
        self.circle = pyglet.graphics.vertex_list(points, ('v2f', vertices))
        self.batch = batch

    def update_position(self, cent_pos):
        vertices = []
        for i in range(self.points):
            angle = np.deg2rad(i/self.points * 360.0)
            x = self.r * np.cos(angle) + cent_pos[0]
            y = self.r * np.sin(angle) + cent_pos[1]
            vertices += [x, y]
        glColor3f(1,0,0)
        self.circle = self.batch.add(self.points, GL_LINES, None, ('v2f', vertices))

