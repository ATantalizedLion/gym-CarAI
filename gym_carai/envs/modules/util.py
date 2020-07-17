import numpy as np
import pyglet
from gym_carai.envs.modules.render import Rect, Line


def center_image(image):
    """ Sets an image's anchor point to the center """
    image.anchor_x = image.width // 2
    image.anchor_y = image.height // 2


def vector_length(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)


def line_overlapping(line1, line2, get_dist=False, printT=False):
    """ Where each line(segment) is a list containing a start and end position
    e.g. line1 = [x1 y1 x2 y2]
         line2 = [x3 y3 x4 y4]"""
    aX, aY, bX, bY = line1
    cX, cY, dX, dY = line2
    denom = ((bX - aX) * (dY - cY)) - ((bY - aY) * (dX - cX))
    num1 = ((aY - cY) * (dX - cX)) - ((aX - cX) * (dY - cY))
    num2 = ((aY - cY) * (bX - aX)) - ((aX - cX) * (bY - aY))
    if get_dist:
        if denom == 0:
            if num1 == 0 and num2 == 0:
                return True, None, None
            else:
                return False, None, None
        r = num1/denom
        s = num2/denom
        if (0 <= r <= 1) and (0 <= s <= 1):
            return True, abs(r*np.sqrt((aX-bX)**2 + (aY-bY)**2)), [aX+r*(bX-aX), aY+r*(bY-aY)]
        else:
            return False, abs(r*np.sqrt((aX-bX)**2 + (aY-bY)**2)), [aX+r*(bX-aX), aY+r*(bY-aY)]
    else:
        if denom == 0:
            if num1 == 0 and num2 == 0:
                return True
            else:
                return False
        r = num1/denom
        s = num2/denom
        if (0 <= r <= 1) and (0 <= s <= 1):
            return True
        else:
            return False

class LineObject:
    def __init__(self, pos, height=4):
        # Load Pos Data
        self.x1, self.y1, self.x2, self.y2 = pos

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

        # Rendering object
        self.sprite = None

    def line(self):
        return np.array([self.x1, self.y1, self.x2, self.y2])

    def create_sprite(self, batch, color):
        self.sprite = Line(batch, self.x1, self.y1, self.x2, self.y2, color=color)

    # update position based upon x,y,rotation
    def update_position(self, pos):
        # pos is either 3 long: x,y,theta, or 4 long, x1,y1,x2,y2
        if len(pos) == 3:
            self.x, self.y, self.rotation = pos
            self.rotation_rad = np.deg2rad(self.rotation)

            if self.sprite is not None:
                self.sprite.update_position_rot(self.rotation, self.x, self.y)

            c = np.cos(self.rotation_rad)
            s = np.sin(self.rotation_rad)
            self.x1 = self.x + c*-self.width/2
            self.x2 = self.x + c*self.width/2
            self.y1 = self.y + s*self.width/2
            self.y2 = self.y - s*self.width/2

        if len(pos) == 4:
            self.x1, self.y1, self.x2, self.y2 = pos
            self.x = (self.x1 + self.x2)/2
            self.y = (self.y1 + self.y2)/2
            self.width = round(np.sqrt((self.x1-self.x2)**2+(self.y1-self.y2)**2))
            if self.x1 != self.x2:
                self.rotation = -np.rad2deg(np.arctan((self.y2-self.y1)/(self.x2-self.x1)))
            else:
                self.rotation = 90.0
            self.rotation_rad = np.deg2rad(self.rotation)

            if self.sprite is not None:
                self.sprite.update_position(self.x1, self.y1, self.x2, self.y2)

    def update_position_x1y1(self, pos):
        ''' Update position based on x1, y1 and rotation, exists for the sensors.'''
        self.x1, self.y1, self.rotation = pos
        self.rotation -= 90
        self.rotation_rad = np.deg2rad(self.rotation)
        c = np.cos(self.rotation_rad)
        s = np.sin(self.rotation_rad)
        self.x = self.x1 - c * -self.width / 2
        self.x2 = self.x + c * self.width / 2
        self.y = self.y1 - s * self.width / 2
        self.y2 = self.y - s * self.width / 2

        if self.sprite is not None:
            self.sprite.update_position(self.x1, self.y1, self.x2, self.y2)

