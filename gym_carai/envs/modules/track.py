import numpy as np
from gym_carai.envs.modules.util import Line


def read_track(filename):
    """"reads data file in shape of type, x1,y1,x2,y2"""
    read_data = np.genfromtxt(filename, delimiter=',')
    return read_data


class TrackBorder(Line):
    _counter = 0

    def __init__(self, pos):
        super().__init__(pos)
        self.solid = 1
        self.set_image('BlackBar.png')
        TrackBorder._counter += 1
        self.id = TrackBorder._counter


class Checkpoint(Line):
    _counter = 0

    def __init__(self, pos, score):
        super().__init__(pos)
        self.solid = 0
        self.set_image('GreenBar.png')
        Checkpoint._counter += 1
        self.id = Checkpoint._counter
        self.score = score


def generate_track(filename):
    track_objects = []
    checkpoints = []
    track_data = read_track(filename)
    for i in range(len(track_data)):
        if track_data[i, 0] == 0:
            track_objects.append(TrackBorder(track_data[i, 1:5]))
        elif track_data[i, 0] == 1:
            checkpoints.append(Checkpoint(track_data[i, 1:5], track_data[i, 5]))
        elif track_data[i, 0] == -1:
            car_position = (track_data[i, 1:4]) # x, y, rotation
    return track_objects, checkpoints, car_position
