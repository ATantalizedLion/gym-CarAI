import numpy as np
from gym_carai.envs.modules.util import LineObject


def read_track(filename):
    """"reads data file in shape of type, x1,y1,x2,y2"""
    read_data = np.genfromtxt(filename, delimiter=',')
    return read_data


class TrackBorder(LineObject):
    _counter = 0

    def __init__(self, pos, batch):
        super().__init__(pos)
        self.solid = 1
        self.create_sprite(batch, color=(0, 0, 0))
        TrackBorder._counter += 1
        self.id = TrackBorder._counter


class TrackCentre(LineObject):
    _counter = 0
    def __init__(self, pos, batch):
        super().__init__(pos)
        self.solid = 1
        self.create_sprite(batch, color=(0, 100, 100))
        TrackCentre._counter += 1
        self.id = TrackCentre._counter


class Checkpoint(LineObject):
    _counter = 0

    def __init__(self, pos, batch):
        super().__init__(pos)
        self.solid = 0
        self.create_sprite(batch, color=(0, 255, 0))
        Checkpoint._counter += 1
        self.id = Checkpoint._counter


def generate_track(filename, batch):
    track_objects = []
    checkpoints = []
    track_data = read_track(filename)
    for i in range(len(track_data)):
        if track_data[i, 0] == 0:
            track_objects.append(TrackBorder(track_data[i, 1:5], batch))
        elif track_data[i, 0] == 1:
            checkpoints.append(Checkpoint(track_data[i, 1:5], batch))
        elif track_data[i, 0] == -1:
            car_position = (track_data[i, 1:4])  # x, y, rotation
    return track_objects, checkpoints, car_position
