import pyglet
import numpy as np

from gym_carai.envs.modules.track import generate_track
from gym_carai.envs.modules.util import *
from gym_carai.envs.modules.car import Car

pyglet.resource.path = ['resources']
pyglet.resource.reindex()


windowHSize = 1920
windowVSize = 1080
score_label_font_size = 36
track_name = 'simpleSquareTrack NoC'

FPS = 60
dt = 1 / FPS
window = pyglet.window.Window(windowHSize, windowVSize)
main_batch = pyglet.graphics.Batch()
debug_batch = pyglet.graphics.Batch()
track_batch = pyglet.graphics.Batch()

# Labels for score and level, scaled based upon score label font size
score = 0
score_label = pyglet.text.Label(text="Current Score: " + str(score),
                                font_name='Times New Roman',
                                font_size=score_label_font_size,
                                x=0.5 * score_label_font_size, y=window.height - 1.1 * score_label_font_size,
                                anchor_x='left', anchor_y='center',
                                batch=main_batch,
                                color=(100, 0, 0, 255))
track_label = pyglet.text.Label(text="Current Track:" + track_name,
                                font_name='Times New Roman',
                                font_size=score_label_font_size * 0.5,
                                x=0.5 * score_label_font_size, y=window.height - 2.2 * score_label.font_size,
                                anchor_x='left', anchor_y='center',
                                batch=main_batch,
                                color=(100, 0, 100, 255))
you_died = pyglet.text.Label(text="YOU DIED",
                             font_name='Times New Roman',
                             font_size=100,
                             x=0.5 * windowHSize, y=windowVSize * 0.5,
                             anchor_x='left', anchor_y='center',
                             color=(255, 0, 0, 255))

# define functions
walls, checkpoints, car_position = generate_track('resources/' + track_name + '.csv')
current_checkpoint = 1

for item in walls:
    item.sprite.batch = track_batch
for item in checkpoints:
    item.sprite.batch = track_batch

car_obj = Car(car_position, debug_batch=debug_batch)
car_obj.sprite.batch = main_batch
car_bumpers = [car_obj.Bumper, car_obj.SideL, car_obj.SideR, car_obj.Rear]
for item in car_bumpers:
    item.sprite.batch = main_batch

envObjects = [car_obj]

sensors = [car_obj.FrontDistanceSensor, car_obj.RearDistanceSensor, car_obj.LeftDistanceSensor, car_obj.RightDistanceSensor]

envObjects = [car_obj]

# redefine draw event
pyglet.gl.glClearColor(1, 1, 1, 1)  # white background

def update(dt, action):
    global current_checkpoint
    global score
    """"observation (object): agent's observation of the current environment
        reward (float) : amount of reward returned after previous action
        done (bool): whether the episode has ended, in which case further step() calls will return undefined results
        info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)"""
    for obj in envObjects:
        obj.update(dt, action)

    for obj in walls:
        for car_bumper in car_bumpers:
            tf, _, _, _, _, _, _ = line_overlapping(car_bumper.line(), obj.line())
            if tf:
                done = 1

    for obj in checkpoints:
        for car_bumper in car_bumpers:
            tf, _, _, _, _, _, _ = line_overlapping(car_bumper.line(), obj.line())
            if tf:
                if obj.id == current_checkpoint:
                    current_checkpoint += 1
                    if current_checkpoint > len(checkpoints):
                        current_checkpoint -= len(checkpoints)
                    score += obj.score
                    score_label.text = "Current Score: " + str(score)

    for sensor in sensors:
        min_distance = 900000
        col_loc = []
        print(sensor.line())
        for obj in walls:
            tf, p, q, t, r, u, s = line_overlapping(sensor.line(), obj.line())
            if tf:
                dist = vector_length(p, p + t*r)
                if dist < min_distance:
                    min_distance = dist
                    col_loc = p+t*r
        if min_distance < 900000:
            sensor.collision_marker.x = col_loc[0]
            sensor.collision_marker.y = col_loc[1]
        else:
            # no collision found, draw off-screen
            sensor.collision_marker.update_position([-50, -50])


@window.event
def on_draw():
    window.clear()
    track_batch.draw()
    main_batch.draw()
    debug_batch.draw()
    if k[key.LEFT] == 1:
        if k[key.RIGHT] == 1:
            action[0] = 0
        else:
            action[0] = -1
    elif k[key.RIGHT] == 1:
        action[0] = 1
    else:
        action[0] = 0
    if k[key.UP] == 1:
        action[1] = 1
    else:
        action[1] = 0
    if k[key.DOWN] == 1:
        action[2] = 1
    else:
        action[2] = 0


if __name__ == '__main__':
    from pyglet.window import key
    k = key.KeyStateHandler()
    window.push_handlers(k)
    action = np.array([0.0, 0.0, 0.0])
    pyglet.clock.schedule_interval(update, dt, action)
    pyglet.app.run()
