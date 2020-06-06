import pyglet
import numpy as np

from v1.game.track import generate_track
from v1.game.util import *
from v1.game.car import Car

pyglet.resource.path = ['../resources']
pyglet.resource.reindex()

windowHSize = 1920
windowVSize = 1080
score_label_font_size = 36
track_name = 'simpleSquareTrack'

FPS = 60
dt = 1 / FPS
window = pyglet.window.Window(windowHSize, windowVSize)
main_batch = pyglet.graphics.Batch()
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
walls, checkpoints, car_position = generate_track('../resources/' + track_name + '.csv')
current_checkpoint = 1

for item in walls:
    item.sprite.batch = track_batch
for item in checkpoints:
    item.sprite.batch = track_batch

carObj = Car(car_position)
carObj.sprite.batch = main_batch
car_bumpers = [carObj.Bumper, carObj.Side1, carObj.Side2, carObj.Rear]


envObjects = [carObj]

# redefine draw event
pyglet.gl.glClearColor(1, 1, 1, 1)  # white background

def update(dt):
    global current_checkpoint
    global score
    for obj in envObjects:
        obj.update(dt)
    for obj in walls:
        tf, dist = line_overlapping(obj.line(), carObj.Bumper.line())
    for obj in checkpoints:
        for car_bumper in car_bumpers:
            tf, dist = line_overlapping(obj.line(), car_bumper.line())
            if tf:
                if obj.id == current_checkpoint:
                    current_checkpoint += 1
                    if current_checkpoint > len(checkpoints):
                        current_checkpoint -= len(checkpoints)
                    score += obj.score
                    score_label.text = "Current Score: " + str(score)


@window.event
def on_draw():
    window.clear()
    track_batch.draw()
    main_batch.draw()


if __name__ == '__main__':
    from pyglet.window import key

    inputs = np.array([0.0, 0.0, 0.0])
    pyglet.clock.schedule_interval(update, dt)
    window.push_handlers(carObj)
    window.push_handlers(carObj.key_handler)
    pyglet.app.run()
