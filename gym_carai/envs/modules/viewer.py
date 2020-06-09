import pyglet
import numpy as np

class Viewer():
    # TODO: Optimize rendering geometry (draw open GL shapes?)
    def __init__(self, width, height, openWin=True):
        self.width = width
        self.height = height
        self.window = pyglet.window.Window(width=self.width, height=self.height)
        self.is_open = open
        self.toDraw = []
        self.score_label = None
        self.track_label = None
        self.time_label = None
        self.episode_label = None
        self.Terminate = None
        self.window.on_close = self.window_closed

    def render(self, return_rgb_array=False):
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.window.clear()
        self.window.dispatch_events()
        # self.transform.enable()
        for obj in self.toDraw:
            obj.draw()

        # write rgb array as in classic control framework
        arr = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
        elif self.is_open:
            self.window.flip()
        return arr if return_rgb_array else self.is_open

    def window_closed(self):
        Terminate = 1
        self.Terminate = 1
        self.close()

    def close(self):
        if self.is_open:
            self.window.close()
            self.is_open = False

    def labels(self, main_batch, score_label_font_size, score, track_name, time, episode):
        if score is not None:
            self.score_label = pyglet.text.Label(text="Current Score: " + str(score),
                                                 font_name='Times New Roman',
                                                 font_size=score_label_font_size,
                                                 x=0.5 * score_label_font_size,
                                                 y=self.height - 1.1 * score_label_font_size,
                                                 anchor_x='left', anchor_y='center',
                                                 color=(100, 0, 0, 255),
                                                 batch=main_batch)
        if track_name is not None:
            self.track_label = pyglet.text.Label(text="Current Track:" + track_name,
                                                 font_name='Times New Roman',
                                                 font_size=score_label_font_size * 0.5,
                                                 x=0.5 * score_label_font_size,
                                                 y=self.height - 2.2 * self.score_label.font_size,
                                                 anchor_x='left', anchor_y='center',
                                                 color=(100, 0, 100, 255),
                                                 batch=main_batch)

        if time is not None:
            self.time_label = pyglet.text.Label(text="Current Episode Time:" + str(time),
                                                font_name='Times New Roman',
                                                font_size=score_label_font_size * 0.5,
                                                x=0.5 * score_label_font_size,
                                                y=self.height - 2.7 * self.score_label.font_size,
                                                anchor_x='left', anchor_y='center',
                                                color=(100, 0, 100, 255),
                                                batch=main_batch)

        if episode is not None:
            self.episode_label = pyglet.text.Label(text="Current episode:" + str(episode),
                                                   font_name='Times New Roman',
                                                   font_size=score_label_font_size * 0.5,
                                                   x=0.5 * score_label_font_size,
                                                   y=self.height - 3.2 * self.score_label.font_size,
                                                   anchor_x='left', anchor_y='center',
                                                   color=(100, 0, 100, 255),
                                                   batch=main_batch)

        return self.score_label, self.track_label, self.time_label, self.episode_label


