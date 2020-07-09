import pyglet
from gym_carai.envs.modules.render import Rect, Line

pyglet.options['debug_gl'] = False  # minor performance increase
window_h_size = 1920
window_v_size = 1080
debug = 0  # renders all bumpers, sensors and collision markers.
window = pyglet.window.Window(width=1920, height=1080)
pyglet.resource.path = ['gym_carai/envs/resources']
pyglet.resource.reindex()


main_batch = pyglet.graphics.Batch()

pyglet.gl.glClearColor(1, 1, 1, 1)


class NodeManager():
    # TODO: Arrow key is change current group (if reach end, add new group)
    # TODO: Current group green, Nodes Green  (Node collision/click area translucent yellow for debug)
    # TODO: Selected node is blue
    # TODO: Non-current groups gray
    # TODO: Think of something for the checkpoints (l/r arrows to switch to checkpoint mode?)
    # TODO: Can drag and drop all nodes of current group, connect in order
    # TODO: Right click to add a node, Adding a node adds it next to the currently selected not
    def __init__(self,x,y,batch):


        raise NotImplementedError

class NodeGroup:
    def __init__(self, x, y, batch):
        stuff = 1

class Node:
    def __init__(self, x, y, batch, nodegroup=None):
        self.x = x
        self.y = y
        self.node = Rect(x, y, 0, 50, 50, batch, color=(0, 33, 71))
        self.detection_distance = 50

    def checkHover(self, x, y):
        if abs(x-self.x) < self.detection_distance and abs(y-self.y) < self.detection_distance:
            return True
        else:
            return False

    def update_pos(self, x, y):
        self.x = x
        self.y = y
        self.node.update_pos(self.x, self.y)

@window.event
def on_mouse_motion(x, y, dx, dy):
    global nodesundermouse
    nodesundermouse = []
    for node in nodes:
        if node.checkHover(x, y):
            nodesundermouse.append(node)


@window.event
def on_draw():
    window.clear()
    main_batch.draw()

@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    global nodesundermouse
    for node in nodesundermouse:
        node.update_pos(node.x + dx, node.y + dy)

@window.event
def on_mouse_press(x, y, button, modifiers):
    if button == pyglet.window.mouse.LEFT:
        leftclick = 1
    elif button == pyglet.window.mouse.RIGHT:
        rightclick = 1
    elif button == pyglet.window.mouse.MIDDLE:
        middleclick = 1

def update(dt):
    do = 1

nodes = []
nodes.append(Node(500,500,main_batch))
if __name__ == '__main__':
    pyglet.clock.schedule_interval(update, 1/60.0)
    pyglet.app.run()