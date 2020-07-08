import pyglet

pyglet.options['debug_gl'] = False  # minor performance increase
window_h_size = 1920
window_v_size = 1080
debug = 0  # renders all bumpers, sensors and collision markers.
window = pyglet.window.Window(width=1920, height=1080)
pyglet.resource.path = ['gym_carai/envs/resources']
pyglet.resource.reindex()


batch = pyglet.graphics.Batch()
nodes = [
    pyglet.sprite.Sprite(pyglet.resource.image('BlueBar.png'), x=200, y=100, batch=batch),
    pyglet.sprite.Sprite(pyglet.resource.image('BlueBar.png'), x=400, y=300, batch=batch),
]

pyglet.gl.glClearColor(1, 1, 1, 1)

class NodeManager():
    # TODO: Arrow key is change current group (if reach end, add new group)
    # TODO: Current group green, Nodes Green  (Node collision/click area translucent yellow for debug)
    # TODO: Selected node is blue
    # TODO: Non-current groups gray
    # TODO: Think of something for the checkpoints (l/r arrows to switch to checkpoint mode?)
    # TODO: Can drag and drop all nodes of current group, connect in order
    # TODO: Right click to add a node, Adding a node adds it next to the currently selected not


@window.event
def on_draw():
    window.clear()
    batch.draw()

@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    for i in items:
        i.x += dx
        i.y += dy

pyglet.app.run()