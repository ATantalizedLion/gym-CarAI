import pyglet
from gym_carai.envs.modules.render import Rect, Line, Circle
from gym_carai.envs.modules.track import TrackBorder, Checkpoint
import numpy as np

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
    # TODO: Checkpoints.. click creates two draggable points, one to the left and one to the right of the cursor,
    #  control score of selected with scrollwheel
    # TODO: Export
    def __init__(self, batch):
        self.batch = batch
        self.dragging = False
        self.node_groups = 0
        self.group_list = []
        self.create_group()
        self.current_group_index = 0
        self.current_group = self.group_list[self.current_group_index]
        self.grid_sizes = [1, 5, 10, 20, 25, 50, 75, 100, 200]
        self.grid_num = 4
        self.grid_size = self.grid_sizes[self.grid_num]
        self.CheckPointGroup = CheckPointGroup(batch)

    def create_group(self):
        self.group_list.append(NodeGroup(self.batch, self.node_groups))
        self.node_groups += 1

    def next_group(self):
        if self.current_group_index == len(self.group_list) - 1:
            self.create_group()
            self.current_group_index += 1
        else:
            self.current_group_index += 1
        if self.current_group_index-1 == -1:
            pass
        else:
            self.current_group.deactivate()

        self.current_group = self.group_list[self.current_group_index]
        self.current_group.activate()

    def prev_group(self):
        if self.current_group_index == 0:
            # self.current_group_index = self.node_groups
            self.current_group.deactivate()
            self.current_group_index = -1
            self.current_group = self.CheckPointGroup
        elif self.current_group_index == -1:
            self.current_group_index += 1
            self.current_group = self.group_list[self.current_group_index]
            self.current_group.activate()
        else:
            self.current_group_index -= 1
            self.current_group.deactivate()
            self.current_group = self.group_list[self.current_group_index]
            self.current_group.activate()

    def activate_group(self, group):
        self.current_group.deactivate()
        self.current_group = group
        self.current_group_index = self.group_list.index(self.current_group)
        self.current_group.activate()

    def delete_group(self):
        self.current_group.delete_all_nodes()

    def create_node(self, x, y, nodegroup=None):
        if nodegroup is None:
            if self.current_group is not None:
                nodegroup = self.current_group
            else:
                return
        node = nodegroup.create_node(x, y)

    def increase_grid(self):
        if self.grid_num < len(self.grid_sizes)-1:
            self.grid_num += 1
        self.grid_size = self.grid_sizes[self.grid_num]

    def decrease_grid(self):
        if self.grid_num > 0:
            self.grid_num -= 1
        self.grid_size = self.grid_sizes[self.grid_num]

    def snap_back(self):
        for node in self.current_group.nodes:
            node.x = node.x_on_grid
            node.y = node.y_on_grid

    def export(self):
        raise NotImplementedError


class NodeGroup:
    def __init__(self, batch, id):
        self.id = id
        self.batch = batch
        self.latest_node = 0
        self.nodes = []
        self.nodes_under_mouse = []
        self.tracks = []
        self.current_node = None

    def create_node(self, x, y):
        node = Node(x, y, self.batch, self.latest_node, NodeGroup=self)
        if self.current_node is None:
            self.nodes.append(node)
        else:
            self.nodes.insert(self.nodes.index(self.current_node), node)
        if len(self.nodes) > 2:
            self.gen_track()
        self.latest_node += 1
        return node

    def delete_node(self, node_id):
        ind = 0
        if self.current_node is not None:
            if self.current_node.id == self.current_node.id:
                self.current_node.deactivate()
        for node in self.nodes:
            if node.id == node_id:
                del self.nodes[ind]
            ind += 1
        self.gen_track()

    def delete_all_nodes(self):
        self.deactivate_node()
        for i in range(len(self.nodes))[::-1]:
            self.nodes[i].delete()

    def activate_node(self, node):
        if self.current_node is not None and node.id == self.current_node.id:
            node.deactivate()
        else:
            self.deactivate_node()
            self.current_node = node
            node.activate()

    def deactivate_node(self):
        if self.current_node is not None:
            self.current_node.deactivate()
        self.current_node = None

    def activate(self):
        for node in self.nodes:
            node.sprite.update_color((0, 33, 71))

    def deactivate(self):
        self.deactivate_node()
        for node in self.nodes:
            node.sprite.update_color((71, 71, 71))

    def del_track(self):
        for track in self.tracks:
            track.sprite.delete()
        self.tracks = []

    def gen_track(self):
        self.del_track()
        if len(self.nodes) > 1:
            i_prev = -1
            for i in range(len(self.nodes)):
                x1 = self.nodes[i_prev].x_on_grid
                y1 = self.nodes[i_prev].y_on_grid
                x2 = self.nodes[i].x_on_grid
                y2 = self.nodes[i].y_on_grid
                self.tracks.append(TrackBorder([x1, y1, x2, y2], self.batch))
                i_prev = i


class Node:
    def __init__(self, x, y, batch, id, NodeGroup):
        self.id = id
        self.x = x
        self.y = y
        self.NodeGroup = NodeGroup
        self.size = 25
        self.detection_distance = self.size / 2
        self.sprite = Rect(x, y, 0, self.size, self.size, batch, color=(0, 33, 71))
        self.x_on_grid = None
        self.y_on_grid = None
        self.update_pos(self.x, self.y)
        self.active_color = (33, 71, 33)
        self.inactive_color = (0, 33, 71)

    def check_hover(self, x, y):
        if abs(x - self.x_on_grid) < self.detection_distance and abs(y - self.y_on_grid) < self.detection_distance:
            return True
        else:
            return False

    def update_pos(self, x, y):
        self.x = x
        self.y = y
        self.x_on_grid = NodeManager.grid_size * round(x / NodeManager.grid_size)
        self.y_on_grid = NodeManager.grid_size * round(y / NodeManager.grid_size)
        self.sprite.update_pos(self.x_on_grid, self.y_on_grid)

    def activate(self):
        self.sprite.update_color(self.active_color)

    def deactivate(self):
        self.sprite.update_color(self.inactive_color)

    def delete(self):
        self.NodeGroup.delete_node(self.id)
        self.sprite.delete()
        del self


class CheckPointGroup:
    def __init__(self, batch):
        self.id = id
        self.batch = batch
        self.latest_node = 0
        self.nodes = []
        self.nodes_under_mouse = []
        self.tracks = []
        self.current_node = None
        self.pairs = []  # contains objects linking two checkpoint nodes.

    def create_node(self, x, y):
        node = CheckPointPair(x, y, self.batch, self.latest_node, self)
        self.pairs.append(node)
        self.latest_node += 1

    def del_track(self):
        for track in self.tracks:
            track.sprite.delete()
        self.tracks = []

    def gen_track(self):
        self.del_track()
        for i in range(len(self.pairs)):
            x1 = self.pairs[i].node1.x_on_grid
            y1 = self.pairs[i].node1.y_on_grid
            x2 = self.pairs[i].node2.x_on_grid
            y2 = self.pairs[i].node2.y_on_grid
            self.tracks.append(Checkpoint([x1, y1, x2, y2], self.batch))

class CheckPointPair:
    def __init__(self, x, y, batch, id, CheckPointGroup):
        self.default_dist = 100
        self.CheckPointGroup = CheckPointGroup
        self.id = id
        self.node1 = CheckPointNode(x-self.default_dist, y, batch, 1, CheckPointPair)
        self.node2 = CheckPointNode(x+self.default_dist, y, batch, 2, CheckPointPair)
        self.nodes = [self.node1, self.node2]


class CheckPointNode():
    def __init__(self, x, y, batch, id, CheckPointPair):
        self.id = id
        self.x = x
        self.y = y
        self.CheckPointPair = CheckPointPair
        self.size = 25
        self.active_color = (255, 94, 19)
        self.inactive_color = (0, 33, 71)
        self.detection_distance = self.size / 2
        self.sprite = Rect(x, y, 0, self.size, self.size, batch, color=self.active_color)
        self.x_on_grid = None
        self.y_on_grid = None
        self.update_pos(self.x, self.y)

    def check_hover(self, x, y):
        if abs(x - self.x_on_grid) < self.detection_distance and abs(y - self.y_on_grid) < self.detection_distance:
            return True
        else:
            return False

    def update_pos(self, x, y):
        self.x = x
        self.y = y
        self.x_on_grid = NodeManager.grid_size * round(x / NodeManager.grid_size)
        self.y_on_grid = NodeManager.grid_size * round(y / NodeManager.grid_size)
        self.sprite.update_pos(self.x_on_grid, self.y_on_grid)

    def activate(self):
        self.sprite.update_color(self.active_color)

    def deactivate(self):
        self.sprite.update_color(self.inactive_color)

    def delete(self):
        self.NodeGroup.delete_node(self.id)
        self.sprite.delete()
        del self


@window.event
def on_draw():
    window.clear()
    pyglet.gl.glEnable(pyglet.gl.GL_BLEND)
    pyglet.gl.glEnable(pyglet.gl.GL_LINE_SMOOTH)
    pyglet.gl.glHint(pyglet.gl.GL_LINE_SMOOTH_HINT, pyglet.gl.GL_DONT_CARE)
    main_batch.draw()


@window.event
def on_mouse_motion(x, y, dx, dy):
    if NodeManager.current_group is not None:
        NodeManager.current_group.nodes_under_mouse = []
        for node in NodeManager.current_group.nodes:
            if node.check_hover(x, y):
                NodeManager.current_group.nodes_under_mouse.append(node)


@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    for node in NodeManager.current_group.nodes_under_mouse:
        node.update_pos(node.x + dx, node.y + dy)
        NodeManager.dragging = True


@window.event
def on_mouse_press(x, y, button, modifiers):
    if button == pyglet.window.mouse.LEFT:
        activated = 0
        for node in NodeManager.current_group.nodes_under_mouse:
            node.NodeGroup.activate_node(node)
            activated = 1
        if activated == 0:
            for group in NodeManager.group_list:
                for node in group.nodes:
                    if node.check_hover(x, y):
                        NodeManager.activate_group(node.NodeGroup)
                        NodeManager.current_group.activate_node(node)

    elif button == pyglet.window.mouse.RIGHT:
        # create node
        NodeManager.create_node(x, y)
    elif button == pyglet.window.mouse.MIDDLE:
        middleclick = 1


@window.event
def on_mouse_release(x, y, button, modifiers):
    if button == pyglet.window.mouse.LEFT:
        NodeManager.current_group.gen_track()
        NodeManager.dragging = False
        NodeManager.snap_back()


@window.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.ESCAPE:
        NodeManager.export()

    if symbol == pyglet.window.key.DELETE and not modifiers & pyglet.window.key.MOD_CTRL:
        for node in NodeManager.current_group.nodes_under_mouse:
            node.delete()

    if symbol == pyglet.window.key.DELETE and modifiers & pyglet.window.key.MOD_CTRL:
        NodeManager.delete_group()

    if symbol == pyglet.window.key.RIGHT:
        NodeManager.next_group()
    elif symbol == pyglet.window.key.LEFT:
        NodeManager.prev_group()

    # elif symbol == pyglet.window.key.Q:
    #     if NodeManager.snap_to_grid:
    #         NodeManager.snap_to_grid = False
    #     else:
    #         NodeManager.snap_to_grid = True


@window.event
def on_mouse_scroll(x, y, scroll_x, scroll_y):
    if scroll_y > 0:
        NodeManager.increase_grid()
    elif scroll_y < 0:
        NodeManager.decrease_grid()


NodeManager = NodeManager(main_batch)

def update(dt):
    current_grid_size_label.text = "Current grid size: " + str(NodeManager.grid_size)
    if NodeManager.current_group_index == -1:
        current_group_label.text = "Current Group: Checkpoints"
    else:
        current_group_label.text = "Current Group: " + str(NodeManager.current_group_index)
    if NodeManager.dragging:
        NodeManager.current_group.gen_track()


current_group_label = pyglet.text.Label(text="Current Group: " + str(NodeManager.current_group.id),
                                        font_name='Times New Roman', font_size=36,
                                        x=0.5 * 36, y=window_v_size - 1.1 * 36,
                                        anchor_x='left', anchor_y='center',
                                        color=(100, 0, 0, 255), batch=main_batch)
current_grid_size_label = pyglet.text.Label(text="Current grid size: " + str(NodeManager.grid_size),
                                            font_name='Times New Roman', font_size=36,
                                            x=0.5 * 36, y=window_v_size - 2.6 * 36,
                                            anchor_x='left', anchor_y='center',
                                            color=(100, 0, 0, 255), batch=main_batch)

if __name__ == '__main__':
    pyglet.clock.schedule_interval(update, 1 / 60.0)
    pyglet.app.run()
