class node:
    def __init__(self, value):
        self.value = value
        self.parent = self
        self.rank = 0

class union_find:
    def __init__(self):
        self.node_values = {}
    def make_set(self, value):
        if value in self.node_values:
            return self.node_values[value]
        else:
            new_node = node(value)
            self.node_values[value] = new_node
            return new_node
    def find(self, node_x):
        if node_x.parent != node_x:
            node_x.parent = self.find(node_x.parent)
        return node_x.parent
    def union(self, node_x, node_y):
        if node_x == node_y:
            return
        node_x_root = self.find(node_x)
        node_y_root = self.find(node_y)
        if node_x_root == node_y_root:
            return
        if node_x_root.rank < node_y_root.rank:
            node_x_root.parent = node_y_root

        elif node_x_root.rank > node_y_root.rank:
            node_y_root.parent = node_x_root

        else:
            node_y_root.parent = node_x_root
            node_x_root.rank += 1
