from graphviz import Digraph


def create_graph(file_name, tree_index=0, value_list=[]):
    with open(file_name, "r") as f:
        tree = f.read().split("Booster")
        tree.pop(0)
        tree = tree[tree_index]
        tree = tree.split("\n")[1:]

        info = {}
        for line in tree:
            line = line.strip().split(",")
            if len(line) <= 1: continue
            node = int(line.pop(0))
            if node > 0:
                value = [float(_) for _ in line]
            elif node < 0:
                # parent, left, right, column, threshold
                value = [int(line[0]), int(line[1]), int(line[2]), int(line[3]), float(line[4])]
            info.update({node: value})

    graph = Digraph()

    def add(node, parent):
        if node < 0:
            label = "X[{}] {:.4f}".format(info[node][3], info[node][4])
            graph.node("NL-{}".format(node), label=label)
            if info[parent][1] == node:
                # left child
                graph.edge("NL-{}".format(parent), "NL-{}".format(node), label="<=")
            if info[parent][2] == node:
                # right child
                graph.edge("NL-{}".format(parent), "NL-{}".format(node), label=">")
            add(info[node][1], node)
            add(info[node][2], node)

        if node > 0:
            if len(value_list) == 0:
                label = "{:.4f}".format(info[node][0])
            else:
                label = ""
                for v in value_list:
                    label += "{:.4f}\n".format(info[node][v])
                label = label.strip()

            graph.node("L-{}".format(node), shape="record", label=label)
            if info[parent][1] == node:
                # left child
                graph.edge("NL-{}".format(parent), "L-{}".format(node), label="<=")
            if info[parent][2] == node:
                # right child
                graph.edge("NL-{}".format(parent), "L-{}".format(node), label=">")
                
    add(-1, -1)
    return graph

