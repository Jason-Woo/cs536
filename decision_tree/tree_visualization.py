import matplotlib.pyplot as plt


def num_leaf(tree):
    """
        return number of leaves of given tree using recursive method
        ----------
        Parameters
        tree: tree_node object
            decision tree used for prediction
        ----------
        Return
        cnt: integer
            the number of leaves of given tree
    """
    if tree.key == -1:
        return 1
    else:
        cnt = 0
        if tree.left:
            cnt += num_leaf(tree.left)
        if tree.right:
            cnt += num_leaf(tree.right)
        return cnt


def depth_tree(decision_tree, depth):
    """
        return depth of given tree using recursive method
        ----------
        Parameters
        tree: tree_node object
            decision tree used for prediction
        depth: integer
            current depth of tree node
        ----------
        Return
        cnt: integer
            the depth of given tree
    """
    if decision_tree.key == -1:
        return depth
    else:
        d1, d2 = 0, 0
        if decision_tree.left:
            d1 = depth_tree(decision_tree.left, depth + 1)
        if decision_tree.right:
            d2 = depth_tree(decision_tree.right, depth + 1)
        return max(d1, d2)


def plot_node(label, node_pos, label_pos):
    """
        add decision tree node/leaf to the plot
        ----------
        Parameters
        label: integer
            the label of the tree node/leaf
        node_pos: tuple of size 2
            The position of the tree node/leaf
        label_pos: tuple of size 2
            The position of the label of tree node/leaf
    """
    createPlot.ax1.annotate(label, xy=label_pos, xycoords='axes fraction',
                            xytext=node_pos, textcoords='axes fraction',
                            va="center", ha="center", bbox=dict(boxstyle="round4", fc="0.8"), arrowprops=dict(arrowstyle="<-"))


def plot_edge_txt(txt, node_pos, label_pos):
    """
        add edge of decision tree to the plot
        ----------
        Parameters
        txt: integer
            the label of the edge
        node_pos: tuple of size 2
            The position of the tree node/leaf
        label_pos: tuple of size 2
            The position of the label of tree node/leaf
    """
    x_pos = (node_pos[0] - label_pos[0]) / 2.0 + label_pos[0] + 0.01
    y_pos = (node_pos[1] - label_pos[1]) / 2.0 + label_pos[1] + 0.01
    createPlot.ax1.text(x_pos, y_pos, txt, va="center", ha="center")


def plotTree(tree, pos, txt):
    """
        use recursive method to plot decision tree
        ----------
        Parameters
        tree: tree_node object
            decision tree used for prediction
        txt: integer
            the label of the edge/node
        pos: tuple of size 2
            The position of the label
    """
    leaf_num = num_leaf(tree)
    root = tree.key
    cntrPt = (plotTree.xOff + (1.0 + float(leaf_num)) / 2.0 / plotTree.tree_width, plotTree.yOff)
    plot_edge_txt(txt, cntrPt, pos)
    plot_node(root, cntrPt, pos)
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.tree_depth
    if tree.left:
        if tree.left.key == -1:
            # if it is a leaf
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.tree_width
            plot_node(tree.left.label, (plotTree.xOff, plotTree.yOff), cntrPt)
            plot_edge_txt('0', (plotTree.xOff, plotTree.yOff), cntrPt)
        else:
            # if it is a tree node
            plotTree(tree.left, cntrPt, '0')
    if tree.right:
        if tree.right.key == -1:
            # if it is a leaf
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.tree_width
            plot_node(tree.right.label, (plotTree.xOff, plotTree.yOff), cntrPt)
            plot_edge_txt('1', (plotTree.xOff, plotTree.yOff), cntrPt)
        else:
            # if it is a tree node
            plotTree(tree.right, cntrPt, '1')
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.tree_depth


def createPlot(decision_tree):
    """
        plot the decision tree
        ----------
        Parameters
        decision_tree: tree_node object
            decision tree used for prediction
    """
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # no ticks
    plotTree.tree_width = float(num_leaf(decision_tree))
    plotTree.tree_depth = float(depth_tree(decision_tree, 1))

    plotTree.xOff = -0.5 / plotTree.tree_width
    plotTree.yOff = 1
    plotTree(decision_tree, (0.5, 1), '')
    plt.show()
