import matplotlib.pyplot as plt

def getNumLeafs(myTree):
    if myTree.key == -1:
        return 1
    else:
        cnt = 0
        if myTree.left:
            cnt += getNumLeafs(myTree.left)
        if myTree.right:
            cnt += getNumLeafs(myTree.right)
        return cnt

def getTreeDepth(myTree, depth):
    if myTree.key == -1:
        return depth
    else:
        d1, d2 = 0, 0
        if myTree.left:
            d1 = getTreeDepth(myTree.left, depth + 1)
        if myTree.right:
            d2 = getTreeDepth(myTree.right, depth + 1)
        return max(d1, d2)

def plotNode(nodeTxt, centerPt, parentPt):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=dict(boxstyle="round4", fc="0.8"), arrowprops=dict(arrowstyle="<-"))


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0] + 0.01
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1] + 0.01
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center")


def plotTree(myTree, parentPt, nodeTxt):  # if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  # this determines the x width of this tree
    firstStr = myTree.key
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt)
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    if myTree.left:
        if myTree.left.key == -1:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(myTree.left.label, (plotTree.xOff, plotTree.yOff), cntrPt)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, '0')
        else:
            plotTree(myTree.left, cntrPt, '0')  # recursion
    if myTree.right:
        if myTree.right.key == -1:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(myTree.right.label, (plotTree.xOff, plotTree.yOff), cntrPt)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, '1')
        else:
            plotTree(myTree.right, cntrPt, '1')  # recursion
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


# if you do get a dictonary you know it's a tree, and the first element will be another dict

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # no ticks
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree, 1))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1
    plotTree(inTree, (0.5, 1), '')
    plt.show()
