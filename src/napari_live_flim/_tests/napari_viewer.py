# Here's a script to play around with napari to help with development
import napari
from qtpy import QtWidgets, QtCore

if __name__ == "__main__":
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
    viewer = napari.Viewer()
    napari.run()
