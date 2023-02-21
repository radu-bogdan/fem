# from PyQt5 import QtWidgets
# app = QtWidgets.QApplication.instance()
# if app is not None:
#     import sip
#     app.quit()
#     sip.delete(app)
#
# import sys
# from PyQt5 import QtCore, QtWebEngineWidgets
# QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
# app = QtWidgets.qApp = QtWidgets.QApplication(sys.argv)

# from PyQt5 import QtCore, QtWidgets, QtWebEngineWidgets


import numpy as np
import plotly.graph_objs as go


fig = go.Figure()
fig.add_scatter(x=np.random.rand(100), y=np.random.rand(100), mode='markers',
                marker={'size': 30, 'color': np.random.rand(100), 'opacity': 0.6, 
                        'colorscale': 'Viridis'});