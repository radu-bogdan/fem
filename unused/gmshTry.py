# import plotly.graph_objects as go

# import plotly.io as pio
# pio.renderers.default = 'browser'

# import pandas as pd

# # Read data from a csv
# z_data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv')

# fig = go.Figure()
# fig.add_trace(go.Surface(z=z_data.values,hidesurface=True))
# fig.update_traces(contours_z=dict(show=True, usecolormap=True,highlightcolor="limegreen", project_z=True))

# # fig.update_layout(title='Mt Bruno Elevation', autosize=True)

# fig.layout.scene.camera.projection.type = "orthographic"

# fig.show()




# import plotly.graph_objects as go
# import numpy as np
# from scipy.interpolate import griddata

# import plotly.io as pio
# pio.renderers.default = 'browser'



# np.random.seed(4231)


# #generate irregular data
# #z-values at the irregular data
# x= -1+2*np.random.rand(100)
# y=  np.random.rand(100)
# z = np.sin(np.pi*2*(x*x+y*y));

# # Define a regular grid over the data
# #e valuate the z-values at the regular grid through cubic interpolation
# xr = np.linspace(x.min(), x.max(), 200); yr = np.linspace(y.min(), y.max(), 100)
# xr, yr = np.meshgrid(xr, yr)
# Z = griddata((x, y), z, (xr, yr) , method='cubic')


# fig=go.Figure(go.Contour(x=xr[0], y=yr[:, 0], z=Z, 
#                          colorscale='curl', 
#                          contours=dict(start=np.nanmin(Z), 
#                          end=np.nanmax(Z), size=0.2)))
# fig.update_layout(title_text='Contour plot from irregular data',
#                   title_x=0.5)

# fig.show()


import plotly.graph_objects as go
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/vortex.csv")

fig = go.Figure(data = go.Cone(
    x=df['x'],
    y=df['y'],
    z=df['z'],
    u=df['u'],
    v=df['v'],
    w=df['w'],
    colorscale='Jet',
    sizemode="absolute",
    sizeref=40))

fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=0.8),
                             camera_eye=dict(x=1.2, y=1.2, z=0.6)))

fig.show()