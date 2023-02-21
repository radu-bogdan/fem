from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
from plotly.graph_objs import *
init_notebook_mode()

trace0 = Scatter(
    x=[1, 2, 3, 4],
    y=[10, 11, 12, 13],
    mode='markers',
    marker=dict(
        size=[40, 60, 80, 100],
    )
)
data = [trace0]
layout = Layout(
    showlegend=False,
    height=600,
    width=600,
)

fig = dict( data=data, layout=layout )

plot(fig)