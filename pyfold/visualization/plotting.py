import plotly.graph_objects as go
from plotly.io import write_image
import numpy as np
from numbers import Number

color_dict = {
    'mountain':'red',
    'valley':'blue',
    'reference':'lightgrey',
    'crease':'black'
}

def plot_pattern(pattern, width=800, height=800, path=None, linewidth=1.):
       # Initialize an empty figure
    fig = go.Figure()

    # Create a trace for the middle points of the lines with invisible markers
    middle_node_trace = go.Scatter(
        x=[],  # Initialize as an empty list
        y=[],  # Initialize as an empty list
        text=[],  # Initialize as an empty list for hover text
        mode='markers',
        hoverinfo='text',
        showlegend=False,
        marker=dict(
            opacity=0  # Make markers invisible
        )
    )

    # Loop through each line in your pattern
    for line in pattern.lines:
        x_values = [line.start.x, line.end.x]
        y_values = [line.start.y, line.end.y]
        line_name = line.name  # Assuming each line has a name attribute

        if isinstance(linewidth, Number):
            line_width = linewidth
        else:
            if hasattr(line, linewidth):
                line_width=getattr(line, linewidth)
            else:
                line_width=0.   

        # Append line trace to figure
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='lines',
            line=dict(color=color_dict[line.type], width=line_width),
            showlegend=False,
            hoverinfo='none'  # No hover info for lines themselves
        ))

        # Calculate and append the midpoint for hover text
        midpoint_x = (line.start.x + line.end.x) / 2
        midpoint_y = (line.start.y + line.end.y) / 2
        middle_node_trace['x'] += (midpoint_x,)  # Add midpoint x as a tuple element
        middle_node_trace['y'] += (midpoint_y,)  # Add midpoint y as a tuple element
        middle_node_trace['text'] += (line_name+'\ndist={}'.format(line.length),)  # Add line name as hover text

    # Add the middle node trace for line hover information
    fig.add_trace(middle_node_trace)

    for vertex in pattern.vertices:
        # Using vertex.name as hover text, assuming each vertex has a name attribute
        fig.add_trace(go.Scatter(x=[vertex.x], y=[vertex.y],
                                 mode='markers', hoverinfo='text', text=[vertex.name],
                                 marker=dict(color='black', size=1),  # Set all vertices to one color, e.g., blue
                                 showlegend=False))  # No legend for vertices


    # Update the layout to adjust the figure size and keep the axes to scale
    fig.update_layout(
        clickmode='event+select',
        width=width,
        height=height,
        autosize=False,
        paper_bgcolor='white',  # Sets the background color of the figure
        plot_bgcolor='white',   # Sets the background color of the plotting area
        xaxis=dict(
            scaleanchor='y',  # This constrains the aspect ratio to match the y-axis
            scaleratio=1,  # This ensures that one unit in x is equal to one unit in y
        ),
        yaxis=dict(
            constrain='domain'  # This option, combined with scaleanchor, helps keep the aspect ratio
        )
    )

    # Hide axis lines
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    # Save the figure to an SVG file
    if path is not None:
        write_image(fig, path, format='svg')


    return fig