import os
import math

import torch

from collections import defaultdict

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import interpolate

def plot_faithfulness(
        outs,
        save_path,
    ):
    """
    plot faithfulness results

    TODO : Plot edges w.r.t. nodes, and other plots if needed
    """
    # If outs contains scalar values, plot lines.
    # If outs contains list of scalar values, plot points. #TODO : can I do some sort of average line plot?

    is_list = False
    for faith in ['faithfulness', 'completeness']:
        thresholds = []
        n_nodes = []
        n_edges = []
        avg_deg = []
        density = []
        # modularity = []
        # z_score = []
        faithfulness = {}
        for t in outs:
            if t == 'complete':
                continue
            if t == 'empty':
                continue
            thresholds.append(t)
            if isinstance(outs[t]['n_nodes'], list):
                is_list = True
            n_nodes.append(outs[t]['n_nodes'])
            n_edges.append(outs[t]['n_edges'])
            avg_deg.append(outs[t]['avg_deg'])
            density.append(outs[t]['density'])
            # modularity.append(outs[t]['modularity'])
            # z_score.append(outs[t]['z_score'])
            for i, fn_name in enumerate(outs[t][faith]):
                if i not in faithfulness:
                    faithfulness[i] = []
                faithfulness[i].append(outs[t][faith][fn_name])
            
        if is_list:
            # n_nodes and all are lists of lists : flatten them
            
            def gaussian_kernel(data, window_size, sigma=5):
                kernel = [math.exp(-0.5 * ((x - window_size // 2) / sigma) ** 2) for x in range(window_size)]
                kernel = [x / sum(kernel) for x in kernel]
                return [sum(data[i + j] * kernel[j] for j in range(window_size)) for i in range(len(data) - window_size + 1)]

            window_size = 45

            n_nodes = [item for sublist in n_nodes for item in sublist]
            perm = sorted(range(len(n_nodes)), key=lambda k: n_nodes[k])

            n_nodes = [n_nodes[i] for i in perm]
            n_nodes = gaussian_kernel(n_nodes, window_size)
            n_edges = [item for sublist in n_edges for item in sublist]
            n_edges = [n_edges[i] for i in perm]
            n_edges = gaussian_kernel(n_edges, window_size)
            avg_deg = [item for sublist in avg_deg for item in sublist]
            avg_deg = [avg_deg[i] for i in perm]
            avg_deg = gaussian_kernel(avg_deg, window_size)
            density = [item for sublist in density for item in sublist]
            density = [density[i] for i in perm]
            density = gaussian_kernel(density, window_size)

            for i in faithfulness:
                faithfulness[i] = [item for sublist in faithfulness[i] for item in sublist]
                faithfulness[i] = [faithfulness[i][j] for j in perm]
                faithfulness[i] = gaussian_kernel(faithfulness[i], window_size)

        fig = make_subplots(
            rows=4 + len(list(faithfulness.keys())),
            cols=1,
        )

        for i, fn_name in enumerate(outs[thresholds[0]][faith]):
            fig.add_trace(go.Scatter(
                    x=n_nodes,
                    y=faithfulness[i],
                    mode='lines' if is_list else 'lines+markers',
                    #title_text=fn_name+" faithfulness vs threshold",
                    name=fn_name,
                ),
                row=i+1, col=1
            )

        fig.add_trace(
            go.Scatter(
                x=n_nodes,
                y=n_nodes,
                mode='lines' if is_list else 'lines+markers',
                #title_text="n_nodes vs threshold",
                name='n_nodes',
            ),
            row=len(list(faithfulness.keys()))+1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=n_nodes,
                y=n_edges,
                mode='lines' if is_list else 'lines+markers',
                #title_text="n_edges vs threshold",
                name='n_edges',
            ),
            row=len(list(faithfulness.keys()))+2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=n_nodes,
                y=avg_deg,
                mode='lines' if is_list else 'lines+markers',
                #title_text="avg_deg vs threshold",
                name='avg_deg',
            ),
            row=len(list(faithfulness.keys()))+3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=n_nodes,
                y=density,
                mode='lines' if is_list else 'lines+markers',
                #title_text="density vs threshold",
                name='density',
            ),
            row=len(list(faithfulness.keys()))+4, col=1
        )

        # Update x-axes to log scale
        fig.update_xaxes(type="linear")
        # Update y axes to crop between -0.5 and 1.5
        for i in range(1, 5 + len(list(faithfulness.keys()))):
            y_range = fig['data'][i-1]['y']
            y_min = max(min(y_range), -0.5)
            y_max = min(max(y_range), 1.5)
            fig.update_yaxes(range=[y_min, y_max], row=i, col=1)

        # default layout is : height=600, width=800. We want to make it a bit bigger so that each plot has the original size
        fig.update_layout(
            height=600 + 400 * (4 + len(list(faithfulness.keys()))),
            width=800,
            title_text=faith+" and graph properties w.r.t. threshold",
            showlegend=True,
        )

        if save_path is None:
            # show the plot
            fig.show()
            
        else:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            fig.write_html(save_path + faith + ".html")
            # also save as png
            fig.write_image(save_path + faith + ".png")
            # also save as json
            fig.write_json(save_path + faith + ".json")

DEFAULT_COLOR = {
    "Marks et al. (original node ablation)": "rgb(255, 228, 181)",
    "Marks et al. (edge ablation)": "rgb(173, 216, 230)",
    "Ours (node ablation)": "rgb(255, 165, 0)",
    "Ours (edge ablation)": "rgb(0, 0, 255)"
}

def fuse_faithfulness(figure_paths, experiment_labels, save_path, default_color=None):
    # load all figures
    figs = []
    for path in figure_paths:
        figs.append(plotly.io.read_json(path))
    
    all_results = {}
    all_trace_names = []
    
    # extract data points.
    for i, experiment in enumerate(experiment_labels):
        all_results[experiment] = {}
        fig = figs[i]
        for trace in fig['data']:
            name = trace['name']
            all_results[experiment][name] = trace['y']
            if name not in all_trace_names:
                all_trace_names.append(name)

    # create a new figure for each trace name
    for trace_name in all_trace_names:
        fig = go.Figure()
        for experiment in experiment_labels:
            fig.add_trace(go.Scatter(
                x=all_results[experiment]['n_nodes'],
                y=all_results[experiment][trace_name],
                mode='lines+markers',
                name=experiment,
                line=dict(color=default_color[experiment] if default_color is not None else DEFAULT_COLOR.get(experiment, None)),
            ))
        fig.update_layout(
            title='',
            xaxis_title='Nodes',
            yaxis_title=trace_name,
            width=800,
            height=375,
            # set white background color
            plot_bgcolor='rgba(0,0,0,0)',
            # add grey gridlines
            yaxis=dict(gridcolor='rgb(200,200,200)',mirror=True,ticks='outside',showline=True),
            xaxis=dict(gridcolor='rgb(200,200,200)', mirror=True, ticks='outside', showline=True),
            showlegend=True
        )
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # save in json, pdf and html
        fig.write_json(save_path + trace_name + '.json')
        fig.write_image(save_path + trace_name + '.pdf')
        fig.write_html(save_path + trace_name + '.html')
