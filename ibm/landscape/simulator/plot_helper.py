import plotly.graph_objects as go
from matplotlib import pyplot as plt
from landscape_helper import *

def get_hover_template():
    return "γ: %{x:.2f}<br>β: %{y:.2f}<br>Exp.Value: %{z:.2f}<extra></extra>"

def get_text_hover_template():
    return get_hover_template().replace('z','text')

def plot_landscape_3d(z):
    # Plot landscape in 3D 

    fig_landscape_3d = go.Figure(data=go.Surface(x=gamma_range, y=beta_range, z=z))

    fig_landscape_3d.update_traces(contours_z=dict(show=True, usecolormap=True, highlightcolor='limegreen', project_z=True))


    fig_landscape_3d.update_layout(title="QAOA MaxCut Landscape 3D", scene=dict(
        xaxis_title="γ",
        yaxis_title="β",
        zaxis_title="expectation value"
    ))
    
    fig_landscape_3d.show()
    
def plot_heatmap(landscape, show=False):
    # Plot Heatmap 
    fig_heatmap = go.Figure(data=go.Heatmap(
                            z=landscape, y=beta_range, x=gamma_range, type = 'heatmap', colorscale = 'viridis', name="Landscape",
                            hovertemplate=get_hover_template()),
                           
                           )

    # Update Layout
    fig_heatmap.update_layout(
        title="QAOA MaxCut Landscape 2D (p=1)", width=1000, height=700, yaxis_title="β", xaxis_title="γ"
    )

    fig_heatmap.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=1.2
    ))
    if show:
        fig_heatmap.show()
    
    return fig_heatmap

# anywhere is some zahlendreher, x!=gamma, y!=beta 
def display_minimum(fig, y, x, z, show=True):
    # Display Global Minimium 
    fig.add_trace(
        go.Scatter(mode="markers", x=[y], y=[x], marker_symbol=[204], text = [z],
                   marker_color="red",
                   hovertemplate="Global Minimium<br>"+get_text_hover_template(), 
                   marker_line_width=1, marker_size=16, name=f"Global Minimium {z}")
    )
    
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=1.2
    ))
    if show:
        fig.show()
        
    return fig 
        
        
def display_optimizer_path(fig, optimizer_gammas, optimizer_betas, maxcut_values, name):
    # Display Optimizer Results
    
    optimizer_fig = go.Figure(fig)

    # Display path 
    optimizer_fig.add_trace(
        go.Scatter(mode="lines", y=optimizer_gammas, x=optimizer_betas, marker_symbol=[200],
                       marker_color="white", marker_line_width=1, marker_size=8, name=f"Optimizerpath - {name}",)
    )

    # Display start point
    optimizer_fig.add_trace(
        go.Scatter(mode="markers", y=[optimizer_gammas[0]], x=[optimizer_betas[0]], marker_symbol=[204],
                       text = [maxcut_values[0]],
                       marker_color="gray", hovertemplate="Startpoint<br>"+get_text_hover_template(), 
                       marker_line_width=1, marker_size=16, name=f"Startpoint {maxcut_values[0]}",))

    # Display end point
    optimizer_fig.add_trace(
        go.Scatter(mode="markers", y=[optimizer_gammas[-1]], x=[optimizer_betas[-1]], marker_symbol=[204],
                   text = [maxcut_values[-1]],    
                   marker_color="green", hovertemplate="Endpoint<br>"+get_text_hover_template(), 
                       marker_line_width=1, marker_size=16, name=f"Endpoint {maxcut_values[-1]}",))

    
    optimizer_fig.show()
    
    
    
def plot_optimizer_maxcut_history(x,y,name):
    fig_maxcut = go.Figure(data=go.Scatter(x=x, y=y))
    fig_maxcut.update_layout(xaxis_title="Evaluation Counts",height=500, width=700, yaxis_title="Evaluated MaxCut Mean", title=f"{name} Optimizer")
    fig_maxcut.show()

def plot_optimizer_energy_history(x,y,name):
    # Plot Optimizer History Energy Evaluation -> not MaxCutMean! 
    fig_energy = go.Figure(data=go.Scatter(x=x, y=y))
    fig_energy.update_layout(xaxis_title="Evaluation Counts", height=500, width=700, yaxis_title="Evaluated Energy Mean", title=f"{name} Optimizer")
    fig_energy.show()
    
    
    
# evaluation plots 

def plot_exp_evaluation_results(results, name):
    
    fig = go.Figure()
    for i,rs in enumerate(results):
        fig.add_trace(
            go.Box(
                y=rs,
                name=f"p={i+1}",
                marker_color='darkblue',
                boxmean=True,
                boxpoints='all',
            )
        )
    fig.update_layout(
        title=name,
        yaxis_title='expectation value',
    )
    fig.show()
    
def plot_exp_evaluation_results_matplotlib(results, name):
    fig1, ax1 = plt.subplots()
    ax1.set_title(name)
    ax1.boxplot(results, meanline=True, showmeans=True)
    ax1.set_ylabel("expectation value")
    ax1.set_xlabel("p")
    fig1.show()
    
def plot_ratio_results_matplotlib(results, name):
    fig1, ax1 = plt.subplots()
    ax1.set_title(name)
    ax1.boxplot(results, meanline=True, showmeans=True)
    ax1.set_ylabel("ratio")
    ax1.set_xlabel("p")
    fig1.show()
    
def plot_approx_ratio_results_matplotlib(results, name):
    fig1, ax1 = plt.subplots()
    ax1.set_title(name)
    ax1.boxplot(results, meanline=True, showmeans=True)
    ax1.set_ylabel("approximation ratio")
    ax1.set_xlabel("p")
    fig1.show()
    

def display_boxplots_results(means, ratios, approx_ratios, prefix=''):
    #plot_exp_evaluation_results(means, f'{prefix}QAOA: Expectation Value')
    plot_exp_evaluation_results_matplotlib(means, f'{prefix}QAOA: Expectation Value')
    plot_ratio_results_matplotlib(ratios, f'{prefix}QAOA: Ratio')
    plot_approx_ratio_results_matplotlib(approx_ratios, f'{prefix}QAOA: Approximation Ratio')
    
    
    
    
