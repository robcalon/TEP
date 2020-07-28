import numpy as np
import pandas as pd

import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt

import matplotlib.colors

from ..utility.utility import invert_dict

## Trafo implementation
## Directed (weighted) graphs
## Plotting order of zones (include? or order? arguments)


def plot_network(network, ax=None, zones=None, line_weights=None, vmin=None, vmax=None, cmap=None,
                 zone_config=None, label_order=None, legend_loc=None, cbar_label=None, cbar_fraction=0.015):
    """Plot a NetworkX-based version of a network.
    
    Parameters
    ----------
    network : pandapowerNet
        PandaPower network representation.
    ax : AxesSubplot, default None
        Axis on which the figure is plotted.
    zones : list, default None
        list of zones to plot
    line_weights : dict, default None
        dictonairy with weights for each line
    vmin, vmax : float, default None
        minimum
    cmap : list, default None
        list of colors to show build a cmap
    zone_config : dict, default None
        d
    label_order : ?
        d
    legend_loc : ?, default None
        d
    cbar_label : ?, default None
        d
    cbar_fraction : ?, default 0.015
        d
    
    Returns
    ------
    ax : AxesSubplot
        Returns the passed axes.
    
    """ 
    if ax is None:
        fig, ax = plt.subplots()
    
    draw_network(network=network, ax=ax, zones=zones, line_weights=line_weights, vmin=vmin, vmax=vmax, 
                 cmap=cmap, zone_config=zone_config, cbar_label=cbar_label, cbar_fraction=cbar_fraction)
    
    sorted_legend(ax=ax, label_order=label_order, legend_loc=legend_loc)
    
    return ax
    
    
def draw_network(network, ax, zones=None, line_weights=None, vmin=None, vmax=None, 
                 cmap=None, zone_config=None, cbar_label=None, cbar_fraction=.015):
    """
    info
    
    Parameters
    ----------
    
    Return
    ------
    ax : AxesSubplot
        Returns the passed axes.
    """
    if zones is None:
        zones = network.bus.zone.unique()
    
    extension = ''
    
    pos = dict(zip(network.bus_geodata.index, zip(network.bus_geodata.x, network.bus_geodata.y)))
    
    if line_weights is not None:
        
        extension = '_weighted'
        
        if isinstance(line_weights, np.ndarray):
            line_weights = pd.Series(line_weights)
        
        if isinstance(line_weights, pd.Series):
            line_weights = line_weights.to_dict()
            
        if not isinstance(line_weights, dict):
            raise TypeError('line_weights must be a dictonairy')
        
        if cmap is None:
            cmap = ['green', 'yellow', 'red']

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('mycmap', cmap)
        
        if vmax is None:

            buss = network.bus[network.bus['zone'].isin(zones)].index
            subset = {key:value for key, value in line_weights.items() if key in buss}
            
            vmax = max(subset.values())
            
        if vmin is None:
            
            buss = network.bus[network.bus['zone'].isin(zones)].index
            subset = {key:value for key,value in line_weights.items() if key in buss}
            
            vmin = max(0, min(subset.values()))
        
        # add colorbar
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        
        if cbar_label is None:
            plt.colorbar(sm, fraction=cbar_fraction, ax=ax)

        else:
            plt.colorbar(sm, fraction=cbar_fraction, ax=ax).set_label(cbar_label)
    
    if zone_config is None:
        zone_config = {}
    elif isinstance(zone_config, pd.DataFrame):
        zone_config = zone_config.set_index('zone')
        zone_config = zone_config.to_dict(orient='index')
    elif not isinstance(zone_config, dict):
        raise TypeError('zone_config is not a pd dataframe or a dict')
    
    for zone in network.bus.zone.unique():
        if zone in zones:
            # collect edge configuration
            edge_color = zone_config.get(zone, {}).get('edge_color{}'.format(extension), None)
            edge_width = zone_config.get(zone, {}).get('edge_width{}'.format(extension), None)
            edge_label = zone_config.get(zone, {}).get('edge_label{}'.format(extension), None)

            # collect node configuration
            node_color = zone_config.get(zone, {}).get('node_color{}'.format(extension), None)
            node_shape = zone_config.get(zone, {}).get('node_shape{}'.format(extension), None)
            node_size  = zone_config.get(zone, {}).get('node_size{}'.format(extension), None)
            node_label = zone_config.get(zone, {}).get('node_label{}'.format(extension), None)

            draw_zone(network=network, zone=zone, pos=pos, ax=ax, line_weights=line_weights, vmin=vmin, 
                      vmax=vmax, edge_color=edge_color, cmap=cmap, edge_width=edge_width, edge_label=edge_label, 
                      node_color=node_color, node_shape=node_shape, node_size=node_size, node_label=node_label)
            
    return ax
    
    
def draw_zone(network, zone, pos, ax, line_weights=None, vmin=None, vmax=None, edge_color=None, edge_width=None, 
              cmap=None, edge_label=None, node_color=None, node_shape=None, node_size=None, node_label=None):
    """
    info
    
    Parameters
    ----------
    
    Return
    ------
    ax : AxesSubplot
        Returns the passed axes.
    """
    
    if zone not in network.bus.zone.unique():
        raise KeyError('"{}" is not a zone in the network'.format(zone))
    
    buses = network.bus[network.bus.zone == zone]
    aclines = network.line[(network.line.from_bus.isin(buses.index)) | (network.line.to_bus.isin(buses.index))]
    dclines = network.dcline[(network.dcline.from_bus.isin(buses.index)) | 
                             (network.dcline.to_bus.isin(buses.index))]    
    
    draw_subgraph(buses=buses, aclines=aclines, dclines=dclines, ax=ax, pos=pos, edge_weights=line_weights,
                  vmin=vmin, vmax=vmax, edge_color=edge_color, edge_width=edge_width, cmap=cmap, 
                  edge_label=edge_label, node_color=node_color, node_shape=node_shape, node_size=node_size,
                  node_label=node_label)
    
    return ax

    
def draw_subgraph(buses, aclines, dclines, pos, ax, edge_weights=None, vmin=None, 
                  vmax=None, edge_color=None, edge_width=None, cmap=None, edge_label=None, 
                  node_color=None, node_shape=None, node_size=None, node_label=None):
    """
    info
    
    Parameters
    ----------
    
    Return
    ------
    ax : AxesSubplot
        Returns the passed axes.
    """
    
    G = nx.Graph()
    
    if buses is not None:
        draw_nodes(G=G, nodes=buses, pos=pos, ax=ax, node_color=node_color, 
                   node_shape=node_shape, node_size=node_size, node_label=node_label)
    
    if dclines is not None:
        draw_edges(G=G, edges=dclines, pos=pos, ax=ax, edge_color=edge_color, 
                   edge_width=edge_width, edge_label=edge_label)
        
    if edge_weights is None:
    
        if aclines is not None:
            draw_edges(G=G, edges=aclines, pos=pos, ax=ax, edge_color=edge_color, 
                       edge_width=edge_width, edge_label=edge_label)
        
    else:
        
        if aclines is not None:
            draw_weighted_edges(G=G, edges=aclines, weights=edge_weights, pos=pos, ax=ax, 
                                edge_width=edge_width, vmin=vmin, vmax=vmax, cmap=cmap, edge_label=edge_label)   
            
    return ax
            
        
def draw_nodes(G, nodes, pos, ax, node_color=None, node_shape=None, node_size=None, node_label=None):
    """
    info
    
    Parameters
    ----------
    
    Return
    ------
    ax : AxesSubplot
        Returns the passed axes.
    """
    
    for index, node in nodes.iterrows():
        G.add_node(index)
    
    nx.draw_networkx_nodes(G, ax=ax, pos=pos, node_color=node_color, node_shape=node_shape,
                          node_size=node_size, label=node_label)
    
    return ax
    

def draw_edges(G, edges, pos, ax, edge_color=None, edge_width=None, edge_label=None):
    """
    info
    
    Parameters
    ----------
    
    Return
    ------
    ax : AxesSubplot
        Returns the passed axes.
    
    """
    current = G.edges()
    
    for index, edge in edges.iterrows():
        G.add_edge(edge.from_bus, edge.to_bus)
    
    nx.draw_networkx_edges(G, ax=ax, pos=pos, edge_color=edge_color, width=edge_width, label=edge_label)
    
    return ax
    
    
def draw_weighted_edges(G, edges, weights, pos, ax, edge_width=None, 
                        vmin=None, vmax=None, cmap=None, edge_label=None):
    """
    info
        
    Parameters
    ----------
    
    Return
    ------
    ax : AxesSubplot
        Returns the passed axes.
    """
        
    for index, edge in edges.iterrows():
        G.add_edge(edge.from_bus, edge.to_bus, weight=weights[index])
    
    nx.draw_networkx_edges(G, ax=ax, pos=pos, width=edge_width, edge_vmin=vmin, edge_vmax=vmax, edge_cmap=cmap, 
                           label=edge_label, edge_color=nx.get_edge_attributes(G, 'weight').values())
    
    return ax
        
    
def sorted_legend(ax, label_order=None, legend_loc=None):
    """
    info
    
    Parameters
    ----------
    
    Return
    ------
    ax : AxesSubplot
        Returns the passed axes.
    
    """
    
    if isinstance(label_order, pd.DataFrame):
        label_order = dict(zip(label_order.label, label_order.position))
    elif not isinstance(label_order, dict):
        raise TypeError('label_order is not a dataframe or a dict')
    
    # get handles and labels of the network
    handles, labels = ax.get_legend_handles_labels()
    
    # drop all duplicate label entries
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    
    # when a custom label_order is passed
    if label_order is not None:
        
        # check if there are no duplicate positions
        if len(label_order) != len(set(label_order.values())):
            raise ValueError('not all positions in label_order are unique')

        # make a translation dictonairy
        inverted_order = invert_dict(label_order)

        # convert label names into positions
        unique = [(h, label_order[l]) for (h, l) in unique]

    # sort positions and corresponding handles
    ordened = sorted(unique, key=lambda x: x[1])

    # when a custom label_order is passed
    if label_order is not None:

        # convert positions back into labels
        ordened = [(h, inverted_order[l]) for (h, l) in ordened]

    # pass handles and labels to legend
    ax.legend(*zip(*ordened), loc=legend_loc, title='Legend')
    
    return ax