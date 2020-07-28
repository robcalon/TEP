import requests
import copy

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

import shapely.geometry
import geopandas.tools


def feature_layer_to_shapely(flayer, columns=None):
    """Transform a Feature Layer to a GeoDataFrame.
    
    Parameters
    ----------
    flayer : FeatureLayer
        Feature Layer that needs to be transformed.
    columns : list, default None
        List of columns names to include in the GeoDataFrame
    
    Returns
    -------
    gdf : GeoDataFrame
        Returns a GeoDataFrame with geometry based on 
        ``Shapely Geometric Objects``."""
    
    sdf = pd.DataFrame.spatial.from_layer(flayer)
    
    crs = layer.properties.extent.spatialReference.latestWkid
    crs = 'epsg:{}'.format(crs)
    
    gdf = esri_to_shapely_geometry(sedf=sdf, geocol='SHAPE', crs=crs, columns=columns)
    
    return gdf


def esri_to_shapely_geometry(sedf, geocol, crs, columns=None):
    """Transform a Spationally Enabled DataFrame to a GeoDataFrame.
    
    Parameters
    ----------
    sefd : DataFrame
        The Spationally Enabled DataFrame, e.g. based on an FeatureLayer.
    geocol : str
        Column name of columns that contains the geometry infromation.
    crs : str
        The Coordinate Reference System (CRS) represented as a ``pyproj.CRS``.
    columns : list, default None
        List of columns in sefd to return.
        
    Returns
    -------
    gdf : GeoDataFrame
        Return a GeoDataFrame with geometry based on 
        ``Shapely Geometric Objects``.""" 
    
    # copy dataframe
    df = sedf.copy(deep=True)
    
    # shake geocol
    del df[geocol]
    
    # iterate over all shape items
    for index, esri in sedf[geocol].items():
        
        if esri is None:
            # add an empty geometry entry
            df.at[index, 'geometry'] = shapely.geometry.Point()
            
        else:
            # map the esri geometry
            shapemap = shapely.geometry.mapping(esri)
            
            # transform esri in shapely
            df.at[index, 'geometry'] = shapely.geometry.shape(shapemap)
    
    # transforms df in gdf with correct crs
    gdf = gpd.GeoDataFrame(df, crs=crs)
        
    # shake unwanted columns
    if columns is not None:
        
        if 'geometry' not in columns:
            columns.append('geometry')
        
        gdf = gdf[columns]
    
    return gdf


def join_spatial_data(buses, geographics, dropnan=False, coords=False, xy=False):
    """Perform a spatial join of the bus and geographic data. Transforms
    the geometry of the buses-argument into Points.
    
    Parameters
    ----------
    buses, geographics : GeoDataFrame
        GeoDataFrame with spatial information.
    dropnan : bool, default False
        Drop empty and NaN geometries inside GeoDataFrame.
    coords : bool, default False
        Store coordinate tuples in a coords column.
    xy : bool, default False
        Store coordinates in seperate x and y columns.
    
    Returns
    -------
    sbuses : GeoDataFrame
        Return a spatialy joined GeoDataFrame.""" 
    
    # check geometry types
    if not isinstance(buses, gpd.GeoDataFrame) | isinstance(geographics, gpd.GeoDataFrame):
        raise TypeError('not a GeoDataFrame')
    
    # check for empty geometry
    if buses['geometry'].isna().sum() + buses['geometry'].is_empty.sum() > 0:
        if dropnan:
            buses = (buses[~(buses['geometry'].is_empty | buses['geometry'].isna())])
        else:
            raise ValueError('"buses" GeoDataFrame contains empty or NaN geometry values')
    
    # aggregate a representative point from busgeometry
    buses['geometry'] = buses.representative_point()

    # perform a spacial join
    # drop index right column
    sbuses = geopandas.tools.sjoin(buses, geographics, how='left')
    sbuses = sbuses.drop('index_right', axis=1)
    
    if coords:
        if 'coords' not in sbuses.columns:
            # get coordinates
            sbuses['coords'] = None
    
    if xy:
        if 'x' not in sbuses.columns:
            sbuses['x'] = None
        if 'y' not in sbuses.columns:
            sbuses['y'] = None
    
    if coords or xy:
        for index, entry in sbuses.iterrows():
            if coords:
                coord = entry.geometry.coords[:]
                sbuses.at[index, 'coords'] = coord[0]

            if xy:
                sbuses.at[index, 'x'] = entry.geometry.x
                sbuses.at[index, 'y'] = entry.geometry.y
        
    # position geometry to back of df
    columns = sbuses.columns.drop('geometry').to_list()
    columns.append('geometry')
    sbuses = sbuses[columns]
    
    return sbuses


def get_gisco_geodata(base, info, key):
    """Fetch geodata from EU's GISCO database.
    
    Parameters
    ----------
    base : string
        String of the URL that specifies the base location
        of the requested geodata.
    info : string
        String subdomain based on the basestring of where
        the overview page of all fetchable geodata is located.
    key : string
        Selected geodata configuration key that will be fetched
        by this function.
        
    Return
    ------
    gdf : GeoDataFrame
        Returns a geodataframe of the selected geodata configuration."""
    
    # construct info url
    url = base + info

    # fetch nuts information
    resp = requests.get(url=url)
    dic = resp.json()

    # get geoformat of key
    geoformat = key.rsplit('.', maxsplit=1)[1]
    
    # construct data url
    data = dic[geoformat][key]
    url = base + data

    # fetch data
    gdf = gpd.read_file(url)
    
    return gdf


def get_nutsdata_eu(key):
    """Fetch NUTS-region geodata from EU's GISCO database. 
    This is a preconfigured call of the get_gisco_geodata
    function.
    
    Parameters
    ----------
    key : string
        Selected geodata configuration key that will be 
        fetched by this function.
        
    Return
    ------
    gdf : GeoDataFrame
        Returns a geodataframe of the selected geodata 
        configuration."""
    
    base = 'https://ec.europa.eu/eurostat/cache/GISCO/distribution/v2/nuts/'
    info = 'nuts-2016-files.json'
    
    gdf = get_gisco_geodata(base=base, info=info, key=key)
    
    return gdf


def plot_basemap(gdf, ax=None, xrange=None, yrange=None, borders=None):
    """Plot a basic map of the passed GeoDataFrame.
    
    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame with map information.
    ax : AxesSubplot, default None
        Axis on which to plot the map.
    xrange, yrange : tuple, default None
        Coordinate ranges to plot on the AxesSubplot
    borders : str, default None
        Columns name of parent class in the GeoDataFrame. The
        argument is used to dissolve the geometries and plot
        borders to group individual geometries.
    
    Return
    ------
    ax : AxesSubplot
        Displays a basic figure of the geometries inside 
        the GeoDataFrame.""" 
    
    # get fig dimensions
    if xrange is not None:
        x = (xrange[1] - xrange[0])
        
    if yrange is not None:
        y = (yrange[1] - yrange[0])
    
    # specify a figure
    if ax is None:
        if xrange is None or yrange is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = plt.subplots(figsize=(x,y))

    # set background color
    ax.set_facecolor('skyblue')

    data = simple_cx(gdf=gdf, xrange=xrange, yrange=yrange)
    
    # plot NUTS-regions within plot domain
    data.plot(ax=ax, ec='slategrey', fc='papayawhip', linewidth=0.25)
        
    # aggregate NUTS-regions to get borders
    # plot boundaries of aggregated geometry
    if borders is not None: 
        countries = data.dissolve(by=borders)
        countries.boundary.plot(ax=ax, ec='slategrey', linewidth=0.75)

    # adjust plot domain    
    if xrange is not None:
        plt.xlim(xrange[0], xrange[1])
    if yrange is not None:
        plt.ylim(yrange[0], yrange[1])

    # remove ticks
    plt.xticks([])
    plt.yticks([])
    
    return ax


def plot_eu_basemap(ax=None, xrange=None, yrange=None, key=None, borders=None):
    """Plot a basic map of EU nutsdata.
    
    Parameters
    ----------
    ax : AxesSubplot, default None
        Axis on which to plot the map.
    xrange, yrange : tuple, default None
        Coordinate ranges to plot on the AxesSubplot
    key : str
        Selected geodata configuration key that will be 
        fetched by the get_eu_nutsdata function.
    borders : str, default None
        Columns name of parent class in the GeoDataFrame. The
        argument is used to dissolve the geometries and plot
        borders to group individual geometries.
    
    Return
    ------
    ax : AxesSubplot
        Displays a basic figure of the geometries inside 
        the GeoDataFrame."""
    
    if key is None:
        key = 'NUTS_RG_03M_2016_4326_LEVL_2.geojson'
        
    if borders is None:
        borders = 'CNTR_CODE'
    
    gdf = get_nutsdata_eu(key=key)
    
    ax = plot_basemap(gdf=gdf, ax=ax, xrange=xrange, yrange=yrange, borders=borders)
    
    return ax

def simple_cx(gdf, xrange=None, yrange=None):
    """Takes a subset of a GeoDataFrame based on a
    coordinate range.
    
    Parameters
    ----------
    gdf : GeoDataFrame
        The GeoDataFrame that is subsetted.
    xrange, yrange : (float, float), default None
        The coordinate range over which the subset is taken
    
    Return
    ------
    sgdf : GeoDataFrame
        Returns a subsetted GeoDataFrame"""
    
    # check geodataframe
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError('gdf is not a GeoDataFrame')
    
    # fetch data within specified plot domain
    if xrange is not None and yrange is not None:    
        sgdf = gdf.cx[xrange[0]:xrange[1], yrange[0]:yrange[1]]
        
    elif xrange is None and yrange is None:
        sgdf = gdf
        
    elif yrange is None:
        sgdf = gdf.cx[xrange[0]:xrange[1], :]
        
    else:
        sgdf = gdf.cx[:, yrange[0]:yrange[1]]
    
    return sgdf
    

def assign_nuts_from_geodata(network, nuts, columns, crs=None, 
                             ignorenan=False, overwrite=False):
    """Assign NUTS regions based on geometry information.
    
    Parameters
    ----------
    network : pandapowerNet
        PandaPower network representation that includes 
        coordinate information.
    nuts : GeoDataFrame
        GeoDataFrame that includes NUTS information and
        geometry.
    columns : str or lists
        Column name(s) of column(s) that contains the mapped 
        NUTS information.
    crs : str, default None
        The Coordinate Reference System (CRS) of the network 
        coordinates, represented as a ``pyproj.CRS``.
    ignorenan : bool, default False
        Ignore NaN-values in returned network.
    overwrite : bool, default False
        Overwrite 
        
    Return
    ------
    net : pandapowerNet
        PandaPower network representation that includes
        NUTS information."""
    
    if isinstance(columns, str):
        columns = [columns]
    
    if not isinstance(columns, list):
        try:
            list(columns)
        except:
            raise TypeError('{} not a list or string'.format(nutscols))
    
    if not isinstance(nuts, gpd.GeoDataFrame):
        raise TypeError('nuts is not a GeoDataFrame')
    
    # validate columns are in nuts
    for column in columns:
        if column not in nuts.columns:
            raise KeyError('"{}" not a column in nuts'.format(column))
    
    # raise error when geometry is assigned to network
    if 'geometry' in columns:
        raise TypeError('Geometries cannot be assigned to a pandapower network')
    
    # copy network
    net = copy.deepcopy(network)

    for column in columns:
        if column in net.bus_geodata:
            if overwrite:
                del net.bus_geodata[column]
            else:
                raise KeyError(f'"{column}" already present in network bus ' +
                               'geodata, set overwrite=True to overwrite the column')    
    
    # convert buses to points
    busgeo = buses_to_points(network=net)
    
    if crs is None:
        busgeo.crs = nuts.crs
    
    else:
        busgeo.crs = crs
         
        if nuts.crs is None:
            nuts.crs(crs)
        else:
            nuts.to_crs(crs)
    
    cslice = copy.deepcopy(columns)
    cslice.append('geometry')

    # perform spatial join and return xy coords.
    busgeo = join_spatial_data(busgeo, nuts[cslice])
    
    # check for nan values
    if ignorenan is False:
        if busgeo[columns].isna().sum().sum() > 0:
            raise ValueError('geometry coordinate(s) in the network bus geodata ' + 
                             'are outside the geometry domain of nuts, set ' +
                             'ingorenan=True to ignore NaN values')
        
    # apply result to network copy
    net.bus_geodata = net.bus_geodata.join(busgeo[columns])
    
    return net


def lines_to_linestrings(network, index=None, method='AC'):
    """Converts selected lines to a LineString based GeoDataFrame.
    
    Parameters
    ----------
    network : pandapowerNet
        Pandapower network representation.
    index : index, default None
        Indices of lines to convert to LineString geometry.
    method : string, default 'AC'
        Linetype on which to apply the function. 
        Can be either 'AC' or 'DC'.
    
    Return
    ------
    gdf : GeoDataFrame
        Returns a GeoDataFrame of the selected lines."""
    
    if method == 'AC':
        lines = network.line.copy(deep=True)
    elif method == 'DC':
        lines = network.dcline.copy(deep=True)
    else:
        raise KeyError('"{}" is an unsupported method'.format(method))
        
    if index is not None:
        lines = lines.iloc[index]
        
    # convert buses to points
    busgeo = buses_to_points(network=network)
    
    if 'geometry' not in lines.columns:
        lines['geometry'] = None
    
    # iterate over each line
    for index, line in lines.iterrows():
        # create line geometry based on bus coordinates
        busa = busgeo['geometry'].iloc[line.from_bus] 
        busb = busgeo['geometry'].iloc[line.to_bus]

        # assign line geometry to lines dataframe
        lines.at[index, 'geometry'] = shapely.geometry.LineString([busa, busb])
    
    # convert to gdf
    gdf = gpd.GeoDataFrame(lines, geometry='geometry')
    
    return gdf


def buses_to_points(network, index=None):
    """Converts selected buses to a Point based GeoDataFrame.
    
    Parameters
    ----------
    network : pandapowerNet
        PandaPower network representation.
    index : index, default None
        Indices of buses to convert to Point geometry
        
    Return
    ------
    gdf : GeoDataFrame
        Returns a GeoDataFrame of the selected buses."""
    
    # get geodata of buses
    if index is None:
        buses = network.bus.copy(deep=True)
        busgeo = network.bus_geodata.copy(deep=True)
    else:
        buses = network.bus.iloc[index]
        busgeo = network.bus_geodata.iloc[index]
    
    # transform dataframe into geodataframe
    geometry = gpd.points_from_xy(busgeo.x, busgeo.y)
    gdf = gpd.GeoDataFrame(buses, geometry=geometry)
    
    return gdf


def simple_border_points(network, borders, method='AC'):
    """Get points where lines that are connected in different zones
    cross a border. The function only accepts a borders GeoDataFrame
    that contains (Multi)LineString geometries. (Multi)Polygons can be
    transformed to (Multi)LineStrings trough the "gdf.boundary" method.
    
    !!!! ONLY AN APPROXIMATION, LINES ARE ASSUMED TO BE 
    "AS THE CROW FLIES" BETWEEN THE FROM AND TO BUSES !!!!
    
    !!!! DOES NOT INCLUDE POINTS OF LINES THAT HAVE A FROM
    AND TO BUS IN THE SAME ZONE, WHERE THE LINE CROSSES A 
    BORDER !!!!
    
    Parameters
    ----------
    network : pandapowerNet
        PandaPower network representation.
    borders : GeoDataFrame
        GeoDataFrame that contains border information.
        Geometries can be of type (Multi)LineString. 
    method : string, default 'AC'
        Linetype on which to apply the function. 
        Can be either 'AC' or 'DC'.
    
    Return
    ------
    gdf : GeoDataFrame
        GeodataFrame with border crossing points of each line.
        Lines that do not cross a border are represented as 
        Empty Geometries."""
    
    if method == 'AC':
        lines = network.line.copy(deep=True)
    elif method == 'DC':
        lines = network.dcline.copy(deep=True)
    else:
        raise KeyError('"{}" is an unsupported method'.format(method))
    
    # get the lines that have a from and to bus in a different zone
    # convert lines to GeoDataFrame
    lines = lines[lines.from_bus.map(network.bus.zone) != lines.to_bus.map(network.bus.zone)]
    lines = lines_to_linestrings(network=network, index=lines.index, method=method)
    
    # define an empty df
    df = pd.DataFrame(index=lines.index)
    df['borders'] = np.empty((len(df), 0)).tolist()
    df['geometry'] = np.empty((len(df), 0)).tolist()
    
    # copy dataframe
    cborders = borders.copy(deep=True)
    
    # convert linestrings to multilinestrings
    cborders.geometry = cborders.geometry.apply(linestring_to_multilinestring)
    
    # iterate over each line in lines
    for idx, line in lines.iterrows():
        # iterate over each region in borders
        for index, region in cborders.iterrows():
                
            # iterate over each LineString in the MultiLineString
            for linestring in region.geometry:
                # if region and line intersect
                if linestring.intersects(line.geometry):

                    # get intersection point
                    res = linestring.intersection(line.geometry)

                    if isinstance(res, Point):
                        df.at[idx, 'borders'].append(index)
                        df.at[idx, 'geometry'].append(res.coords[:][0])

                    elif isinstance(res, MultiPoint):
                        for point in res:
                            df.at[idx, 'borders'].append(index)
                            df.at[idx, 'geometry'].append(point.coords[:][0])

                    else:
                        raise TypeError('"{}" is an unsupported type'.format(res.type))
    
    # find unique values
    df.borders = df.borders.apply(pd.unique)
    df.geometry = df.geometry.apply(pd.unique)
    
    # apply geometry
    df.geometry = df.geometry.apply(coords_to_multipoint)
    
    # transform to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    
    return gdf


def linestring_to_multilinestring(geometry):
    """Helper function to convert a LineString to a MultiLineString.

    Parameters
    ----------
    geometry : LineString or MultiLineString
        LineString that has to be converted to a MultiLineString.

    Return
    ------
    geometry : MultiLineString
        Returns a MultiLineString geometry."""

    if isinstance(geometry, MultiLineString):
        return geometry
    elif isinstance(geometry, LineString):
        return shapely.geometry.MultiLineString([geometry])
    else:
        raise NotImplemented('"{}" is not implemented'.format(geometry.type))

        
def coords_to_multipoint(coords):
    """Helper function to convert coordinates to a Point or a MultiPoint.

    Parameters
    ----------
    coords : list of tuples
        List of tuples with Point coordinates.

    Return
    ------
    geometry : Point or MultiPoint
        Returns a Point or MultiPoint geometry."""

    if len(coords) == 1:
        return shapely.geometry.Point(coords)
    else:
        return shapely.geometry.MultiPoint(coords)