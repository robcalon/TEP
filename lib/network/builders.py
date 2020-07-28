import copy

import pandas as pd
import pandapower as pp

def build_base_network(dir):
    """
    Build a PandaPower network based from csv files.
    
    Parameters
    ----------
    dir : str
        Network path to directory in which the csv-files are stored.
    
    Returns
    -------
    network : pandapowerNet
        Returns a pandapowerNet network representation.
    """
    
    # create empty network
    network = pp.create_empty_network()
    
    # add buses
    network = add_buses(network, file=f'{dir}/buses.csv')
        
    # add lines
    network = add_aclines(network, file=f'{dir}/linesAC.csv')
    network = add_dclines(network, file=f'{dir}/linesDC.csv')

    # add trafos
    network = add_trafos(network, file=f'{dir}/trafos.csv')

    # add slack nodes
    network = add_slack(network, file=f'{dir}/slack.csv')

    return network

    
def build_network(dir, zones):
    """
    Build a PandaPower network based from csv files.
    
    Parameters
    ----------
    dir : str
        Network path to directory in which the csv-files are stored.
    zones : string or list
        List of exogenous zones for which the exchange mechanisms
        in this function are implemented.
    
    Returns
    -------
    network : pandapowerNet
        Returns a pandapowerNet network representation.
    """

    # create empty network
    network = pp.create_empty_network()
    
    # add buses
    network = add_buses(network, file=f'{dir}/buses.csv')
        
    # add lines
    network = add_aclines(network, file=f'{dir}/linesAC.csv')
    network = add_dclines(network, file=f'{dir}/linesDC.csv')

    # add trafos
    network = add_trafos(network, file=f'{dir}/trafos.csv')

    # add slack nodes
    network = add_slack(network, file=f'{dir}/slack.csv')

    # add power elements
    network = add_load(network, file=f'{dir}/loads.csv')
    network = add_sgen(network, file=f'{dir}/sgen.csv')
        
    # add exchange elements
    network = add_simple_exchange(network, zones=zones)
    
    return network
    

def check_arguments(df, file):
    """
    Helper function to validate argument specifications.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame with network characteristics.
    file : str
        Path to csv-file with network characteristics.
        
    Returns
    -------
    df : DataFrame
        Returns a DataFrame with network characteristics.
    """
    
    if (file is None) & (df is None):
        raise KeyError('Not enough arguments: specify either a file or a df')
        
    if (file is not None) & (df is not None):
        raise KeyError('Too many arguments: specify either a file or a df')
        
    if file is not None:
        df = pd.read_csv(file)
        
    return df
    
    
def add_buses(network, df=None, file=None):
    """
    Add buses to the specified network. Pass either a DataFrame or a pathstring to a csv-file.
    
    Parameters
    ----------
    network : pandapowerNet
        PandaPower network representation.
    df : DataFrame
        DataFrame with bus characteristics.
    file : str
        Path to csv-file with bus characteristics.
        
    Return
    ------
    net : pandapowerNet
        PandaPower network representation with added buses.
    """

    # copy network
    net = copy.deepcopy(network)
    
    # load buses
    buses = check_arguments(df, file)

    # add buses to network
    for index, bus in buses.iterrows():
        pp.create_bus(net, name=bus['name'], vn_kv=bus.vn_kv, type=bus.type, 
                      zone=bus.zone, in_service=bus.in_service, geodata=(bus.latitude, bus.longitude))
        
    return net


def add_aclines(network, df=None, file=None):
    """
    Add AC-lines to the specified network. Pass either a DataFrame or a pathstring to a csv-file.
    
    Parameters
    ----------
    network : pandapowerNet
        PandaPower network representation.
    df : DataFrame
        DataFrame with bus characteristics.
    file : str
        Path to csv-file with AC-line characteristics.
        
    Return
    ------
    net : pandapowerNet
        PandaPower network representation with added AC-lines.
    """
    
    # copy network
    net = copy.deepcopy(network)
    
    # load AC lines
    linesAC = check_arguments(df, file)
    
    # create dictonairy of busnames and busindices
    busdict = dict(zip(net.bus.name, net.bus.index))

    # parameter lines
    parameter_lines = linesAC[linesAC.std_type.isna() == True]

    for index, line in parameter_lines.iterrows():       
        pp.create_line_from_parameters(net, name=line['name'], from_bus=busdict[line.from_bus], 
                                       to_bus=busdict[line.to_bus], length_km=line.length_km, 
                                       r_ohm_per_km=line.r_ohm_per_km, x_ohm_per_km=line.x_ohm_per_km, 
                                       c_nf_per_km=line.c_nf_per_km, max_i_ka=line.max_i_ka,
                                       parallel=line.parallel, type=line.type, in_service=line.in_service,
                                       max_loading_percent=line.max_loading_percent)

    # default lines
    default_lines = linesAC[linesAC.std_type.isna() == False]

    for index, line in default_lines.iterrows():
        pp.create_line(net, name=line['name'], from_bus=busdict[line.from_bus], 
                       to_bus=busdict[line.to_bus], length_km=line.length_km, std_type=line.std_type,
                       parallel=line.parallel, in_service=line.in_service, 
                       max_loading_percent=line.max_loading_percent)
        
    return net

    
def add_dclines(network, df=None, file=None):
    """
    Add DC-lines to the specified network. Pass either a DataFrame or a pathstring to a csv-file.
    
    Parameters
    ----------
    network : pandapowerNet
        PandaPower network representation.
    df : DataFrame
        DataFrame with bus characteristics.
    file : str
        Path to csv-file with DC-line characteristics.
        
    Return
    ------
    net : pandapowerNet
        PandaPower network representation with added DC-lines.
    """
    
    # copy network
    net = copy.deepcopy(network)
    
    # load DC lines
    linesDC = check_arguments(df, file)

    # create dictonairy of busnames and busindices
    busdict = dict(zip(net.bus.name, net.bus.index))
    
    for index, line in linesDC.iterrows():
        pp.create_dcline(net, name=line['name'], from_bus=busdict[line.from_bus], 
                         to_bus=busdict[line.to_bus], p_mw=line.p_mw, loss_mw=line.loss_mw,
                         loss_percent=line.loss_percent, vm_from_pu=line.vm_from_pu, 
                         vm_to_pu=line.vm_to_pu, in_service=line.in_service)
        
    return net

        
def add_trafos(network, df=None, file=None):
    """
    Add transformers to the specified network. Pass either a DataFrame or a pathstring to a csv-file.
    
    Parameters
    ----------
    network : pandapowerNet
        PandaPower network representation.
    df : DataFrame
        DataFrame with bus characteristics.
    file : str
        Path to csv-file with transformer characteristics.
        
    Return
    ------
    net : pandapowerNet
        PandaPower network representation with added transformers.
    """

    # copy network
    net = copy.deepcopy(network)
    
    # load trafos
    trafos = check_arguments(df, file)

    # create dictonairy of busnames and busindices
    busdict = dict(zip(net.bus.name, net.bus.index))
    
    # parameter trafos
    parameter_trafos = trafos[trafos.std_type.isna() == True]

    for index, trafo in parameter_trafos.iterrows():       
        pp.create_transformer_from_parameters(net, name=trafo['name'], hv_bus=busdict[trafo.hv_bus],
                                              lv_bus=busdict[trafo.lv_bus], sn_mva=trafo.sn_mva,
                                              vn_hv_kv=trafo.vn_hv_kv, vn_lv_kv=trafo.vn_lv_kv, 
                                              vkr_percent=trafo.vkr_percent, vk_percent=trafo.vk_percent,
                                              pfe_kw=trafo.pfe_kw, i0_percent=trafo.i0_percent, 
                                              in_service=trafo.in_service, parallel=trafo.parallel)

    # default trafos
    default_trafos = trafos[trafos.std_type.isna() == False]

    for index, trafo in default_trafos.iterrows():
        pp.create_transformer(net, name=trafo['name'], hv_bus=busdict[trafo.hv_bus],
                              lv_bus=busdict[trafo.lv_bus], std_type=trafo.std_type, 
                              in_service=trafo.in_service, parallel=trafo.parallel)
        
    return net


def add_slack(network, df=None, file=None):
    """
    Add slacknodes to the specified network. Pass either a DataFrame or a pathstring to a csv-file.
    
    Parameters
    ----------
    network : pandapowerNet
        PandaPower network representation.
    df : DataFrame
        DataFrame with bus characteristics.
    file : str
        Path to csv-file with slacknode characteristics.
        
    Return
    ------
    net : pandapowerNet
        PandaPower network representation with added slacknodes.
    """
    
    # copy network
    net = copy.deepcopy(network)
    
    # load external grids
    slacks = check_arguments(df, file)

    # create dictonairy of busnames and busindices
    busdict = dict(zip(net.bus.name, net.bus.index))
    
    for index, slack in slacks.iterrows():
        pp.create_ext_grid(net, name=slack['name'], bus=busdict[slack.bus], vm_pu=slack.vm_pu, 
                           va_degree=slack.va_degree, in_service=slack.in_service)

    return net


def add_load(network, df=None, file=None):
    """
    Add load elements to the specified network. Pass either a DataFrame or a pathstring to a csv-file.
    
    Parameters
    ----------
    network : pandapowerNet
        PandaPower network representation.
    df : DataFrame
        DataFrame with bus characteristics.
    file : str
        Path to csv-file with load element characteristics.
        
    Return
    ------
    net : pandapowerNet
        PandaPower network representation with added load elements.
    """
    
    # copy network
    net = copy.deepcopy(network)
    
    # load loads
    loads = check_arguments(df, file)

    # create dictonairy of busnames and busindices
    busdict = dict(zip(net.bus.name, net.bus.index))
    
    for index, load in loads.iterrows():
        pp.create_load(net, name=load['name'], bus=busdict[load.bus], p_mw=0,
                       scaling=load.scaling, type=load.type, in_service=load.in_service)

    return net


def add_sgen(network, df=None, file=None):
    """
    Add static-generation elements to the specified network. Pass either a DataFrame or a pathstring to a csv-file.
    
    Parameters
    ----------
    network : pandapowerNet
        PandaPower network representation.
    df : DataFrame
        DataFrame with static-generation element characteristics.
    file : str
        Path to csv-file with static-generation element characteristics.
        
    Return
    ------
    net : pandapowerNet
        PandaPower network representation with added stratic-generation elements.
    """
    
    # copy network
    net = copy.deepcopy(network)
    
    # load static generators
    sgens = check_arguments(df, file)

    # create dictonairy of busnames and busindices
    busdict = dict(zip(net.bus.name, net.bus.index))

    for index, sgen in sgens.iterrows():
        pp.create_sgen(net, name=sgen['name'], bus=busdict[sgen.bus], p_mw=sgen.p_mw, 
                       scaling=sgen.scaling, type=sgen.type, in_service=sgen.in_service)

    return net


def add_simple_exchange(network, zones, ntcfactor=1.0, maxload=True):
    """
    Add exchange elements of exogenous zones to the network. 
    Exchange is based on the thermal capacity of each line multiplies 
    by the maximum loading percentage of each interconnected line. 
    
    Important
    ---------
    Make sure that all lines that are connected to an exogenous bus 
    originate from the same endogenous bus. This prevents the occorance
    of transit flows in exogenous regions.
    
    Parameters
    ----------
    network : pandapowerNet
        Pandapower network representation.
    zones : string or list
        List of exogenous zones for which the exchange mechanisms
        in this function are implemented.
    ntcfactor : float, default 1
        Fractional of evaluated capacity available for exchange.
    maxload : bool, default True
        Consider the maximum loading percentage of the AC-lines in
        the returned capacity.
        
    Return
    ------
    net : pandapowerNet
        Returns a pandapower network with exchange elements.
    """
    
    # copy network
    net = copy.deepcopy(network)
    
    # convert string to list
    if isinstance(zones, str):
        zones = [zones]
    
    # convert everything else to a list
    if not isinstance(zones, list):
        try:
            list(zones)
        except:
            raise TypeError('zones is not of type list')
    
    # get all buses that are exogenous to the model
    exchange = network.bus[network.bus.zone.isin(zones)]

    # iterate over each bus
    for index, bus in exchange.iterrows(): 

        # empty list
        zone = set()

        # iterate over ac and dc lines
        for line in [network.line, network.dcline]:

            # get all lines that are connected to the bus
            line = line[(line.from_bus == index) | (line.to_bus == index)]

            # get all buses to which the lines are connected
            for col in ['from_bus', 'to_bus']:

                # do not include the exogenous bus in this list
                ids = line[col][line[col] != index].unique().tolist()

                # extend the set
                zone.update(ids)

        # check set length
        if len(zone) != 1:
            # raise error if set contains multiple zones
            raise NotImplementedError(f'Exogenous bus "{index}" has incoming lines that are ', 
                                      'connected to multiple different stations')

        # get the bus index
        # get the zone of the bus
        zone = list(zone)[0]
        zone = network.bus.zone[zone]

        # specify a unique name id
        name = bus['name'] + '_' + zone

        # fetch capacity of each line connected to the bus
        p_mw = get_line_capacity_to_bus(network=network, bus=index, cf=ntcfactor, maxload=maxload)

        # create load element
        pp.create_load(net, name=name, bus=index, p_mw=p_mw,
                       scaling=1.0, type=zone, in_service=True)

        # create sgen element
        pp.create_sgen(net, name=name, bus=index, p_mw=p_mw, 
                       scaling=1.0, type=zone, in_service=True)
        
    return net

    
def assign_sgen_properties(network, properties, fuels, carbonprice=None, columns=None):
    """
    Add specific sgen properties to the sgen DataFrame
    in a network.
    
    Parameters
    ----------
    network : pandapowerNet
        PandaPower network representation.
    properties : DataFrame
        DataFrame that contains the basic properties
        of each type of sgen in the network.
    fuels : DataFrame
        DataFrame that contains the basic properties
        of each fuel type used by the sgen types.
    carbonprice : float, default None
        Carbon emission price per tonne to take into 
        account for each sgen type.
    columns : list, default None
        List of column names to map to the sgen
        DataFrame in the network.
        
    Return
    ------
    net : pandapowerNet
        Returns the network including a mapping
        of the sgen properties specified in columns.
    """
    
    # copy the passed network
    net = copy.deepcopy(network)
    
    # map the fuel properties on the sgen properties
    prop = properties.join(fuels.set_index('name'), on='fuel_type')

    # evaluate the fuel consumption per mwh, multiply by 100 to convert efficiency to fraction
    # fuel consumption multiplied by carbon density to evaluate carbon emissions per mwh
    # fuel consumption muliplied by fuel price
    prop['fuel_units_mwh'] = (3600 * 100) / prop.fuel_mj_unit / prop.efficiency_percent
    prop['co2_tonne_mwh'] = prop.fuel_units_mwh * prop.co2_tonne_unit
    prop['fuel_euro_mwh'] = prop.fuel_euro_unit * prop.fuel_units_mwh

    if carbonprice is None:
        # cost parameters without CO2 price included
        prop['marginal_euro_mwh'] = prop.opex_euro_mwh + prop.fuel_euro_mwh

    else:
        # cost parameters with CO2 price included
        prop['co2_euro_mwh'] = prop.co2_tonne_mwh * carbonprice
        prop['marginal_euro_mwh'] = (prop.opex_euro_mwh + prop.fuel_euro_mwh
                                    + prop.co2_euro_mwh)

    # set index to shared key in sgen
    prop = prop.set_index('name')
    
    if columns is None:
        # map df to sgen based on shared key
        net.sgen = net.sgen.join(prop, on='type')
    
    else:
        # map subsetted df to sgen based on shared key
        net.sgen = net.sgen.join(prop[columns], on='type')
    
    return net

def get_line_capacity(network, line, cf=1.0, maxload=False):
    """
    Evaluate the capacity of a pandapower AC-line based 
    on the thermal capacity of the line.
    
    Parameters
    ----------
    network : pandapowerNet
        Pandapower network representation.
    line : index
        Index of the evaluated AC-line.
    cf : float, default 1
        Fraction of line capacity available for transmission.
    maxload : bool, default False
        Consider the maximum loading percentage of the AC-line in
        the returned capacity.
        
    Return
    ------
    capacity : float
        Transmission capacity in MVA available in the evaluated line.
    """
    
    # check if line is in network
    if line not in network.line.index:
        raise KeyError('line-index not in network')

    # get voltage level from the connected bus
    bus = network.line.iloc[line].from_bus
    vn_kv = network.bus.vn_kv[bus]

    # calculate thermal capacity based on line properties
    thermal = vn_kv * network.line.max_i_ka[line] * math.sqrt(3)
    
    if maxload:
        # consider maximum loading percentage in capacity calculation
        loading = network.line.max_loading_percent[line] * (1/100)
        cf = loading * cf
    
    # evalaute line capacity
    capacity = thermal * cf

    return capacity


def get_line_capacity_to_bus(network, bus, cf=1.0, maxload=False):
    """
    Calculate the load that can be transported to the selected bus.
    Evaluates the capacity based on the thermal capacity of each 
    line that is connected to the specified bus.

    Parameters
    ----------
    network : pandapowerNet
        Pandapower network representation.
    bus : index
        Index of the evaluated network bus.
    cf : float, default 1
        Fraction of line capacity available for transmission.
    maxload : bool, default False
        Consider the maximum loading percentage of the AC-lines in
        the returned capacity.
        
    Return
    ------
    capacity : float
        Transmission capacity in MVA available to the evaluted bus.
    """
    
    #  check if bus is in network
    if bus not in network.bus.index:
        raise KeyError('bus-index not in network')

    # subset AC-lines that are connected to the specified bus
    lines = network.line.index[(network.line.to_bus == bus) | (network.line.from_bus == bus)]
    
    
    if maxload:
        # consider maximum line loading in results
        # calculate capacity of each AC-line and sum the result
        AC = sum([get_line_capacity(network=network, line=index, cf=cf, maxload=True) for index in lines])
    else:
        # don't consider maximum line loading
        # calculate capacity of each AC-line and sum the result
        AC = sum([get_line_capacity(network=network, line=index, cf=cf, maxload=False) for index in lines])
        
    # subset DC-lines that are connected to the specified bus and retrieve capacity
    DC = network.dcline.p_mw[(network.dcline.to_bus == bus) | (network.dcline.from_bus == bus)].sum() * cf

    # calculate total capacity
    capacity = AC + DC
    
    return capacity