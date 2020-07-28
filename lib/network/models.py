import os
import time
import copy

import pandas as pd
import pandapower as pp

class NetworkModel:
    """Evaluates the loadflow of a network model.
    
    This 'base' model evaluates the loadflows of a network model 
    through the regionalization of electricity curves that are passed
    by the specified client."""
    
    def __init__(self, network, runs, client, scenario=None, ETMdir=None, *args, **kwargs):
        """Initialization procedure of the class parameters.
        
        To do:
         - Generalize ETMdir reference
         - Optional categorization
         - Clearify regionalization (not always multiple files)
        
        Parameters
        ----------
        network : pandapowerNet
            Pandapower network representation of the evaluated network.
        runs : int or list
            list of the 'hours' that are evaluated. An integer is 
            intepreted as the upper bound of a ranged list.
        client : object
            callable object that returns the curves that are evaluated
            in the loadflow calculations.
        scenario : dict, default None
            scenario that specifies the base conditions under which
            the network is evaluated.
        ETMdir : str, default None
            string with the path to the directory that specifies the
            categorization and regionalization of the curves that 
            are passed by the client.
            
        The passed args and kwargs are passed to the pd.read_csv().
        
        Returns
        -------
        results : dict
            Returns a dictonairy with the results of the loadflow 
            calculations."""
        
        # set params
        self.network = network
        self.client = client
        self.scenario = scenario
        
        if isinstance(runs, int):
            runs = [x for x in range(runs)]

        if not isinstance(runs, list):
            raise TypeError('runs must be a list or integer')
            
        self.runs = runs
        
        # set ETMdir
        if ETMdir is None:
            self.ETMdir = 'data/ETM/'
        else:
            self.ETMdir = ETMdir
        
        # set categorization df
        self.parm_cat = pd.read_csv(self.ETMdir + 'parameter_categorization.csv', *args, **kwargs)
        self.parm_cat = self.parm_cat.set_index('etm_key')
        
        # set regionalization dfs
        self.sgen_regs = self._get_csv_files_in_folder(self.ETMdir + '/sgen_regionalization', *args, **kwargs)
        self.load_regs = self._get_csv_files_in_folder(self.ETMdir + '/load_regionalization', *args, **kwargs)
        
    def __enter__(self):
        return self
    
    def __exit__(self, *args, **kwargs):
        return None
    
    def _get_csv_files_in_folder(self, folder, *args, **kwargs):
        """Helper function to load multiple csv-files from a specified folder.
        
        To do:
         - Index_col = 0 is somehow default here?
        
        Parameters
        ----------
        folder : str
            string with the path to the folder from which to load
            the .csv files.
        
        The passed args and kwargs are passed to the pd.read_csv().
        
        Returns
        -------
        items : list
            Returns a list with the dataframes of the loaded 
            .csv-files in the specified folder."""
        
        items = []
        
        for file in os.listdir(folder):
            if file.endswith('.csv'):
                df = pd.read_csv(f'{folder}/{file}', index_col=0, *args, **kwargs)
                items.append(df)
            
        if len(items) == 0:
            raise ValueError(f'There appear to be no files in "{folder}"')
            
        return items
            
    
#     def construct_network(self, networkdir):
#         """construct network based on regionalization"""
        
#         # reference reg df
#         sgen = self.sgen_regs[0]
#         load = self.load_regs[0]
        
#         # build base network
#         network = build_base_network(networkdir)

#         # create dictonairy of busnames and busindices
#         busdict = dict(zip(network.bus.name, network.bus.index))

#         # add sgen elements
#         for item in sgen.columns:
#             for bus in sgen.index:
#                 if not pd.isna(sgen.loc[bus, item]):
#                     pp.create_sgen(network, name=bus + '_' + item, bus=busdict[bus], 
#                                    p_mw=0, scaling=1, type=item, in_service=True)

#         # add load elements
#         for item in load.columns:
#             for bus in load.index:
#                 if not pd.isna(load.loc[bus, item]):
#                     pp.create_load(network, name=bus + '_' + item, bus=busdict[bus], 
#                                    p_mw=0, scaling=1, type=item, in_service=True)

#         return network
        
        
    def __call__(self, **kwargs):
        """Call procedure of the class that is EMA-workbench compatible.
        
        Returns
        -------
        results : dict
            Returns a dictonairy of the results specified under the store
            results function of this class."""        
        
        # init run results
        self.run_results = None
        
        # !!! DEFAULT TO NONE AFTER TESTING !!! #
        #      - should be in scenario, if not passed as kwarg.
        
        # extract params from kwargs
        sgen_reg = kwargs.get('sgen_reg', 0)
        load_reg = kwargs.get('load_reg', 0)
        
        # fetch relevant reg dfs
        self.sgen_reg = self.sgen_regs[sgen_reg]
        self.load_reg = self.load_regs[load_reg]
        
        # call client for curves
        self.curves = self.client_procedure(**kwargs)
        
        # translate ETMkeys to CustomKeys
        self.sgen, self.load = self.categorize_curves(self.curves)

        # evaluate each run
        list(map(lambda run: self.evaluate_run(run), self.runs))
        
        # evaluate results
        results = self.evaluate_results()
        
        return results
    
    def evaluate_results(self):
        """Procedure to evaluates the results that are returned
        during the call procedure.
        
        Returns
        -------
        results : dict
            Returns a dictonairy with results that can be accessed 
            by the EMA-workbench."""
        
        # initialize result dict
        results = {}
        
        # calculate loading percentage as fraction of maximum line loading
        loading_percent = self.run_results
        max_loading_percent = self.network.line.max_loading_percent
        results['loading_percent'] = loading_percent.div(max_loading_percent, axis=0)
            
        # calculate overload score as dimensionless unit
        overload_score = loading_percent.where(loading_percent - 1 > 0, 1) - 1
        results['overload_score'] = overload_score.round(3)
        
        return results
        
    def client_procedure(self, **kwargs):
        """Procedure to fetch the hourly curves 
        from the connected client.
        
        The connected client must be compatible with the
        procedure that is specified in this function.
        
        Returns
        -------
        curves : DataFrame
            Returns the hourly electricity curves that are
            used to evaluate the network."""
        
        # make sure default scenario cannot be changed.
        scenario = copy.deepcopy(self.scenario)
        
        # check if scenario was passed
        if scenario is None:
            curves = self.client(**kwargs)
        else:
            curves = self.client(scenario=scenario, **kwargs)
            
        return curves
    
    def categorize_curves(self, curves):
        """Aggregate ETM parameters into model parameters.
        
        Parameters
        ----------
        curves : DataFrame
            DataFrame that contains the hourly electricity
            curves from the ETM."""
                
        # ignore time column
        curves = curves.iloc[:, 1:]
        
        # transpose curves and reset index
        curves = curves.T.reset_index()
        curves = curves.rename(columns={'index': 'key'})
        
        # map 'ppdf' and 'key' from lookup table to df
        curves['ppdf'] = curves.key.map(self.parm_cat.network_ppdf)
        curves['key'] = curves.key.map(self.parm_cat.network_key)
        
        # group curves based on ppdf category
        sgen = curves[curves.ppdf == 'sgen'].groupby('key').sum().T
        load = curves[curves.ppdf == 'load'].groupby('key').sum().T
        
        return sgen, load

    def evaluate_run(self, run):
        """Procedure to evaluate a single run (hour).
        
        Parameters
        ----------
        run : int
            number of the evaluated run"""
                        
        # copy network
        network = copy.deepcopy(self.network)
        
        # fetch run data
        sgen_run, load_run = self.regionalize_curves(network, run)

        # update network dataframe
        network.sgen.p_mw.update(sgen_run)
        network.load.p_mw.update(load_run)
        
        # set dcline direction
        network = self.direct_dclines(network, run)
        
        # evaluate loadflow
        pp.rundcpp(network)
                
        # evaluate the results of the run
        self.evaluate_run_result(network, run)
                       
    def regionalize_curves(self, network, run):
        """Procedure to regionalize the data for the passed
        hour to the power elements in the network.
        
        Parameters
        ----------
        run : int
            number of the evaluated run.
        network : pandapowerNet
            PandaPower network representation.
            
        Returns
        -------
        sgen_run, load_run : Series
            Returns the regionalized sgen and load Series
            for the passed run."""
        
        # get run data
        sgen_run = self.sgen.iloc[run]
        load_run = self.load.iloc[run]

        # multiply run data with regionalization
        sgen_run = self.sgen_reg.mul(sgen_run)
        load_run = self.load_reg.mul(load_run)
        
        # reset index to store bus info
        sgen_run = sgen_run.reset_index()
        load_run = load_run.reset_index()
        
        # melt dataframes to get respective ppdfs
        sgen_run = sgen_run.melt(id_vars='bus', var_name='type', value_name='p_mw')
        load_run = load_run.melt(id_vars='bus', var_name='type', value_name='p_mw')
        
        # replace index with network ppdf element names
        sgen_run.index = sgen_run.bus + '_' + sgen_run.type
        load_run.index = load_run.bus + '_' + load_run.type
        
        # map p_mw data to ppdf in network
        sgen_run = network.sgen['name'].map(sgen_run.p_mw)
        load_run = network.load['name'].map(load_run.p_mw)
        
        return sgen_run, load_run
        
    def direct_dclines(self, network, run):
        """Procedure to change the direction of DC-lines when
        the relative exchange position of a DC line changes.
        
        DC-lines in PandaPower have a prespecified direction. 
        To evaluate a dynamicly set network, the relative 
        exchange position of the lines has to be considered and
        changed approriatly to ensure the network remains balanced.
        
        Parameters
        ----------
        network : pandapowerNet
            PandaPower network representation.
        run : int
            number of the evaluated run.
            
        Returns
        -------
        network : pandapowerNet
            PandaPower network representation."""
        
        # !!! HARDCODED !!! #
        # specify exchange zones
        zones = ['BE00', 'DE00', 'DKW1', 'NOS0', 'UK00']
        
        # get relevant indices
        imprt = [f'electricity_import_{zone}' for zone in zones]
        exprt = [f'electricity_export_{zone}' for zone in zones]

        # get run data
        sgen_run = self.sgen.iloc[run]
        load_run = self.load.iloc[run]
        
        # subset exchange data
        imprt = sgen_run[sgen_run.index.isin(imprt)]
        exprt = load_run[load_run.index.isin(exprt)]
        
        # !!! HARDCODED !!! #
        # change index to desired format
        imprt.index = imprt.index.str.replace('electricity_import', 'NL00')
        exprt.index = exprt.index.str.replace('electricity_export', 'NL00')
        
        # make imprt values negative
        # subset all negative values
        imprt *= -1
        imprt = imprt[imprt < 0] 

        # merge import and export information
        exchange = exprt.copy(deep=True)
        exchange.update(imprt)
        
        # specify helper variables
        columns = {'from_bus': 'to_bus', 'to_bus': 'from_bus'}
        dtypes = {'from_bus': int, 'to_bus': int}

        # iterate over each exchange item 
        for index, item in exchange.items():

            # get from zone and to zone
            fzone, tzone = index.split('_')

            # specify import condition under which buses must be switched
            c1 = ((network.dcline.from_bus.map(network.bus.zone) == fzone) 
                  & (network.dcline.to_bus.map(network.bus.zone) == tzone))

            # specify export condition under which buses must be switched
            c2 = ((network.dcline.from_bus.map(network.bus.zone) == tzone) 
                  & (network.dcline.to_bus.map(network.bus.zone) == fzone))

            # in case of relative import 
            if item < 0:
                # subset misdirected dclines
                dclines = network.dcline[c1]

            else:
                # subset misdirected dclines
                dclines = network.dcline[c2]

            # rename columns that have to be switched
            dclines = dclines.rename(columns=columns)

            # update network
            network.dcline.update(dclines)

            # subset from corrected df
            dclines = network.dcline[c1 | c2]
            
            # apply dispatch to lines
            dclines.p_mw = (dclines.p_mw / dclines.p_mw.sum()) * abs(item)

            # update the df again
            network.dcline.update(dclines)

            # change dtypes back to int after updates
            network.dcline = network.dcline.astype(dtypes)
            
        return network
    
    def evaluate_run_result(self, network, run):
        """Procedure that stores the results of each run.
        
        The results stored in self.run_results can be accessed in the 
        evaluate_results procedure.
        
        Parameters
        ----------
        network : pandapowerNet
            evaluated PandaPower network representation.
        run : int
            number of the evaluated run."""
        
        # init run results dict
        if self.run_results is None:
            self.run_results = pd.DataFrame(columns=self.runs)
        
        # store result of line loading percentages
        self.run_results[run] = network.res_line.loading_percent