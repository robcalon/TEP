import copy

from ..ETM.etm_client import ETMClient
    
class CallableETMClient(ETMClient):
    """update client procedure of Networkmodel to fit client"""
    
    def __call__(self, scenario, **kwargs):
        """call procedure"""
        
        # assign all kwargs as user values
        user_values = {'user_values': kwargs}

        # copy and update scenario
        scenario = copy.deepcopy(scenario)
        scenario.update(user_values)
        
        # reset ETM scenario
        self.reset_scenario()
        
        # change scenario parameters
        self.change_scenario_parameters(scenario)
        
        # get hourly results
        curves = self.get_hourly_electricity_curves() 
            
        return curves
    

class CSVClient:
    """Autoload .csv file with curves from a directory.
    
    When a year is passed to the scenario in the NetworkModel,"""
    
    def __init__(self, dir, year, *args, **kwargs):
        """specify dir from which to load the files"""
        
        self.dir = dir
        self.year = year
        self._args = args
        self._kwargs = kwargs
        
    def __call__(self, scenario, **kwargs):
        """call to get curves"""
        
        # copy scenario
        scenario = copy.deepcopy(scenario)
        scenario.update(kwargs)
        
        # get year from scenario
        year = scenario.get('year', None)
        
        if year is None:
            raise KeyError('"year" is not present in the model scenario.')
        
        # load time series
        curves = self.load_time_series(year)

        return curves
        
    def load_time_series(self, year):

        # reference (kw)args
        args = self._args
        kwargs = self._kwargs
        
        for file in os.listdir(self.dir):
            if file.endswith(f'{year}.csv'):
                filepath = self.dir + '/' + file
                curves = pd.read_csv(filepath, *args, **kwargs)
        
        return curves