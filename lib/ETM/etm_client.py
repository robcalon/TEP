"""Async interaction with ETM.
This code is an async adaptation of the ETM_API 
published on https://github.com/quintel/third-party."""

import io
import json
import warnings

import aiohttp
import asyncio
import nest_asyncio

import pandas as pd


class ETMSession:
    def __init__(self, beta_engine=False):        
        """init ETMSession"""
        
        # set beta argument
        self.beta_engine = beta_engine
        
        # set base url
        if self.beta_engine is False:
            # connect to production engine
            self._base_url = 'https://engine.energytransitionmodel.com/api/v3'
        else:
            # connect to beta engine
            self._base_url = 'https://beta-engine.energytransitionmodel.com/api/v3'
        
        try:
            # check for available loop
            asyncio.get_event_loop()
                
        except RuntimeError:
            # create new event loop
            event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(event_loop)
            
        return self
    
    def __repr__(self):
        """repr. string"""
        
        return f'ETMSession({self.beta_engine})'
        
    def _make_url(self, path):
        """make url"""

        return self._base_url + path
    
    async def _handle_status_error(self, resp):
        """helper to check returned status and show error message"""
        
        # get error message
        error = await resp.json()
        error = error.get('errors')

        # help to understand ETM error message
        print(f'ClientResponseError: {resp.status} - {resp.reason}')
        
        nr = len(str(resp.status)) + len(resp.reason) + 24
        print(nr * '-')
        
        print('ErrorMessage(s):')
        for message in error:
            print(f'--> {message}')
        print()

        # raise status
        resp.raise_for_status()
        
    async def get(self, post, *args, **kwargs):
        """simple aiohttp.ClientSession.get() wrapper"""
        
        # construct url
        url = self._make_url(post)
        
        # get url response and transform response into text
        async with aiohttp.ClientSession() as session:    
            async with session.get(url, *args, **kwargs) as resp:

                # check for error
                if resp.status >= 400:
                    await self._handle_status_error(resp)
                else:
                    return await resp.text(encoding='utf-8')
    
    async def put(self, post, *args, **kwargs):
        """simple aiohttp.ClientSession.put() wrapper"""
        
        # construct url
        url = self._make_url(post)
        
        # put json at url and transform response into json
        async with aiohttp.ClientSession() as session:    
            async with session.put(url, *args, **kwargs) as resp:

                # check for error
                if resp.status >= 400:
                    await self._handle_status_error(resp)
                else:
                    return await resp.json(encoding='utf-8')
            
    async def post(self, post, *args, **kwargs):
        """simple aiohttp.ClientSession.post() wrapper"""

        # construct url
        url = self._make_url(post)
        
        # post json at url and tranform response into json
        async with aiohttp.ClientSession() as session:
            async with session.post(url, *args, **kwargs) as resp:

                # check for error
                if resp.status >= 400:
                    await self._handle_status_error(resp)
                else:
                    return await resp.json(encoding='utf-8')
        
    async def delete(self, post, *args, **kwargs):
        """simple aiohttp.ClientSession.delete() wrapper"""
        
        # construct url
        url = self._make_url(post)
        
        # delete request at url
        async with aiohttp.ClientSession() as session:
            async with session.delete(url, *args, **kwargs) as resp:

                # check for error
                if resp.status >= 400:
                    await self._handle_status_error(resp)
                else:
                    return await resp.json(encoding='utf-8')
        

# TD: gqueries could be used better to get units of each parameter, see api documentation.
        
class ETMClient(ETMSession):
    """simple client to interact with ETM through public API"""
    def __init__(self, scenario_id=None, beta_engine=False):
        """
        """
        # init ETMSession
        super().__init__(beta_engine=beta_engine)
        
        # set passed parameters
        self.scenario_id = scenario_id
    
    def __repr__(self):
        """reproduction string"""
        return f'ETMClient({self.scenario_id}, {beta_engine})'
    
    def __str__(self):
        """simple string"""
        return f'ETMClient[{self.scenario_id}]'
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args, **kwargs):
        return None
    
    def _create_new_scenario(self, scenario):
        """
        Create a new scenario in the ETM. The scenario_id is saved so we can
        continue from the new scenario later on.
        """
        required = ['title', 'area_code','end_year', 'description', 'protected']
        
        for key in required:
            if key not in scenario.keys():
                raise KeyError(f'"{key}" is not specified in the passed scenario')
        
        data = {"scenario": scenario}
        headers = {'Connection':'close'}
        
        process = self.post('/scenarios', data=data, headers=headers)
        resp = asyncio.run(process)
        
        resp = json.loads(resp)
        self.scenario_id = str(resp['scenario_id'])
        
        warnings.warn(f'"{str(self)}" scenario_id changed to "{self.scenario_id}"')
        
    def return_gqueries(self, json):
        """
        Extracts information from text p by first converting to a pandas dataframe.
        """
        p_gqueries = json["gqueries"]
        df = pd.DataFrame.from_dict(p_gqueries, orient="index")
        
        return df

    def get_energy_flows(self):
        """
        """
        headers = {'Connection':'close'}
        post = f'/scenarios/{self.scenario_id}/energy_flow'
        
        process = self.get(post, headers=headers)
        resp = asyncio.run(process)
        
        return pd.read_csv(io.StringIO(resp))

    def get_application_demands(self):
        """
        """
        headers = {'Connection':'close'}
        post = f'/scenarios/{self.scenario_id}/application_demands'

        process = self.get(post, headers=headers)
        resp = asyncio.run(process)
    
        return pd.read_csv(io.StringIO(resp))

    def get_production_parameters(self):
        """
        """
        headers = {'Connection':'close'}
        post = f'/scenarios/{self.scenario_id}/production_parameters'
        
        process = self.get(post, headers=headers)
        resp = asyncio.run(process)
        
        return pd.read_csv(io.StringIO(resp))

    def get_hourly_electricity_curves(self):
        """
        """
        headers = {'Connection':'close'}
        post = f'/scenarios/{self.scenario_id}/curves/merit_order'
        
        process = self.get(post, headers=headers)
        resp = asyncio.run(process)
        
        return pd.read_csv(io.StringIO(resp))
    
    def get_hourly_electricity_price_curve(self):
        """
        """
        headers = {'Connection':'close'}
        post = f'/scenarios/{self.scenario_id}/curves/electricity_price'
        
        process = self.get(post, headers=headers)
        resp = asyncio.run(process)
        
        return pd.read_csv(io.StringIO(resp))
    
    def get_hourly_household_heat_curves(self):
        """
        """
        headers = {'Connection':'close'}
        post = f'/scenarios/{self.scenario_id}/curves/household_heat'

        process = self.get(post, headers=headers)
        resp = asyncio.run(process)
        
        return pd.read_csv(io.StringIO(resp))

    def get_hourly_gas_curves(self):
        """
        Get data export for hourly network gas curves
        """
        headers = {'Connection':'close'}
        post = f'/scenarios/{self.scenario_id}curves/network_gas'
        
        process = self.get(post, headers=headers)
        resp = asyncio.run(process)
        
        return pd.read_csv(io.StringIO(resp))
    
    def get_hourly_hydrogen_curves(self):
        """
        Get data export for hourly hydrogen curves
        """
        headers = {'Connection':'close'}
        post = f'/scenarios/{self.scenario_id}/curves/hydrogen'
        
        process = self.get(post, headers=headers)
        resp = asyncio.run(process)
        
        return pd.read_csv(io.StringIO(resp))

    def get_hourly_heat_network_curves(self):
        """
        Get data export for hourly heat network curves
        """
        headers = {'Connection':'close'}
        post = f'/scenarios/{self.scenario_id}/curves/heat_network'
        
        process = self.get(post, headers=headers)
        resp = asyncio.run(process)
        
        return pd.read_csv(io.StringIO(resp))
    
    def get_scenario_templates(self):
        """
        Get dataframe of available templates within the ETM. From this, a
        scenario id can be extracted. Note that not all scenario ids seem
        to work.
        """
        headers = {'Connection':'close'}

        process = self.get(post, headers=headers)
        resp = asyncio.run(process)
        
        templates = json.loads(resp)
        
        return pd.DataFrame.from_dict(templates)
    
    def reset_scenario(self):
        """
        Resets scenario with scenario_id
        """
        data = {"reset": True}

        headers = {'Connection':'close'}
        post = f'/scenarios/{self.scenario_id}'
        
        process = self.put(post, json=data, headers=headers)
        resp = asyncio.run(process)
        
        self.current_metrics = self.return_gqueries(resp)
    
    def get_inputs(self):
        """
        Get list of available inputs. Can be used to search parameter space?
        """
        headers = {'Connection':'close'}
        post = f'/scenarios/{self.scenario_id}/inputs'
        
        process = self.get(post, headers=headers)
        resp = asyncio.run(process)

        inputs = json.loads(resp)
        df = pd.DataFrame.from_dict(inputs, orient='index')
        
        df = df.reset_index()
        df = df.rename(columns={'index': 'key'})
        
        return df
    
    def get_flexibility_order(self):
        """
        Get flexbility order of scenario.
        """
        headers = {'Connection':'close'}
        post = f'/scenarios/{self.scenario_id}/flexibility_order'
        
        process = self.get(post, headers=headers)
        resp = asyncio.run(process)
        
        return json.loads(resp)
    
    def get_heat_network_order(self):
        """
        Get heat network order of scenario.
        """
        headers = {'Connection':'close'}
        post = f'/scenarios/{self.scenario_id}/heat_network_order'
        
        process = self.get(post, headers=headers)
        resp = asyncio.run(process)
        
        return json.loads(resp)
    
    def get_current_metrics(self, gquery_metrics):
        """
        Perform a gquery on the the ETM model. gquery_metrics is a list of
        available ggueries.
        """
        data = {"detailed": True, "gqueries": gquery_metrics}

        headers = {'Connection':'close'}
        post = f'/scenarios/{self.scenario_id}'
        
        process = self.put(post, json=data, headers=headers)
        resp = asyncio.run(process)
        
        return self.return_gqueries(resp)
    
    def get_custom_curves(self):
        """
        Get all custom curves within the scenario
        """
        headers = {'Connection': 'close'}
        post = f'/scenarios/{self.scenario_id}/custom_curves'
        
        process = self.get(post, headers=headers)
        resp = asyncio.run(process)
        
        return json.loads(resp)
    
    def get_custom_curve(self, curve_type):
        """
        Get specific custom curve data
        """
        headers = {'Connection': 'close'}
        post = f'/scenarios/{self.scenario_id}/custom_curves/{curve_type}'
        
        process = self.get(post, headers=headers)
        resp = asyncio.run(process)
        
        return json.loads(resp)
    
    def change_inputs(self, user_values):
        """
        Change inputs to ETM according to dictionary user_values. Also the
        metrics are updated by passing a gquery via gquery_metrics
        """
        data = {"scenario": {"user_values": user_values}, "detailed": True}
        
        headers = {'Connection':'close'}
        post = f'/scenarios/{self.scenario_id}'
        
        process = self.put(post, json=data, headers=headers)
        asyncio.run(process)
            
    def change_flexibility_order(self, flexibility_order):
        """
        Change flexibility order to ETM according to object flexibility_order.
        """
        data = {'flexibility_order': flexibility_order}
        
        headers = {'Connection':'close'}
        post = f'/scenarios/{self.scenario_id}/flexibility_order'
        
        process = self.put(post, json=data, headers=headers)
        asyncio.run(process)
            
    def change_heat_network_order(self, heat_network_order):
        """
        Change heat network order to ETM according to object heat_network_order.
        """         
        data = {'heat_network_order': heat_network_order}
        
        headers = {'Connection':'close'}
        post = f'/scenarios/{self.scenario_id}/heat_network_order'
        
        process = self.put(post, json=data, headers=headers)
        asyncio.run(process)
                      
    def change_scenario_parameters(self, scenario):
        """
        """
        # update slider user values
        if scenario.get('user_values') is not None:
            self.change_inputs(scenario['user_values'])

        # update flexibility order
        if scenario.get('flexibility_order') is not None:
            self.change_flexibility_order(scenario['flexibility_order'])

        # update heat network order
        if scenario.get('heat_network_order') is not None:
            self.change_heat_network_order(scenario['heat_network_order'])
            
        # upload custom price curves
        if scenario.get('price_curves') is not None:
            self.upload_custom_price_curves(scenario['price_curves'])
            
    async def _upload_custom_price_curve(self, curve_type, price_data):
        """
        Upload a custom curve
        """        
        
        # get df properties
        filename = curve_type + '.csv'
        price_data = price_data.to_string(index=False)

        # convert to form-data
        data = aiohttp.FormData()
        data.add_field('file', price_data, filename=filename)
            
        headers = {'Connection': 'close'}
        post = f'/scenarios/{self.scenario_id}/custom_curves/{curve_type}'
        
        await self.put(post, data=data, headers=headers) 
              
    async def _upload_custom_price_curves(self, price_curves):
        """
        Upload multiple custom curves at once when all curves are bundled in a single df
        """
        
        if not isinstance(price_curves, pd.DataFrame):
            raise TypeError('must be pandas dataframe')
           
        # only accept if column names are correct
        accepted = [f'interconnector_{nr + 1}_price' for nr in range(6)]
        
        # check columns in dataframe
        if not price_curves.columns.isin(accepted).all():
                raise KeyError(f'price curves contains a columsn name that is not accepted')

        tasks = []
        
        for col in price_curves.columns:
            work = self._upload_custom_price_curve(col, price_curves[col])
            task = asyncio.create_task(work)
            tasks.append(task)
            
        await asyncio.gather(*tasks)
        
    def upload_custom_price_curves(self, price_curves):
        """info"""
        
        process = self._upload_custom_price_curves(price_curves)
        asyncio.run(process)
        
    def delete_custom_price_curve(self, curve_type):
        """
        Delete a custom curve
        """
        headers = {'Connection': 'close'}
        post = f'/scenarios/{self.scenario_id}/custom_curves/{curve_type}'
        
        process = self.delete(post, headers=headers)
        asyncio.run(process)
                        
    def create_scenario_copy(self, scenario):
        """
        post a scenario that is coppied.
        """                
        if 'scenario_id' not in scenario.keys():
            raise KeyError('"scenario_id" must be passed when copying a scenario.')
    
        prohibited = ['user_values', 'heat_network_order', 'flexibility_order']
        
        for key in prohibited:
            if key in scenario.keys():
                raise KeyError(f'"{key}" cannot be passed when copying a scenario.')
        
        self._create_new_scenario(scenario)
        
    def create_custom_scenario(scenario):
        """
        create new scenario
        """
        if 'scenario_id' in scenario.keys():
            raise KeyError('"scenario_id" cannot be in the passed scenario.')
        
        self._create_new_scenario(scenario)
        
    def get_user_values(self):
        """
        """        
        # get user values
        user_values = self.get_inputs()

        try:
            # get all user specified values
            user_values = user_values[~user_values.user.isna()]

            # reset index
            user_values = user_values.reset_index(drop=True)

            return user_values
        
        except:
            raise KeyError('No user_values have been set')
            
    def get_scenario_parameters(self, **kwargs):
        """
        dict of scenario parameters. contains only changed user values,
        the (default) flexibility order and the (default) heat network order.
        """
        # create scenario
        scenario = kwargs
        
        try:
            # convert user values to dict
            user_values = self.get_user_values()
            user_values = dict(zip(user_values.key, user_values.user))
        
        except:
            user_values = {}
        
        # set fetchable parameters
        scenario['user_values'] = user_values
        
        # be aware this might change! 
        # https://github.com/quintel/etengine/issues/1109
        scenario['flexibility_order'] = self.get_flexibility_order()
        scenario['heat_network_order'] = self.get_heat_network_order()
        
        # TD: Get each custom parameters as well in some sort of format?
        # fetch dict with all and then add the parameters individualy in an overwrite?

        # update scenario with kwargs
        scenario.update(kwargs)
        
        return scenario