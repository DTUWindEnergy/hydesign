# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 13:08:01 2024

@author: mikf
"""
import openmdao.api as om


class battery_with_reliability(om.ExplicitComponent):
    """
    """
    def __init__(
        self, 
        life_h = 25*365*24,
        reliability_ts_battery=None,
        reliability_ts_trans=None,
        ):
        """
        """ 
        super().__init__()
        self.life_h = life_h
        self.reliability_ts_battery = reliability_ts_battery
        self.reliability_ts_trans = reliability_ts_trans
        
    def setup(self):
        self.add_input(
            'b_t',
            desc="Battery charge/discharge power time series w/o reliability",
            units='MW',
            shape=[self.life_h])

        self.add_output(
            'b_t_rel',
            desc="Battery charge/discharge power time series with reliability",
            units='MW',
            shape=[self.life_h])

    def compute(self, inputs, outputs):
        if ((self.reliability_ts_battery is None) or (self.reliability_ts_trans is None)):
            outputs['b_t_rel'] = inputs['b_t']
            return
        outputs['b_t_rel'] = inputs['b_t'] * self.reliability_ts_battery[:self.life_h] * self.reliability_ts_trans[:self.life_h]
    

class wpp_with_reliability(om.ExplicitComponent):
    """
    """
    def __init__(
        self, 
        life_h = 25*365*24,
        reliability_ts_wind=None,
        reliability_ts_trans=None,
        ):
        """
        """ 
        super().__init__()
        self.life_h = life_h
        self.reliability_ts_wind = reliability_ts_wind
        self.reliability_ts_trans = reliability_ts_trans
        
    def setup(self):
        # self.add_input(
        #     'Nwt',
        #     val=1,
        #     desc="Number of wind turbines")
        self.add_input(
            'wind_t',
            desc="WPP power time series w/o reliability",
            units='MW',
            shape=[self.life_h])
        self.add_output(
            'wind_t_rel',
            desc="WPP power time series with reliability",
            units='MW',
            shape=[self.life_h])

    def compute(self, inputs, outputs):
        if ((self.reliability_ts_wind is None) or (self.reliability_ts_trans is None)):
            outputs['wind_t_rel'] = inputs['wind_t']
            return
        outputs['wind_t_rel'] = inputs['wind_t'] * self.reliability_ts_wind[:self.life_h] * self.reliability_ts_trans[:self.life_h]
    

class pvp_with_reliability(om.ExplicitComponent):
    """
    """
    def __init__(
        self, 
        life_h = 25*365*24,
        reliability_ts_pv=None,
        reliability_ts_trans=None,
        ):
        """
        """ 
        super().__init__()
        self.life_h = life_h
        self.reliability_ts_pv = reliability_ts_pv
        self.reliability_ts_trans = reliability_ts_trans
        
    def setup(self):
        # self.add_input(
        #     'solar_MW',
        #     val=1,
        #     desc="Solar PV plant installed capacity",
        #     units='MW')
        self.add_input(
            'solar_t',
            desc="PVP power time series w/o reliability",
            units='MW',
            shape=[self.life_h])
        self.add_output(
            'solar_t_rel',
            desc="PVP power time series with reliability",
            units='MW',
            shape=[self.life_h])

    def compute(self, inputs, outputs):
        if ((self.reliability_ts_pv is None) or (self.reliability_ts_trans is None)):
            outputs['solar_t_rel'] = inputs['solar_t']
            return
        outputs['solar_t_rel'] = inputs['solar_t'] * self.reliability_ts_pv[:self.life_h] * self.reliability_ts_trans[:self.life_h]
    

