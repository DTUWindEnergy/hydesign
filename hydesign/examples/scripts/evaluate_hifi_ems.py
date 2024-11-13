# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:20:33 2024

@author: mikf
"""

def main():
    if __name__ == '__main__':
        import os
        
        from hydesign.examples import examples_filepath
        from hydesign.assembly.hpp_assembly_hifi_dems import hpp_model
        
        sim_pars_fn = os.path.join(examples_filepath, 'Europe/hpp_pars_HiFiEMS.yml')
        hpp = hpp_model(sim_pars_fn=sim_pars_fn,
                        input_ts_da=os.path.join(examples_filepath, 'HiFiEMS_inputs/Weather/input_ts_DA.csv'),
                        input_ts_ha=os.path.join(examples_filepath, 'HiFiEMS_inputs/Weather/input_ts_HA.csv'),
                        input_ts_rt=os.path.join(examples_filepath, 'HiFiEMS_inputs/Weather/input_ts_RT.csv'),
                        market_fn=os.path.join(examples_filepath, 'HiFiEMS_inputs/Market/Market2021.csv'),)
        inputs = dict(clearance=20, 
                      sp=350,
                      p_rated=10, 
                      Nwt=12, 
                      wind_MW_per_km2=6,
                      solar_MW=10,
                      surface_tilt=25, 
                      surface_azimuth=180, 
                      DC_AC_ratio=1.5,
                      b_P=40,
                      b_E_h=3,
                      )
        
        res = hpp.evaluate(**inputs)
        hpp.print_design(list(inputs.values()), res)
main()
