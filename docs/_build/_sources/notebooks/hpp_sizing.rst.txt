Size a HPP plant based on a simplified hpp model
================================================

This **static** notebook illustrates how to solve hybrid power plant
sizing optimization in a specific location based on preselected turbine
type.

**To execute this setup a server is required as it relies on parallel
evaluation of the model.**

Executable versions of this notebook and a submission script for a SCRUM based server are provided:

`./hydesign/examples/hydesign_sizing_simple_hpp.ipynb`
`./hydesign/examples/hydesign_sizing_simple_hpp.sh`
`./hydesign/examples/hydesign_sizing.sh`

Sizing a hybrid power plant consists on designing the following
parameters:

Design Variables
~~~~~~~~~~~~~~~~

**Wind Plant design:**

-  Number of wind turbines in the wind plant [-] (``Nwt``)
-  Wind power installation density [MW/km2] (``wind_MW_per_km2``): This
   parameter controls how closely spaced are the turbines, which in
   turns affect how much wake losses are present.

**PV Plant design:**

-  Solar plant power capacity [MW] (``solar_MW``)

**Battery Storage design:**

-  Battery power [MW] (``b_P``)
-  Battery energy capacity in hours [MWh] (``b_E_h``): Battery storage
   capacity in hours of full battery power (``b_E = b_E_h * b_P``).
-  Cost of battery power fluctuations in peak price ratio [-]
   (``cost_of_batt_degr``): This parameter controls how much penalty is
   given to do ramps in battery power in the HPP operation.

Possible objective functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  LCOE: Levelized cost of energy.
-  IRR: Internal rate of return. It is not defined for projects that
   produce negative net present values (NPV). Hydesign returns IRR = 0 if the
   NPV < 0. Nevertheless, optimizations can be problematic for sites
   without a clear case.
-  NPV/CAPEX: Net present value over total CAPEX. A good proxy variable,
   that will produce optimal sites with the optimal IRR, but that is
   defined on sites with negative NPV.

The available variables for optimization are:

::

    'NPV_over_CAPEX',
    'NPV [MEuro]',
    'IRR',
    'LCOE [Euro/MWh]',
    'CAPEX [MEuro]',
    'OPEX [MEuro]',
    'penalty lifetime [MEuro]',

.. code:: ipython3

    # Install hydesign if needed
    try:
        import hydesign
    except ModuleNotFoundError:
        !pip install git+https://gitlab.windenergy.dtu.dk/TOPFARM/hydesign.git
        
    import pandas as pd
    from hydesign.EGO_surrogate_based_optimization import EGO_path
    from hydesign.examples import examples_filepath

Optimize using EGO
~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    EGO_simple = f'{EGO_path}EGO_surrogate_based_optimization_simple_hpp.py'

.. code:: ipython3

    %run $EGO_simple --help


.. parsed-literal::

    usage: EGO_surrogate_based_optimization_simple_hpp.py [-h] [--example EXAMPLE]
                                                          [--name NAME]
                                                          [--longitude LONGITUDE]
                                                          [--latitude LATITUDE]
                                                          [--altitude ALTITUDE]
                                                          [--input_ts_fn INPUT_TS_FN]
                                                          [--sim_pars_fn SIM_PARS_FN]
                                                          [--opt_var OPT_VAR]
                                                          [--rotor_diameter_m ROTOR_DIAMETER_M]
                                                          [--hub_height_m HUB_HEIGHT_M]
                                                          [--wt_rated_power_MW WT_RATED_POWER_MW]
                                                          [--surface_tilt_deg SURFACE_TILT_DEG]
                                                          [--surface_azimuth_deg SURFACE_AZIMUTH_DEG]
                                                          [--DC_AC_ratio DC_AC_RATIO]
                                                          [--num_batteries NUM_BATTERIES]
                                                          [--n_procs N_PROCS]
                                                          [--n_doe N_DOE]
                                                          [--n_clusters N_CLUSTERS]
                                                          [--n_seed N_SEED]
                                                          [--max_iter MAX_ITER]
                                                          [--final_design_fn FINAL_DESIGN_FN]
    
    optional arguments:
      -h, --help            show this help message and exit
      --example EXAMPLE     ID (index( to run an example site, based on
                            ./examples/examples_sites.csv
      --name NAME           Site name
      --longitude LONGITUDE
                            Site longitude
      --latitude LATITUDE   Site latitude
      --altitude ALTITUDE   Site altitude
      --input_ts_fn INPUT_TS_FN
                            Input ts file name
      --sim_pars_fn SIM_PARS_FN
                            Simulation parameters file name
      --opt_var OPT_VAR     Objective function for sizing optimization, should be
                            one of: ['NPV_over_CAPEX','NPV [MEuro]','IRR','LCOE
                            [Euro/MWh]','CAPEX [MEuro]','OPEX [MEuro]','penalty
                            lifetime [MEuro]']
      --rotor_diameter_m ROTOR_DIAMETER_M
                            WT rotor diameter [m]
      --hub_height_m HUB_HEIGHT_M
                            WT hub height [m]
      --wt_rated_power_MW WT_RATED_POWER_MW
                            WT rated power [MW]
      --surface_tilt_deg SURFACE_TILT_DEG
                            PV surface tilt [deg]
      --surface_azimuth_deg SURFACE_AZIMUTH_DEG
                            PV surface azimuth [deg]
      --DC_AC_ratio DC_AC_RATIO
                            PV DC/AC ratio, this ratio defines how much
                            overplanting of DC power is done with respect the
                            inverter. P_DC/P_AC [-]
      --num_batteries NUM_BATTERIES
                            Maximum number of batteries to be considered in the
                            design.
      --n_procs N_PROCS     Number of processors to use
      --n_doe N_DOE         Number of initial model simulations
      --n_clusters N_CLUSTERS
                            Number of clusters to explore local vs global optima
      --n_seed N_SEED       Seed number to reproduce the sampling in EGO
      --max_iter MAX_ITER   Maximum number of parallel EGO ierations
      --final_design_fn FINAL_DESIGN_FN
                            File name of the final design stored as csv




.. code:: ipython3

    %run $EGO_simple \
        --example 0 \
        --opt_var "NPV_over_CAPEX"\
        --rotor_diameter_m 100\
        --hub_height_m 120\
        --wt_rated_power_MW 2\
        --surface_tilt_deg 20\
        --surface_azimuth_deg 180\
        --DC_AC_ratio 1\
        --num_batteries 2\
        --n_procs  1\
        --n_doe 31\
        --n_clusters 2\
        --n_seed 0\
        --max_iter 10\
        --final_design_fn 'hydesign_simple_design_0.csv'


.. parsed-literal::

    Selected example site:
    ---------------------------------------------------
    case                                              India
    name                              Indian_site_good_wind
    longitude                                     77.500226
    latitude                                       8.334294
    altitude                                     679.803454
    input_ts_fn    India/input_ts_Indian_site_good_wind.csv
    sim_pars_fn                          India/hpp_pars.yml
    price_fn                  India/Indian_elec_price_t.csv
    price_col                                         Price
    Name: 0, dtype: object
    
    
    
    
    Sizing a HPP plant at Indian_site_good_wind:
    
    longitude = 77.50022582725498
    latitude = 8.334293917013909
    altitude = 679.8034540123396
    
    rotor_diameter_m = 100.0
    hub_height_m = 120.0
    wt_rated_power_MW = 2.0
    surface_tilt_deg = 20.0
    surface_azimuth_deg = 180.0
    DC_AC_ratio = 1.0
    
    
    
    Initial 31 simulations took 14.73 minutes
    
    Update sm and extract candidate points took 0.01 minutes
    Check-optimal candidates: new 4 simulations took 1.92 minutes
      rel_yopt_change = -1.13E-01
    Iteration 1 took 1.94 minutes
    
    Update sm and extract candidate points took 0.01 minutes
    Check-optimal candidates: new 4 simulations took 1.57 minutes
      rel_yopt_change = 0.00E+00
    Iteration 2 took 1.59 minutes
    
    Update sm and extract candidate points took 0.01 minutes
    Check-optimal candidates: new 3 simulations took 1.44 minutes
      rel_yopt_change = -1.12E-01
    Iteration 3 took 1.46 minutes
    
    Update sm and extract candidate points took 0.02 minutes
    Check-optimal candidates: new 4 simulations took 1.91 minutes
      rel_yopt_change = 0.00E+00
    Iteration 4 took 1.94 minutes
    
    Update sm and extract candidate points took 0.02 minutes
    Check-optimal candidates: new 3 simulations took 1.43 minutes
      rel_yopt_change = -2.18E-02
    Iteration 5 took 1.46 minutes
    
    Update sm and extract candidate points took 0.02 minutes
    Check-optimal candidates: new 3 simulations took 1.45 minutes
      rel_yopt_change = 0.00E+00
    Iteration 6 took 1.48 minutes
    
    Update sm and extract candidate points took 0.02 minutes
    Check-optimal candidates: new 4 simulations took 0.82 minutes
      rel_yopt_change = 0.00E+00
    Iteration 7 took 0.85 minutes
    
    Update sm and extract candidate points took 0.02 minutes
    Check-optimal candidates: new 4 simulations took 1.87 minutes
      rel_yopt_change = 0.00E+00
    Iteration 8 took 1.91 minutes
    
    Surrogate based optimization is converged.
    
    Design:
    ---------------
    Nwt: 113
    wind_MW_per_km2 [MW/km2]: 6.616
    solar_MW [MW]: 203
    b_P [MW]: 30
    b_E_h [h]: 4
    cost_of_battery_P_fluct_in_peak_price_ratio: 1.025
    
    
    NPV_over_CAPEX: 0.558
    NPV [MEuro]: 217.751
    IRR: 0.105
    LCOE [Euro/MWh]: 23.633
    CAPEX [MEuro]: 390.337
    OPEX [MEuro]: 8.092
    penalty lifetime [MEuro]: 0.000
    AEP [GWh]: 1549.256
    GUF: 0.590
    grid [MW]: 300.000
    wind [MW]: 226.000
    solar [MW]: 203.000
    Battery Energy [MWh]: 120.000
    Battery Power [MW]: 30.000
    Total curtailment [GWh]: 248.917
    Awpp [km2]: 34.160
    Rotor diam [m]: 100.000
    Hub height [m]: 120.000
    Number_of_batteries: 2.000
    
    Optimization with 8 iterations and 60 model evaluations took 27.86 minutes
    


.. code:: ipython3

    %run $EGO_simple \
        --example 0 \
        --opt_var "NPV_over_CAPEX"\
        --rotor_diameter_m 100\
        --hub_height_m 120\
        --wt_rated_power_MW 2\
        --surface_tilt_deg 20\
        --surface_azimuth_deg 180\
        --DC_AC_ratio 1\
        --num_batteries 1\
        --n_procs  31\
        --n_doe 31\
        --n_clusters 16\
        --n_seed 0\
        --max_iter 10\
        --final_design_fn 'hydesign_simple_design_0.csv'


.. parsed-literal::

    Selected example site:
    ---------------------------------------------------
    case                                              India
    name                              Indian_site_good_wind
    longitude                                     77.500226
    latitude                                       8.334294
    altitude                                     679.803454
    input_ts_fn    India/input_ts_Indian_site_good_wind.csv
    sim_pars_fn                          India/hpp_pars.yml
    price_fn                  India/Indian_elec_price_t.csv
    price_col                                         Price
    Name: 0, dtype: object
    
    
    
    
    Sizing a HPP plant at Indian_site_good_wind:
    
    longitude = 77.50022582725498
    latitude = 8.334293917013909
    altitude = 679.8034540123396
    
    rotor_diameter_m = 100.0
    hub_height_m = 120.0
    wt_rated_power_MW = 2.0
    surface_tilt_deg = 20.0
    surface_azimuth_deg = 180.0
    DC_AC_ratio = 1.0
    
    
    
    Initial 31 simulations took 0.49 minutes
    
    Update sm and extract candidate points took 0.05 minutes
    Check-optimal candidates: new 20 simulations took 0.49 minutes
      rel_yopt_change = -2.25E-01
    Iteration 1 took 0.56 minutes
    
    Update sm and extract candidate points took 0.06 minutes
    Check-optimal candidates: new 21 simulations took 0.5 minutes
      rel_yopt_change = 0.00E+00
    Iteration 2 took 0.58 minutes
    
    Update sm and extract candidate points took 0.07 minutes
    Check-optimal candidates: new 19 simulations took 0.52 minutes
      rel_yopt_change = -3.09E-02
    Iteration 3 took 0.61 minutes
    
    Update sm and extract candidate points took 0.08 minutes
    Check-optimal candidates: new 21 simulations took 0.52 minutes
      rel_yopt_change = 0.00E+00
    Iteration 4 took 0.63 minutes
    
    Update sm and extract candidate points took 0.09 minutes
    Check-optimal candidates: new 26 simulations took 0.54 minutes
      rel_yopt_change = 0.00E+00
    Iteration 5 took 0.66 minutes
    
    Update sm and extract candidate points took 0.1 minutes
    Check-optimal candidates: new 23 simulations took 0.54 minutes
      rel_yopt_change = 0.00E+00
    Iteration 6 took 0.68 minutes
    
    Surrogate based optimization is converged.
    
    Design:
    ---------------
    Nwt: 131
    wind_MW_per_km2 [MW/km2]: 8.873
    solar_MW [MW]: 155
    b_P [MW]: 41
    b_E_h [h]: 2
    cost_of_battery_P_fluct_in_peak_price_ratio: 1.772
    
    
    NPV_over_CAPEX: 0.534
    NPV [MEuro]: 224.614
    IRR: 0.103
    LCOE [Euro/MWh]: 23.870
    CAPEX [MEuro]: 420.919
    OPEX [MEuro]: 8.973
    penalty lifetime [MEuro]: 0.038
    AEP [GWh]: 1664.925
    GUF: 0.634
    grid [MW]: 300.000
    wind [MW]: 262.000
    solar [MW]: 155.000
    Battery Energy [MWh]: 82.000
    Battery Power [MW]: 41.000
    Total curtailment [GWh]: 618.019
    Awpp [km2]: 29.529
    Rotor diam [m]: 100.000
    Hub height [m]: 120.000
    Number_of_batteries: 1.000
    
    Optimization with 6 iterations and 161 model evaluations took 4.67 minutes
    


