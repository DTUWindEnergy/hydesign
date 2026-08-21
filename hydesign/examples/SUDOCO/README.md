# Coupled wind-control tutorial data

`hkn_layout.csv` contains the 69-turbine Hollandse Kust Noord (HKN) reference
layout used by the coupled GNN/load tutorial. The tutorial evaluates complete
`hydesign.assembly.hpp_pywake_gnn_loads.hpp_model` objects with the repository's
HKN weather and Dutch MDF time series; it does not replay precomputed EMS
dispatch results.

`hkn_yaw_modes_168h.csv` and `generate_hkn_yaw_modes.py` are retained as a
compact research-data regeneration route. They are not inputs to the public
notebook.
