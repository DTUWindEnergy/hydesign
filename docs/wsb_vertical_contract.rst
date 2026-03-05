WSB Vertical Contract
=====================

Scope
-----

This document fixes the first full modular vertical slice to the baseline
technology set:

- wind
- solar
- battery

For now, this vertical slice excludes additional technology families such as
P2X and excludes strategy or market variants such as balancing market,
constant-output, and HiFi EMS.


Layer Source Of Truth
---------------------

- EMS: ``hydesign/ems/ems.py``
- Costs: ``hydesign/costs/costs.py``
- Finance: ``hydesign/finance/finance.py``
- Assembly: ``hydesign/assembly/hpp_assembly.py``


Current Generated WSB Artifact Set
----------------------------------

The current baseline generated ``wind + solar + battery`` project contains:

- ``weather.py``
- ``wind.py``
- ``pv.py``
- ``ems.py``
- ``ems_long_term.py``
- ``battery_degradation.py``
- ``reliability.py``
- ``costs.py``
- ``finance.py``
- ``assembly.py``

These files are generated from the original sources above and kept together
under one combined project folder.

The generated WSB project is now locally rooted for the main plant-model
layers, but it still depends on shared library support modules such as
``hydesign.openmdao_wrapper``, ``hydesign.look_up_tables``,
``hydesign.nrel_csm_wrapper``, and third-party packages. Those are currently
treated as stable infrastructure, not project-specific generated artifacts.


Current WSB Layer Interfaces
----------------------------

The current baseline WSB assembly uses these layer boundaries:

EMS outputs consumed later:

- ``wind_t_ext``
- ``solar_t_ext``
- ``price_t_ext``
- ``hpp_t``
- ``hpp_curt_t``
- ``b_t``
- ``b_E_SOC_t``
- ``penalty_t``

Long-term EMS / degradation outputs consumed by finance:

- ``hpp_t_with_deg``
- ``penalty_t_with_deg`` mapped into finance input ``penalty_t``

Costs components used in WSB assembly:

- wind costs:
  inputs ``Nwt``, ``hh``, ``d``, ``p_rated``, ``wind_t``
  outputs ``CAPEX_w``, ``OPEX_w``
- solar costs:
  inputs ``solar_MW``, ``DC_AC_ratio``
  outputs ``CAPEX_s``, ``OPEX_s``
- battery costs:
  inputs ``b_P``, ``b_E``, ``SoH``
  outputs ``CAPEX_b``, ``OPEX_b``
- shared costs:
  inputs ``G_MW``, ``Awpp``, ``Apvp``
  outputs ``CAPEX_sh``, ``OPEX_sh``

Finance inputs in WSB assembly:

- operational inputs:
  ``price_t_ext``, ``hpp_t_with_deg``, ``penalty_t``
- cost inputs:
  ``CAPEX_w``, ``CAPEX_s``, ``CAPEX_b``, ``CAPEX_el``
  ``OPEX_w``, ``OPEX_s``, ``OPEX_b``, ``OPEX_el``
- WACC inputs:
  ``wind_WACC``, ``solar_WACC``, ``battery_WACC``
- tax input:
  ``tax_rate``

Finance outputs expected by assembly:

- ``CAPEX``
- ``OPEX``
- ``revenues``
- ``NPV``
- ``IRR``
- ``NPV_over_CAPEX``
- ``LCOE``
- ``mean_AEP``
- ``penalty_lifetime``
- ``break_even_PPA_price``


Modularization Order
--------------------

For the WSB vertical slice, the recommended order is:

1. modular EMS
2. modular costs
3. modular finance
4. modular assembly

This order keeps integration risk down because assembly only becomes modular
after the lower-layer contracts are already stable.
