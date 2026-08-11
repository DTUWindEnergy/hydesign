Technology deployment schedules
===============================

The base wind-solar-battery ``hpp_model`` supports plants whose technologies
are commissioned and retired in different project years. Add a
``technologies`` list to the simulation-parameter YAML file:

.. code-block:: yaml

   technologies:
     - name: wind
       deployment_year: 0
       lifetime: 25
       capex_phasing:
         years: [-1, 0]
         shares: [0.2, 0.8]
     - name: battery
       deployment_year: 5
       lifetime: 20
       capex_phasing:
         years: [-1, 0]
         shares: [0.3, 0.7]
     - name: solar
       deployment_year: 9
       lifetime: 25

Years are zero based. Here wind operates in project years 0--24, the battery
in years 5--24, and solar in years 9--33. The project horizon is derived from
the latest retirement, so it is 34 years. A technology omitted from the list
is not part of the plant. ``pv``/``pvp``, ``wpp``, and ``bess``/``storage``
are accepted as aliases for solar, wind, and battery.

The same schedule can be supplied when the evaluator is instantiated. Values
passed to the constructor take precedence over the YAML file:

.. code-block:: python

   from hydesign.assembly.hpp_assembly import hpp_model

   hpp = hpp_model(
       sim_pars_fn="hpp_pars.yml",
       technologies={
           "wind": {"deployment_year": 0, "lifetime": 25},
           "battery": {"deployment_year": 5, "lifetime": 20},
           "solar": {"deployment_year": 9, "lifetime": 25},
       },
   )

For every distinct operating stage, HyDesign runs the EMS with only the active
generation and storage technologies. The resulting representative year is
repeated only over that stage's contiguous operating interval, then the stage
intervals are concatenated. Degradation starts at each deployment, while
production and reliability are limited to the corresponding operating
lifetime.

``capex_phasing.years`` is relative to that technology's deployment year and shares are normalized to
one. If phasing is omitted, the complete technology CAPEX is paid in its
deployment year. 

