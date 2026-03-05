Assembly Module Contract
========================

Goal
----

The WSB assembly generator should no longer be a source-rewrite of the full
``hpp_assembly.py`` implementation. It should compose the generated
``hpp_model`` from declared assembly module contributions.


Current WSB Assembly Modules
----------------------------

The current baseline ``wind + solar + battery`` assembly is composed from:

- ``base``: shared system wiring, weather preprocessing component, baseline EMS,
  common parameter setup, and system-level metrics
- ``wind``: wind production, degradation, reliability, wind design variables,
  and wind-specific reporting terms
- ``solar``: solar production, degradation, reliability, solar design variables,
  and solar-specific reporting terms
- ``battery``: battery degradation, battery reliability, battery design
  variables, and battery-specific reporting terms
- ``operations``: long-term degraded operation propagation
- ``finance``: cost components, finance component, WACC and tax inputs, and
  finance reporting terms


Module Contract
---------------

Each assembly module contributes some combination of:

- import lines for local generated artifacts
- initialization setup lines
- OpenMDAO component declarations
- fixed ``prob.set_val(...)`` lines
- design variable declarations
- evaluation-time setup lines
- evaluation-time ``prob.set_val(...)`` lines
- post-run metric lines

The final generated ``hpp_model`` is a composition of these declared
contributions, while the shared ``hpp_base`` helper still comes from the
original source of truth.


Extension Rule
--------------

Future developers should add a new assembly module file and register it in
``hydesign.modules.assembly.registry`` instead of editing one monolithic
generated template.
