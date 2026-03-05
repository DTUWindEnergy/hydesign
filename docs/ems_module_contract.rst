EMS Module Contract
===================

Source of Truth
---------------

For EMS modularization, the behavioral baseline is the original
``hydesign/ems`` package, not ``hydesign/ems_modular``.

Phase 1 specifically targets:

- original CPLEX behavior only
- parity with the original wind-solar-battery EMS in ``hydesign/ems/ems.py``
- extension points that future developers can use to add new technologies

``hydesign/ems_modular`` may still be useful as inspiration, but it is not the
reference implementation.


What Is Common In The Original EMS
----------------------------------

Across the original EMS files, the reusable CPLEX flow is:

1. build OpenMDAO-facing inputs and outputs
2. normalize scalar inputs and create a synthetic hourly pandas index
3. allocate lifetime and batch result arrays
4. split the time series into batches
5. for each batch:

   a. slice batch data
   b. compute peak-price penalty helpers
   c. create a docplex model
   d. define variables
   e. add objective terms
   f. add constraints
   g. solve
   h. extract batch outputs into numpy arrays

6. finalize state variables such as end-of-horizon SOC
7. expand batch outputs to lifetime outputs

This sequence is visible in:

- ``hydesign/ems/ems.py`` for the base wind-solar-battery case
- ``hydesign/ems/ems_P2X.py`` for the first technology extension


Phase 1 EMS Modules
-------------------

The initial EMS module split should be:

- ``base``:
  shared dispatch scaffolding, HPP output, curtailment, peak-hour penalty,
  batching, solve loop, and lifetime expansion
- ``wind``:
  wind production input and power-balance contribution
- ``solar``:
  solar production input and power-balance contribution
- ``battery``:
  battery variables, SOC constraints, smoothing cost, and battery outputs
- ``p2x``:
  electrolyzer, hydrogen storage, hydrogen revenue, and hydrogen-specific outputs


Important Boundary
------------------

Not every existing EMS file represents a technology.

Files such as:

- ``ems_constant_output.py``
- ``ems_BM.py``
- ``ems_hifi.py``
- ``ems_hifi_dems.py``

represent dispatch modes, market setups, or scenario variants, not technology
modules. Those should later be modeled as a separate axis, for example
``strategy`` or ``market`` modules, rather than being forced into the
technology list.

Phase 1 should stay focused on the technology axis.


Module Contract
---------------

Each EMS module must declare:

- its name and kind
- which original file(s) are the source of truth
- its dependencies
- which component inputs it adds
- which component outputs it adds
- which batch outputs it owns
- which execution stages it contributes to

The current execution stages are:

- ``openmdao_inputs``
- ``openmdao_outputs``
- ``compute_prepare``
- ``batch_allocate``
- ``batch_slice``
- ``cplex_variables``
- ``cplex_power_balance``
- ``cplex_objective``
- ``cplex_constraints``
- ``cplex_extract``
- ``finalize_batch_outputs``
- ``lifetime_outputs``


Developer Rule Of Thumb
-----------------------

To add a future technology, a developer should ideally do only this:

1. add a new file under ``hydesign/modules/ems/``
2. declare the module metadata and dependencies
3. implement contributions only for the execution stages that technology needs
4. register the module in one registry
5. verify parity against a reference case

If adding a technology requires editing many unrelated modules, the contract is
too weak and should be improved before continuing.


Current Code Scaffold
---------------------

The repo now includes:

- ``hydesign/modules/ems/contracts.py`` for the stage and metadata contract
- ``hydesign/modules/ems/base.py`` for the shared EMS module metadata
- ``hydesign/modules/ems/wind.py``
- ``hydesign/modules/ems/solar.py``
- ``hydesign/modules/ems/battery.py``
- ``hydesign/modules/ems/p2x.py``
- ``hydesign/modules/ems/registry.py`` for composition and dependency checks

This scaffold does not yet change the running EMS code. It fixes the design
target before migrating solver logic.
