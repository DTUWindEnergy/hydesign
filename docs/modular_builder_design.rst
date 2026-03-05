Modular Builder Design
======================

Goal
----

HyDesign should move from predefined variant files such as
``hpp_assembly_P2X.py`` or ``finance_P2X.py`` to an explicit build step:

- the user selects a technology list
- HyDesign generates the concrete project files for that selection
- the generated files stay in one user-facing folder
- later runs can skip regeneration and reuse the generated project

For EMS specifically, the original implementation in ``hydesign/ems`` is the
behavioral reference point for modularization. ``hydesign/ems_modular`` can be
used as inspiration, but not as the parity baseline.


Recommended Layout
------------------

::

    hydesign/
      modules/
        assembly/
          base.py
          wind.py
          solar.py
          battery.py
          p2x.py
        ems/
          base.py
          wind.py
          solar.py
          battery.py
          p2x.py
        finance/
          base.py
          wind.py
          solar.py
          battery.py
          p2x.py
        costs/
          base.py
          wind.py
          solar.py
          battery.py
          p2x.py
        reliability/
          base.py
          wind.py
          solar.py
          battery.py
          p2x.py
      project_builder/
        contracts.py
        registry.py
        builder.py
      combined_projects/
        <project_name>/
          __init__.py
          manifest.yml
          ems.py
          costs.py
          finance.py
          reliability.py
          assembly.py


Naming Recommendation
---------------------

Use ``project_builder`` for the builder logic and ``combined_projects`` for the
generated outputs.

Reason:

- ``project_builder`` clearly describes the explicit generation step
- ``combined_projects`` makes the output folder user-facing and easy to inspect
- the generated folder remains importable as normal Python code


Builder Contract
----------------

The explicit build step should receive a normalized request with these fields:

- ``project_name``: stable user-facing project name
- ``technologies``: ordered technology list such as ``("wind", "solar", "battery")``
- ``artifacts``: generated files to build, for example ``("ems", "finance", "assembly")``
- ``enable_build``: whether to run the generation step in this execution
- ``regenerate``: force rebuild even if an existing generated project matches
- ``output_root``: root folder for generated projects
- ``builder_version``: invalidates old manifests when the generator changes

Build behavior should be:

- if ``enable_build=False``, skip generation and reuse the existing combined project
- if ``enable_build=True`` and ``regenerate=True``, always rebuild
- if ``enable_build=True`` and ``regenerate=False``, rebuild only when missing or stale


Manifest Contract
-----------------

Each generated project should write a ``manifest.yml`` with at least:

- ``project_name``
- ``project_slug``
- ``technologies``
- ``artifacts``
- ``builder_version``
- ``generated_files``
- ``source_hashes``
- ``built_at_utc``

The manifest is the cache key for deciding whether a rebuild is required.


Migration Rule
--------------

Do not replace the existing predefined files all at once.

Use this sequence:

1. keep current files as compatibility entry points
2. move reusable logic into ``hydesign.modules.<artifact>``
3. generate one combined project that reproduces an existing variant
4. verify parity with tests
5. switch callers to the generated project path only after parity is stable


Phase 1 Scope
-------------

The current scaffold defines the contract for these core generated artifacts:

- ``ems``
- ``costs``
- ``finance``
- ``reliability``
- ``assembly``

Additional generated artifacts can be added later through the registry without
changing the user-facing build request format.


Public Builder API
------------------

The user-facing entrypoint should now be the public project builder in
``hydesign.project_builder``:

- ``load_project_build_spec(...)`` reads a YAML or JSON spec
- ``build_project(...)`` resolves dependencies and generates the project

Current phase-1 YAML example:

::

    project_name: wind_solar_battery_cplex
    technologies:
      - wind
      - solar
      - battery
    artifacts:
      - assembly
    output_root: hydesign/combined_projects
    regenerate: true

For the baseline WSB case, requesting ``assembly`` expands to the full local
artifact set required by the generated assembly.
