Welcome to hydesign
=====================
**HyDesign** is an open-source tool developed by DTU Wind for design and control of utility scale wind-solar-storage based hybrid power plant. It provides a methodology for sizing 
these plants by solving a nested optimization problem: an outer sizing optimization 
and an internal operation optimization. The outer optimization maximizes net present value while considering capital expenditures, 
and the internal optimization focuses on energy management. HyDesign incorporates turbine selection, wake loss models, and degradation models for wind and PV systems, 
and includes battery degradation. The tool applies a parallel "efficient global optimization" algorithm to ensure efficient and effective global optimization.

	**- Key Features:**

		- **Hybrid Power Plant Optimization**: Optimizes the combination of wind, solar, and battery storage to maximize value.
		- **Nested Optimization**: Combines sizing and operation optimization for comprehensive results.
		- **Degradation Models**: Includes degradation over time for accurate long-term performance predictions.
		- **Open-Source**: Available for use and modification by the community.

	**- Use Cases:**

		HyDesign is particularly useful in scenarios where hybrid power plants can provide better returns than single technology plants, 
		such as in locations with comparable wind and solar resources or in electricity markets with variable prices.
		
The :ref:`Quickstart </notebooks/Quickstart.ipynb>` section shows how to set up and perform some basic operations in hydesign.

Documentation Content
------------------------

    .. toctree::
        :maxdepth: 1

	license_citation
	installation
	notebooks/ChangeLog
	tutorials
	api
	contact_info









    
