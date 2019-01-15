# Thermal data processing for ThermCERT

Codes for producing ThermCERT data layers derived from satellite thermal
imagery. 

## Procedure in brief

1. Obtain Landsat imaging for a specified region (county) and date range
2. Mask clouds etc, and aggregate time-series.
3. Use ambient temperature data to normalise geographical variations
4. Use Land-Use/Land-Cover data to filter residential areas, and aggregate spatially by LSOA to derive heat map or "Residential Heat Loss" (RHL) vector
5. Combine derived RHL vector with energy consumption data to derive "Residential Energy Efficiency Metric" (REEM) vector
6. Perform validation of these outputs

## process module
This python module contains functions to produce all the required outputs 
* *builder.py* is the main module for producing RHL outputs (steps 1-4 in the procedure above)
* *combine_lst_gdf.py* is a simple script for combining outputs from *builder* for multiple counties and extracting the required table column for RHL
* *config.py* specifies the configuration for *builder*
* *diagnostics.py* contains functions for producing some diagnostic plots from the *builder* procedure
* *estimate_energy_eff.py* is used for estimating REEM from RHL and energy consumption data
* *validate_epc.py* is used for validating RHL outputs against EPC data
* *corine.py* contains functions for processing the Corine land cover raster (or equally the combined UA+Corine raster)
* *landsat.py* contains functions for interacting with Landsat data on the cloud
* *met_climate.py* contains functions for processing Met Office weather data to obtain ambient temperature maps
* *land_surface_temperature.py* is the module for estimating LST from Landsat thermal images, including atm corr and emissivity corr

## process/common
This module contains some general utility functions that are not specific to this project
* *geoplot.py* contains convenience functions for plotting geospatial data, wrapping various functions from matplotlib, rasterio, geopandas
* *raster_utils.py* contains some general functions for interacting with raster geodata and wrapping certain rasterio functions  

## Running the code
1. Modify *config.py*
2. Run main function in *builder.py* with required parameters
3. Use diagnostic plots to check masking etc
4. Use *combine_lst_gdf.py* to combine outputs
5. Use *estimate_energy_eff.py* to produce REEM output
6. Use *validate_epc.py* to validate results (if required)


## Notebooks
Jupyter notebooks were used to develop the algorithm and are mostly superceded by the modular code stored in the subdirectory
* *LSOA_LST_builder.ipynb* encompasses most of steps 1-4 in the procedure outlined above, and produces raster and vector outputs of aggregated Land Surface Temperature (LST) or RHL
* *LST_SNR_analysis.ipynb* estimates uncertainties in LST/RHL raster/vector outputs by analysing variance in time-series data and estimating signal-to-noise ratio (SNR)
* *LSOA_EPC_aggregator.ipynb* aggregates EPC records by LSOA and by date range
* *EPC_LST_Comparisson.ipynb* performs comparisons between LST/RHL results and aggregated EPC data for validation purposes
* *ECO_LST_Comparison.ipynb* performs comparisons between LST/RHL results and historical ECO installations from the E.ON database for validation purposes
* *Urban_Atlas_Combine.ipynb* takes the Copernicus Urban Atlas vector data for a number of regions and combines them with the underlying Corine raster to produce a combined raster map of Land Use/Land Cover containing the most high-resolution data available for a contiguous region