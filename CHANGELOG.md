# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]
### Updated
- Basemap api updated to use open street map [#138](https://github.com/IN-CORE/pyincore-viz/issues/138)

## [1.8.3] - 2023-08-16
### Changed
- Pytest changed to use micromamba [#128](https://github.com/IN-CORE/pyincore-viz/issues/128)

### Fixed
- Conda package building and uploading github action [#131](https://github.com/IN-CORE/pyincore-viz/issues/131)

## [1.8.2] - 2023-06-14

### Changed
- Updated ipyleaflet to 0.17.2 [#117](https://github.com/IN-CORE/pyincore-viz/issues/117)

### Fixed
- Documentation docker build process [#119](https://github.com/IN-CORE/pyincore-viz/issues/119)

## [1.8.1] - 2022-11-16

### Changed
- Deprecated inventory_to_geodataframe method from plotutil. [#89](https://github.com/IN-CORE/pyincore-viz/issues/89)
- Allow case insensitive matching between fragility curve parameter and demand types [#28](https://github.com/IN-CORE/pyincore-viz/issues/28)

### Fixed
- Pypi build error caused from missing Azure site [#93](https://github.com/IN-CORE/pyincore-viz/issues/93)
- CRS error caused by rasterio. [#86](https://github.com/IN-CORE/pyincore-viz/issues/86)

## [1.8.0] - 2022-09-14

### Added
- Missing docstrings in GeoUtil. [#82](https://github.com/IN-CORE/pyincore-viz/issues/82)
- Doc strings to the methods that are missing them. [#82](https://github.com/IN-CORE/pyincore-viz/issues/82)
- Pytest github action. [#16](https://github.com/IN-CORE/pyincore-viz/issues/16)

### Changed
- Deprecated get_standard_x_y method from plotutil. [#62](https://github.com/IN-CORE/pyincore-viz/issues/62)
- Updated ipyleaflet and codes were refactored. [#81](https://github.com/IN-CORE/pyincore-viz/issues/81)

## [1.7.0] - 2022-07-27

### Changed
- Gdf map visualization accepts both dataset and gdf. [#72](https://github.com/IN-CORE/pyincore-viz/issues/72)
- Updated network visualization. [#68](https://github.com/IN-CORE/pyincore-viz/issues/68)

## [1.6.0] - 2022-06-29

### Added
- Household income and poverty visualization to pop result table. [#55](https://github.com/IN-CORE/pyincore-viz/issues/55)
- Raster hazard overlay in gdf map visualization. [#58](https://github.com/IN-CORE/pyincore-viz/issues/58)

### Changed
- Made pyincore-viz build with legacy naming for pypi publish. [#57](https://github.com/IN-CORE/pyincore-viz/issues/57)
- Revised README for pypi publication. [#63](https://github.com/IN-CORE/pyincore-viz/issues/63)

## [1.5.1] - 2022-04-12

### Fixed
- Boolean comparison in adding dislocation dataframe. [#52](https://github.com/IN-CORE/pyincore-viz/issues/52)

## [1.5.0] - 2022-02-07

### Added
- Multiple vector dataset visualization. [#19](https://github.com/IN-CORE/pyincore-viz/issues/19)

### Changed
- Rename master branch to main. [#37](https://github.com/IN-CORE/pyincore-viz/issues/37)
- Rename housing unit allocation to population result table and updated. [#42](https://github.com/IN-CORE/pyincore-viz/issues/42)
- Make viz methods compatible with the DFR3 naming changes done on pyincore (v1.3.0) [#40](https://github.com/IN-CORE/pyincore-viz/issues/40)

### Fixed
- Auto build of documentation container. [#34](https://github.com/IN-CORE/pyincore-viz/issues/34)

## [1.4.1] - 2021-12-08

### Added
- Add *mplcursors* and *palettable* libraries needed for Retrofit notebook example. [#29](https://github.com/IN-CORE/pyincore-viz/issues/29)

## [1.4.0] - 2021-10-27

### Added
- Conda build recipe for publishing the pyincore-data to conda channel. [#7](https://github.com/IN-CORE/pyincore-viz/issues/7)
- Choropleth map using single dataset with multiple field. [#2](https://github.com/IN-CORE/pyincore-viz/issues/2)
- Choropleth map using multiple datasets. [#3](https://github.com/IN-CORE/pyincore-viz/issues/3)
- Analysis visualization for Housing Unit Allocation. [#5](https://github.com/IN-CORE/pyincore-viz/issues/5)
- Automatic documentation container build and publish. [#9](https://github.com/IN-CORE/pyincore-viz/issues/9)

### Changed
- remove check for fragilityCurveRefactored and support for old format [#20](https://github.com/IN-CORE/pyincore-viz/issues/20)

### Fixed
- Bug in getting gdf map from the geoserver. [#14](https://github.com/IN-CORE/pyincore-viz/issues/14)

## [1.3.0] - 2021-07-28

### Added
- pyincore-data reference link to pyincore-viz doc [INCORE1-1299](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-1299)

### Changed
- docstrings expanded in pyincore viz classes in Google notation [INCORE1-1216](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-1216)
- Standardize the x, y, z coordinates in fragility curves plotting to Numpy object [INCORE1-1276](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-1276)
- replace key with fullName [INCORE1-1296](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-1296)
- Changed default map provider to "OpenStreetMaps.Mapnik" from "Stamen.Terrain" to avoid issues loading points that are not on land (bridges). User can configure map provider if needed. [INCORE1-1268](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-1268)

## [1.2.0] - 2021-06-16

### Added
- Legend for earthquake visualization [INCORE1-1214](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-1214)
- Ability to plot Raster dataset [INCORE1-1222](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-1222)

### Changed
- Renamed the method 'map_raster_overlay_from_file' to 'map_raster_overlay_from_file' to avoid confusion [INCORE1-1224](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-1224)

### Fixed
- Combine refactored and legacy fragility plotting into the main function "get_fragility_plot" for backward compatibility [INCORE1-1228](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-1228)

## [1.1.1] - 2021-05-21

### Added
- Ability to visualize dataset based earthquakes for a specific demand type [INCORE1-1202](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-1202)

## [1.1.0] - 2021-05-19

### Added
- Geodataframe based heatmap [INCORE1-1128](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-1128)
- Visualize refactored (equation) based 2d and 3d fragility curves [INCORE1-1144](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-1144)

## [1.0.0] - 2021-04-14

### Added
- Heatmap for point, line, and polygon dataset [INCORE1-1081](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-1081)

### Fixed
- pytest for network dataset [INCORE1-866](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-866)
- pyincore-viz documentation [INCORE1-1103](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-1103)

## [0.3.0] - 2021-02-15

### Added
- Fit ipyleaflet map boundaries using bounding box [INCORE1-597](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-597)

### Fixed
- Update demand_types and demand_units when plotting fragility curves [INCORE1-998](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-998)

## [0.2.5] - 2020-12-10

### Changed
- updated ipyleaflet to 0.13 [INCORE1-866](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-866)

## [0.2.4] - 2020-10-28

### Added
- Method to plot fragilities [INCORE1-777](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-777)
- Plot map for table datasets. [INCORE1-821](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-821)
- Plot map for local raster dataset. [INCORE1-599](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-599)
- Plot map for vector, raster, and table dataset together. [INCORE1-598](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-598)
- Created a method to view fragility. [INCORE1-777](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-777)
- Created a plot map for list of csv dataset in ipyleaflet. [INCORE1-764](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-764)

### Changed
- Test file has been changed to use individual method for testing.  [INCORE1-768](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-768)
- Test file changed to use id token. [INCORE1-801](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-801)
- Pyincore-viz uses pytest. [INCORE1-768](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-768)

### Fixed
-  cleaning layer when map layer selection changed. [INCORE1-808](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-808)

## [0.2.3] - 2020-09-03

### Added
- documentation, links, docker script [INCORE1-654](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-654)
- Network dataset mapping method. [INCORE1-602](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-602) 
- Logger for log reporting. [INCORE1-748](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-748)

## [0.2.2] - 2020-07-31

### Added
- PEP-8 format test to the library.  [INCORE1-706](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-706)
- wms layer availability checking [INCORE1-609](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-609)

## [0.2.1] - 2020-05-27

### Added
- jupyter notebook example under examples folder [INCORE1-595](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-595)
- geo utility methods (e.g. mergeing bounding box) [INCORE1-595](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-595)

### Fixed
- fixed geomap error in displaying point layer due to jupyterlab bug [INCORE1-563](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-563)
- made consistent API regarding arguments of methods [INCORE1-595](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-595)

### Removed
- Unused code (methods) are removed (map_csv_from_dir, create_basemap_ipylft, load_all_data, create_map_widgets, create_geo_map, create_point_icon) [INCORE1-595](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-595)
- test_pyincore_viz.py, jupyter notebook files and sample data folder because of API changes and code changes [INCORE1-611](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-611)

### Updated
- updated jupyter notebook test file more generic so it can run without any modification [INCORE1-556](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-556)
- updated dependent python modules in setup.py [INCORE1-595](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-595)

## [0.2.0] - 2020-02-28

### Added
- map using inventory data and column [INCORE1-488](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-448)
- map using inventory data and csv files in the folder [INCORE1-488](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-448)
- histogram from csv with column [INCORE1-488](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-448)
- histogram for meandamage with damage analysis [INCORE1-488](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-448)

## [0.1.1] - 2019-12-23

### Fixed
- Geoserver urls to point to the ones in Kubernetes environment

## [0.1.0] - 2019-12-20
pyIncore viz release for IN-CORE v1.0

### Added
- CHANGELOG, CONTRIBUTORS, and LICENSE

