# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]
### Added
- CSV file mapping tool using the csv files in given folder. [INCORE1-506](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-506)
- Network dataset mapping method. [INCORE1-602](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-602) 

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

