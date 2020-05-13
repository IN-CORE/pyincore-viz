# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]

### [Fixed]
- fixed geomap error in displaying point layer due to jupyterlab bug [INCORE1-563](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-563)
- removed unused code and folium package
- made consistent API regarding arguments of methods
- added geo utility methods (e.g. mergeing bounding box)
- added dependent python modules

### [Updated]
- updated jupyter notebook test file more generic so it can run without any modification [INCORE1-556](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-556)
- added jupyter notebook example under examples folder

### [Known Issues]
- can't fit the map (ipyleaflet) with given bounding box
- can't plot multiple datasets (with geopandas.plot() and rasterio.plot())
- can't make ipyleaflet map with raster datasets 
- can't specify styles of layers (datasets)
- performance issue with ipyleaflet with large vector dataset (e.g. joplin building inventory) causes crash of browser
- needed to update pytest 

## [0.2.0] - 2020-02-28

### Added
- added map using inventory data and column [INCORE1-488](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-448)
- added map using inventory data and csv files in the folder [INCORE1-488](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-448)
- added histogram from csv with column [INCORE1-488](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-448)
- added histogram for meandamage with damage analysis [INCORE1-488](https://opensource.ncsa.illinois.edu/jira/browse/INCORE1-448)

## [0.1.1] - 2019-12-23

### [Fixed]
- Geoserver urls to point to the ones in Kubernetes environment

## [0.1.0] - 2019-12-20
pyIncore viz release for IN-CORE v1.0

### [Added]
- CHANGELOG, CONTRIBUTORS, and LICENSE

