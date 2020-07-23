# Connect to IN-CORE Services
from pyincore import IncoreClient
from pyincore.dataservice import DataService
from pyincore.hazardservice import HazardService
from pyincore_viz.geoutil import GeoUtil as viz
from pyincore import Dataset

from pyincore.globals import INCORE_API_DEV_URL

client = IncoreClient(INCORE_API_DEV_URL)

# testing datasets
tornado_hazard_id = "5dfa32bbc0601200080893fb"
joplin_bldg_inv_id = "5df7d0de425e0b00092d0082"
eq_hazard_id = "5b902cb273c3371e1236b36b"
shelby_hopital_inv_id = "5a284f0bc7d30d13bc081a28"
shelby_road_id = "5a284f2bc7d30d13bc081eb6"

# Creating a Dataset object with ID and Data Service
eq_metadata = HazardService(client).get_earthquake_hazard_metadata(eq_hazard_id)
eq_dataset_id = eq_metadata['rasterDataset']['datasetId']

# visualize earthquake
eq_dataset = Dataset.from_data_service(eq_dataset_id, DataService(client))
viz.plot_earthquake(eq_hazard_id, client)

# get shelvy building inventory and road
sh_bldg_inv = Dataset.from_data_service(shelby_hopital_inv_id, DataService(client))
sh_road = Dataset.from_data_service(shelby_road_id, DataService(client))

# visualize building inventory
viz.plot_map(sh_bldg_inv, column="struct_typ", category=False, basemap=True)

# visualize building inventory from geoserver
viz.get_wms_map([sh_bldg_inv, sh_road], zoom_level=10)
viz.get_gdf_map([sh_bldg_inv, sh_road], zoom_level=10)
viz.get_gdf_wms_map([sh_bldg_inv], [sh_road], zoom_level=10)

# visualize tornado
viz.plot_tornado(tornado_hazard_id, client, basemap=False)

tornado_dataset_id = HazardService(client).get_tornado_hazard_metadata(tornado_hazard_id)['datasetId']
tornado_dataset = Dataset.from_data_service(tornado_dataset_id, DataService(client))

viz.get_gdf_map([tornado_dataset], zoom_level=11)

# get joplin building inventory
joplin_bldg_inv = Dataset.from_data_service(joplin_bldg_inv_id, DataService(client))

# using wms layer for joplin building inv. gdf will crash the browser
viz.get_gdf_wms_map([tornado_dataset], [joplin_bldg_inv], zoom_level=11)
