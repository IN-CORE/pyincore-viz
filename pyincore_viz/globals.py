# Copyright (c) 2019 University of Illinois and others. All rights reserved.
#
# This program and the accompanying materials are made available under the
# terms of the Mozilla Public License v2.0 which accompanies this distribution,
# and is available at https://www.mozilla.org/en-US/MPL/2.0/

import os
import logging
from logging import config as logging_config

PACKAGE_VERSION = "0.1.1"

INCORE_GEOSERVER_WMS_URL = "https://incore-geoserver.ncsa.illinois.edu/geoserver/incore/wms"
INCORE_GEOSERVER_DEV_WMS_URL = "https://incore-dev-kube.ncsa.illinois.edu/geoserver/incore/wms"

PYINCORE_VIZ_ROOT_FOLDER = os.path.dirname(os.path.dirname(__file__))

INCORE_API_PROD_URL = "https://incore.ncsa.illinois.edu"
TEST_INCORE_API_PROD_URL = "http://incore.ncsa.illinois.edu:31888"
INCORE_API_DEV_URL = "https://incore-dev-kube.ncsa.illinois.edu"
TEST_INCORE_API_DEV_URL = "http://incore-dev-kube.ncsa.illinois.edu:31888"
