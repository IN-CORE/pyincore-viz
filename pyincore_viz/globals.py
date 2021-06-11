# Copyright (c) 2019 University of Illinois and others. All rights reserved.
#
# This program and the accompanying materials are made available under the
# terms of the Mozilla Public License v2.0 which accompanies this distribution,
# and is available at https://www.mozilla.org/en-US/MPL/2.0/

import os
import logging
from logging import config as logging_config

PACKAGE_VERSION = "1.2.0"

INCORE_GEOSERVER_WMS_URL = "https://incore.ncsa.illinois.edu/geoserver/incore/wms"
INCORE_GEOSERVER_DEV_WMS_URL = "https://incore-dev.ncsa.illinois.edu/geoserver/incore/wms"

PYINCORE_VIZ_ROOT_FOLDER = os.path.dirname(os.path.dirname(__file__))

LOGGING_CONFIG = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'logging.ini'))
logging_config.fileConfig(LOGGING_CONFIG)
LOGGER = logging.getLogger('pyincore-viz')
