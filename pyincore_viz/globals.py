# Copyright (c) 2019 University of Illinois and others. All rights reserved.
#
# This program and the accompanying materials are made available under the
# terms of the Mozilla Public License v2.0 which accompanies this distribution,
# and is available at https://www.mozilla.org/en-US/MPL/2.0/

import os
import logging
from logging import config as logging_config

PACKAGE_VERSION = "1.10.0"

INCORE_GEOSERVER_WMS_URL = "https://tools.in-core.org/geoserver/incore/wms"
INCORE_GEOSERVER_DEV_WMS_URL = (
    "https://dev.in-core.org/geoserver/incore/wms"
)

INCORE_API_DEV_URL = "https://dev.in-core.org"
INCORE_API_URL = "https://tools.in-core.org"

PYINCORE_VIZ_ROOT_FOLDER = os.path.dirname(os.path.dirname(__file__))
USER_HOME = os.path.expanduser("~")
USER_CACHE_DIR = ".incore"
PYINCORE_USER_CACHE = os.path.join(USER_HOME, USER_CACHE_DIR)

LOGGING_CONFIG = os.path.abspath(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "logging.ini")
)
logging_config.fileConfig(LOGGING_CONFIG)
LOGGER = logging.getLogger("pyincore-viz")
