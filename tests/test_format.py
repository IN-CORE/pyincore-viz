# Copyright (c) 2019 University of Illinois and others. All rights reserved.
#
# This program and the accompanying materials are made available under the
# terms of the Mozilla Public License v2.0 which accompanies this distribution,
# and is available at https://www.mozilla.org/en-US/MPL/2.0/
import pycodestyle
import os

from pyincore_viz.globals import PYINCORE_VIZ_ROOT_FOLDER

paths = [
    os.path.join(PYINCORE_VIZ_ROOT_FOLDER, 'pyincore_viz'),
    os.path.join(PYINCORE_VIZ_ROOT_FOLDER, 'tests/pyincore_viz/'),
]


def test_conformance(paths=paths):
    """Test that pyIncore conforms to PEP-8."""
    style = pycodestyle.StyleGuide(quiet=False, max_line_length=120)
    result = style.check_files(paths)
    assert result.total_errors == 0
