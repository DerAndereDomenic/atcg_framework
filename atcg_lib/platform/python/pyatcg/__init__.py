import sys

required_version = (3, 8)

if sys.version_info[:2] < required_version:
    msg = "%s requires Python %d.%d+" % (__package__, *required_version)
    raise RuntimeError(msg)

del required_version
del sys

import pathlib
import charonload

PROJECT_ROOT_DIRECTORY = pathlib.Path(__file__).parents[4]

VSCODE_STUBS_DIRECTORY = PROJECT_ROOT_DIRECTORY / "typings"


charonload.module_config["_c_pyatcg"] = charonload.Config(
    # All paths must be absolute
    project_directory=PROJECT_ROOT_DIRECTORY,
    cmake_options={
        "ATCG_CUDA_BACKEND": "On",
        "ATCG_PYTHON_MODULE": "On",
        "ATCG_BUILD_EXAMPLES": "Off",
    },
    stubs_directory=VSCODE_STUBS_DIRECTORY,
    build_type="RelWithDebInfo",
    verbose=False,
    stubs_invalid_ok=True,
)

from _c_pyatcg import *  # type: ignore
