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

import os

headless = os.getenv("ATCG_HEADLESS")
cuda = os.getenv("ATCG_CUDA_BACKEND")
build_type = os.getenv("ATCG_BUILD_TYPE")
stubs_dir = os.getenv("ATCG_STUBS_DIR")
verbose = os.getenv("ATCG_VERBOSE")

if not headless:
    headless = "Off"

if not cuda:
    cuda = "On"

if not build_type:
    build_type = "RelWithDebInfo"

if not stubs_dir:
    stubs_dir = PROJECT_ROOT_DIRECTORY / "typings"

if not verbose:
    verbose = False

del os


charonload.module_config["_c_pyatcg"] = charonload.Config(
    # All paths must be absolute
    project_directory=PROJECT_ROOT_DIRECTORY,
    cmake_options={
        "ATCG_PYTHON_MODULE": "On",
        "ATCG_BUILD_EXAMPLES": "Off",
        "ATCG_CUDA_BACKEND": cuda,
        "ATCG_HEADLESS": headless,
    },
    stubs_directory=stubs_dir,
    build_type=build_type,
    verbose=verbose,
    stubs_invalid_ok=True,
)

from _c_pyatcg import *  # type: ignore
