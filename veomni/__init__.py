# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from .utils.import_utils import (
    is_veomni_patch_available,
)
from .utils.logging import get_logger
import torch
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version
from packaging.version import parse as parse_version

logger = get_logger(__name__)


def _safe_apply_patches():
    if is_veomni_patch_available():
        from veomni_patch import apply_patch

        apply_patch()
        logger.info_rank0("✅ veomni_patch is available")
    else:
        logger.info_rank0("❌ veomni_patch is not available")


_safe_apply_patches()

__version__ = "v0.1.0"


def is_torch_npu_available() -> bool:
    """Check the availability of NPU"""
    try:
        import torch_npu  # noqa: F401
        return torch.npu.is_available()
    except ImportError:
        return False
    
    
is_npu_available = is_torch_npu_available()
if is_npu_available:
    from .models.transformers import npu_patch as npu_patch
    package_name = "transformers"
    required_version_spec = "4.50.4"
    try:
        installed_version = get_version(package_name)
        installed = parse_version(installed_version)
        required = parse_version(required_version_spec)
        if installed < required:
            raise ValueError(
                f"{package_name} version >= {required_version_spec} is required on ASCEND NPU, current version is "
                f"{installed}."
            )
    except PackageNotFoundError as e:
        raise ImportError(
            f"package {package_name} is not installed, please run pip install {package_name}=={required_version_spec}"
        ) from e