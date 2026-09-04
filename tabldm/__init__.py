# Copyright (C) 2026 Xiaomi Corporation
# SPDX-License-Identifier: Apache-2.0
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
"""Xiaomi TabLDM — inference-only build.

This package contains only the default-mode inference path for the TabLDM
tabular foundation model: load a checkpoint and run ``fit`` / ``predict`` /
``predict_proba``. Training, prior-data generation, fine-tuning, forecasting,
unsupervised, and SHAP extras are intentionally excluded.

Public estimators are imported eagerly so that ``from tabldm import
TabLDMClassifier`` works out of the box.
"""

from ._model import InferenceConfig
from ._sklearn import (
    TabLDMClassifier,
    TabLDMRegressor,
)
from .__about__ import __version__

__all__ = [
    "TabLDMClassifier",
    "TabLDMRegressor",
    "InferenceConfig",
]
