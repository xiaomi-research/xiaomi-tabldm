Tutorials
=========

A collection of runnable examples demonstrating typical TabLDM workflows.

Before running the examples, install the project in editable mode from the
repository root::

    pip install -e .

The examples use ``cpu`` by default to make them portable. Set ``device="cuda"``
to run on a GPU, or use ``CUDA_VISIBLE_DEVICES`` to select a specific card.

Examples
--------

===============================  =====================================================
Example                          Demonstrates
===============================  =====================================================
``getting_started.py``           Basic classification, regression, and cross-validation
``mixed_data_types.py``          Numeric, categorical, boolean, and missing values
``classification_metrics.py``    Labels, probabilities, and classification metrics
``regression_quantiles.py``      Mean predictions and predictive quantiles
``kv_cache.py``                  Faster repeated prediction with one training context
``local_checkpoint.py``          Local checkpoint loading and CPU/CUDA selection
===============================  =====================================================

Checkpoint locations
--------------------

By default, TabLDM downloads the configured checkpoint from Hugging Face Hub on
the first ``fit()`` call. To run fully offline, pass an existing checkpoint file
to ``model_path``. The default Hub checkpoint paths are::

    checkpoints/clf_stage3_moe1_step-10000.ckpt
    checkpoints/reg_stage3_moe1_step-10000.ckpt

The examples also support local checkpoint overrides. When set, these
environment variables take precedence over the default Hub checkpoints::

    export TABLDM_CLF_CKPT=/path/to/classifier.ckpt
    export TABLDM_REG_CKPT=/path/to/regressor.ckpt
