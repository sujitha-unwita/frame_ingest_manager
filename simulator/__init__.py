"""Simulated MCU, for development with no hardware attached.

Selected by `transport.kind: mock` in config.yaml.

NOT named `mock/`: a package called `mock` exists on PyPI, and if it were
ever installed in the environment Python could load that one instead of this
folder -- a shadowing bug that is unpleasant to diagnose.
"""
