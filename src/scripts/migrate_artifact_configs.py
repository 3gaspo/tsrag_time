#!/usr/bin/env python3
"""Migrate existing run manifests and selectors to the current artifact contract."""

from reconcile_artifact_contract import migrate_configs_main


if __name__ == "__main__":
    migrate_configs_main()
