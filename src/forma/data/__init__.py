"""Forma data package.

Intentionally EMPTY of imports. ``config_utils``, ``transforms`` and
``quarters`` are torch-free; importing them (e.g. ``from forma.data.config_utils
import resolve_dataset_path``) must NOT drag in ``forma_data`` and therefore
torch / lightning / torch_geometric. The scoring slice
(``forma.scoring.evaluate``) relies on this so it can resolve dataset paths
without the model stack installed. Import ``forma_data`` explicitly where the
model path actually needs it.
"""
