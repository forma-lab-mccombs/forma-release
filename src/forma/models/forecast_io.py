"""Canonical forecast-write contract, shared by every forecast producer.

These symbols live here (a low-level, dependency-free module) rather than in
``train.py`` so that *both* ``train.py`` and ``models/baselines.py`` can import
them. ``train.py`` imports ``baselines``, so the constants cannot live in
``train.py`` without forcing ``baselines`` into a circular import -- which is why
the streaming baseline writer historically diverged onto a non-canonical path
(float64, fastparquet-append, no finalize). Keep the contract here and every
writer can share it.

The contract: ``evaluate.py`` reads only ``FORECAST_SCHEMA_COLS`` (``sigma``
optional); anything else a model emits is diagnostic and just bloats the parquet.
The float payload is stored float32 -- forecasts live in the asinh target space
where R2/MAE/NLL resolve ~3-4 sig figs, so float32 (~7 sig figs) is lossless for
our metrics and halves the float64 baseline columns.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import pandas as pd

# evaluate.py reads firm_id/target/quarter/forecast_horizon/prediction (+sigma).
FORECAST_SCHEMA_COLS = ['firm_id', 'target', 'quarter', 'forecast_horizon',
                        'prediction', 'sigma', 'model']
# The float payload columns, downcast to float32 before writing.
FORECAST_FLOAT_COLS = ('prediction', 'sigma')
# Parquet codec for forecast files. zstd shaves ~10% off the (high-entropy float32)
# payload vs snappy at no read cost; fastparquet (evaluate.py's reader) reads it fine.
FORECAST_PARQUET_COMPRESSION = 'zstd'


def finalize_forecast_frame(forecasts_df: pd.DataFrame) -> pd.DataFrame:
    """Enforce the canonical forecast schema before writing: drop non-canonical
    (diagnostic) columns and downcast prediction/sigma to float32.

    Idempotent: it only re-applies the same column selection and float32 cast, so
    a second call (e.g. on an already-finalized or float32-round-tripped frame) is
    a harmless no-op re-cast -- it never re-derives or recomputes anything.
    """
    keep = [c for c in FORECAST_SCHEMA_COLS if c in forecasts_df.columns]
    extra = [c for c in forecasts_df.columns if c not in keep]
    if extra:
        print(f"  Forecast schema: dropping {len(extra)} non-canonical column(s): {extra}")
    out = forecasts_df[keep].copy()
    for col in FORECAST_FLOAT_COLS:
        if col in out.columns:
            out[col] = out[col].astype('float32')
    return out


@contextmanager
def atomic_parquet_writers(final_paths, compression=FORECAST_PARQUET_COMPRESSION):
    """Multi-path, lazily-opened, crash-safe parquet writer for chunked forecasts.

    Both streaming forecast producers -- the baseline target-group path in
    ``models/baselines.py`` and the Lightning row-chunk path
    (``train._stream_save_forecasts``) -- write a forecast incrementally across
    many ``write_table`` calls. Each used to carry its own near-identical .tmp +
    ``os.replace`` + ``try/finally`` cleanup scaffolding; this shares it so the two
    can't drift.

    Yields a ``write_table(pyarrow.Table)`` callable. Each output path is written
    to a sibling ``<path>.tmp`` and ``os.replace``d into the canonical path only
    after the body finishes cleanly and the writer is closed. A crash mid-stream
    leaves a stray ``.tmp`` (removed here on the way out), never a truncated parquet
    at the canonical path that a later ``evaluate.py`` would silently read as a
    complete forecast.

    Writers open lazily on the first ``write_table`` (so the parquet schema is
    taken from the first table written, and a body that writes nothing -- e.g. a
    zero-row stream -- produces no file at all, matching the pre-streaming
    behavior). Uses pyarrow's ``ParquetWriter`` deliberately: do NOT swap it for a
    fastparquet append-writer to match ``evaluate.py``'s read engine. pyarrow auto
    RLE_DICTIONARY-encodes low-cardinality float columns (Forma's ~2%-unique
    preds); fastparquet's append path does not, inflating Forma forecasts ~33%
    (benchmarked). pyarrow-zstd round-trips through fastparquet cleanly (the
    read-side invariant), so this is safe.
    """
    # pyarrow import is function-local so this module stays importable (and
    # lightweight) without pyarrow installed -- mirrors how the callers import it.
    import pyarrow.parquet as pq

    final_paths = [str(p) for p in final_paths]
    tmp_paths = [p + '.tmp' for p in final_paths]
    writers = [None] * len(final_paths)

    def write_table(tbl):
        for j, tmp in enumerate(tmp_paths):
            if writers[j] is None:
                writers[j] = pq.ParquetWriter(tmp, tbl.schema, compression=compression)
            writers[j].write_table(tbl)

    try:
        yield write_table
        # Body completed cleanly: close each open writer and atomically move its
        # .tmp into place. Gated on the writer being open, so a stream that wrote
        # nothing leaves writers all None here and produces no file.
        for j, w in enumerate(writers):
            if w is not None:
                w.close()
                writers[j] = None
                os.replace(tmp_paths[j], final_paths[j])
    finally:
        # On the clean path the writers are already closed (None) and the .tmp
        # files already moved, so this is a no-op; it only does work on failure:
        # close any still-open writer and remove the partial .tmp it left behind.
        for w in writers:
            if w is not None:
                try:
                    w.close()
                except Exception:
                    pass
        for tmp in tmp_paths:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
