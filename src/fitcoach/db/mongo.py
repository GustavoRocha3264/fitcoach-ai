"""Mongo LandmarkStore: persists raw `(T, 33, 4)` landmark arrays per session.

Mongo is a poor fit for tabular numbers, but it shines for opaque blobs keyed
by an integer session_id. We serialize the float32 array as raw bytes plus a
small shape/dtype metadata so loads are trivially reversible.
"""
from __future__ import annotations

from typing import Any

import numpy as np


_EXPECTED_SHAPE_TAIL = (33, 4)


class LandmarkStore:
    """One document per session_id, holding the landmark tensor as bytes."""

    def __init__(self, db: Any, *, collection: str = "landmarks") -> None:
        self._coll = db[collection]
        # Idempotent index for cheap session_id lookups.
        self._coll.create_index("session_id", unique=True)

    def save_landmarks(self, *, session_id: int, landmarks: np.ndarray) -> None:
        if landmarks.ndim != 3 or landmarks.shape[1:] != _EXPECTED_SHAPE_TAIL:
            raise ValueError(
                f"landmarks must be shape (T, 33, 4), got {landmarks.shape}"
            )
        arr = np.ascontiguousarray(landmarks, dtype=np.float32)
        doc = {
            "session_id": session_id,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "data": arr.tobytes(),
        }
        # `replace_one(upsert=True)` makes save idempotent per session_id.
        self._coll.replace_one(
            {"session_id": session_id}, doc, upsert=True
        )

    def load_landmarks(self, *, session_id: int) -> np.ndarray:
        doc = self._coll.find_one({"session_id": session_id})
        if doc is None:
            raise LookupError(f"no landmarks for session {session_id}")
        arr = np.frombuffer(doc["data"], dtype=np.dtype(doc["dtype"]))
        return arr.reshape(tuple(doc["shape"])).copy()
