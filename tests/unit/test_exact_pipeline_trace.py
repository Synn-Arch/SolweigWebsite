"""Check that correctness instrumentation preserves distinctions it promises."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


PATH = Path(__file__).resolve().parents[2] / "tools/capture_exact_pipeline.py"
SPEC = importlib.util.spec_from_file_location("exact_pipeline_trace", PATH)
trace = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trace)


def test_streamed_encoder_preserves_special_bits_and_strided_order():
    raw = np.array([0, 0x80000000, 0x7fc01234, 0x7f800000, 0xff800000, 0x3f800000], dtype=np.uint32)
    value = raw.view(np.float32).reshape(2, 3).T
    encoder = trace.ExactValueEncoder(np, chunk_bytes=8)
    record = encoder.encode(value)
    assert record["raw_sha256"] == hashlib.sha256(value.tobytes(order="C")).hexdigest()
    assert record["strides"] == list(value.strides)
    assert record["float_masks"]["negative_zero"]["count"] == 1
    assert record["float_masks"]["nan"]["count"] == 1
    assert record["float_masks"]["posinf"]["count"] == 1
    assert record["float_masks"]["neginf"]["count"] == 1
    assert all(chunk.nbytes <= 8 for chunk in encoder._iter_array_chunks(value))
    copied = encoder.encode(value.copy())
    assert copied["raw_sha256"] == record["raw_sha256"]
    assert copied["strides"] != record["strides"]
    changed = value.copy()
    changed[0, 0] = -0.
    assert encoder.encode(changed)["raw_sha256"] != record["raw_sha256"]


def test_scalar_container_types_and_empty_arrays_are_preserved():
    encoder = trace.ExactValueEncoder(np, chunk_bytes=16)
    assert encoder.encode(1)["type"] != encoder.encode(1.)["type"]
    assert encoder.encode(np.float32(1))["type"] != encoder.encode(np.float64(1))["type"]
    assert encoder.encode([])["type"] != encoder.encode(())["type"]
    assert encoder.encode(-0.)["binary64_be_hex"] != encoder.encode(0.)["binary64_be_hex"]
    assert encoder.encode(np.empty((0, 3), dtype=np.float32))["size"] == 0
    with pytest.raises(TypeError, match="object arrays"):
        encoder.encode(np.array([object()], dtype=object))


def test_capture_hashes_immediately_without_retaining_mutable_fields(tmp_path):
    array = np.arange(10, dtype=np.float32)
    path = tmp_path / "records.jsonl"
    sink = trace.TraceSink(path, trace.ExactValueEncoder(np, chunk_bytes=8))
    sink.capture_fields("state_checkpoint", 0, (array,), ["Tgmap1"], [array])
    array[0] = -1
    sink.capture_fields("state_checkpoint", 1, (array,), ["Tgmap1"], [array])
    summary = sink.finish()
    first, second = map(json.loads, path.read_text().splitlines())
    assert first["fields"][0]["value"]["raw_sha256"] != second["fields"][0]["value"]["raw_sha256"]
    assert second["previous_payload_sha256"] == first["payload_sha256"]
    assert summary["records"] == 2
    assert summary["jsonl_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
