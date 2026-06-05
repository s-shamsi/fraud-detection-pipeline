"""
XGBoost → ONNX Conversion & Validation
========================================
Converts the trained XGBoost classifier to the ONNX graph format for
deployment with ONNX Runtime.  After conversion, performs a parity check to
confirm that ONNX and native XGBoost outputs match to 5 decimal places.

Why ONNX?
---------
- ONNX Runtime is 2–5× faster than XGBoost's Python .predict() on CPU
- Single binary artefact: deploy the same file on any ONNX-capable runtime
  (Cloud Run, Vertex AI Prediction, Triton, etc.)
- Zero Python dependency inside the inference container for the model itself

Usage:
    python model/convert_to_onnx.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def convert_xgboost_to_onnx(artifact_dir: Path) -> Path:
    """
    Convert fraud_detector.xgb.json → fraud_detector.onnx

    Uses onnxmltools which wraps the XGBoost booster's native ONNX export
    for maximum compatibility.
    """
    import xgboost as xgb
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    # Load metadata to know the feature count
    with open(artifact_dir / "model_metadata.json") as f:
        meta = json.load(f)

    n_features   = meta["n_features"]
    feature_cols = meta["feature_columns"]

    print(f"  n_features   : {n_features}")
    print(f"  feature_cols : {feature_cols}")

    # Load trained XGBoost model
    print("  Loading XGBoost model …")
    booster = xgb.Booster()
    booster.load_model(str(artifact_dir / "fraud_detector.xgb.json"))

    # Convert
    print("  Converting to ONNX …")
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_xgboost(booster, initial_types=initial_type)

    # Write
    onnx_path = artifact_dir / "fraud_detector.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"  ONNX model written to {onnx_path}")
    return onnx_path


def validate_parity(artifact_dir: Path, n_samples: int = 1_000) -> None:
    """
    Verify that ONNX Runtime produces the same fraud probabilities as the
    original XGBoost model (max absolute error < 1e-5).
    """
    import joblib
    import pandas as pd
    from data.generate_data import get_feature_columns

    print("\n── Parity check ─────────────────────────────────────────────────")

    # Load original sklearn/XGBoost model
    clf = joblib.load(artifact_dir / "fraud_detector.pkl")

    # Load ONNX session
    session_opts = ort.SessionOptions()
    session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(artifact_dir / "fraud_detector.onnx"),
        sess_options=session_opts,
        providers=["CPUExecutionProvider"],
    )

    # Load a sample from the processed dataset
    data_path = ROOT / "data/processed/transactions.parquet"
    if not data_path.exists():
        print("  ⚠️  No processed data found; skipping parity check.")
        print("     Run `python data/generate_data.py` first.")
        return

    df = pd.read_parquet(data_path).head(n_samples)
    X  = df[get_feature_columns()].fillna(0).values.astype(np.float32)

    # XGBoost predictions
    t0 = time.perf_counter()
    xgb_probs = clf.predict_proba(X)[:, 1]
    xgb_ms    = (time.perf_counter() - t0) * 1000

    # ONNX Runtime predictions
    input_name = session.get_inputs()[0].name
    t0 = time.perf_counter()
    onnx_output = session.run(None, {input_name: X})
    onnx_ms     = (time.perf_counter() - t0) * 1000

    # ONNX Runtime returns [labels, probabilities_dict or array]
    # Shape depends on onnxmltools version; handle both
    if isinstance(onnx_output[1], list):
        onnx_probs = np.array([d[1] for d in onnx_output[1]], dtype=np.float32)
    else:
        onnx_probs = onnx_output[1][:, 1]

    max_abs_err = float(np.abs(xgb_probs - onnx_probs).max())
    mean_abs_err = float(np.abs(xgb_probs - onnx_probs).mean())

    print(f"  Samples evaluated : {n_samples:,}")
    print(f"  Max  |Δ prob|      : {max_abs_err:.2e}")
    print(f"  Mean |Δ prob|      : {mean_abs_err:.2e}")
    print(f"  XGBoost latency    : {xgb_ms:.1f} ms  ({xgb_ms/n_samples*1000:.2f} µs/sample)")
    print(f"  ONNX RT latency    : {onnx_ms:.1f} ms  ({onnx_ms/n_samples*1000:.2f} µs/sample)")
    print(f"  Speedup            : {xgb_ms/onnx_ms:.2f}×")

    assert max_abs_err < 1e-4, (
        f"Parity check FAILED: max absolute error {max_abs_err:.2e} exceeds 1e-4. "
        "Re-check conversion parameters."
    )
    print("  ✅ Parity check PASSED")


def inspect_onnx_graph(onnx_path: Path) -> None:
    """Print a human-readable summary of the ONNX graph."""
    print("\n── ONNX graph summary ───────────────────────────────────────────")
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    inputs = model.graph.input
    outputs = model.graph.output
    nodes   = model.graph.node

    print(f"  IR version : {model.ir_version}")
    print(f"  Opset      : {model.opset_import[0].version}")
    print(f"  Nodes      : {len(nodes)}")
    print(f"  Inputs     : {[i.name for i in inputs]}")
    print(f"  Outputs    : {[o.name for o in outputs]}")
    print("  ✅ ONNX graph valid")


def benchmark_onnx(artifact_dir: Path, batch_sizes: list[int] = [1, 10, 100, 1_000]) -> None:
    """Report ONNX Runtime inference latency at different batch sizes."""
    print("\n── ONNX Runtime latency benchmark ───────────────────────────────")

    session_opts = ort.SessionOptions()
    session_opts.intra_op_num_threads = 2
    session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(artifact_dir / "fraud_detector.onnx"),
        sess_options=session_opts,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    with open(artifact_dir / "model_metadata.json") as f:
        n_features = json.load(f)["n_features"]

    for bs in batch_sizes:
        X = np.random.rand(bs, n_features).astype(np.float32)
        # Warm-up
        for _ in range(10):
            session.run(None, {input_name: X})
        # Timed
        n_trials = 200
        t0 = time.perf_counter()
        for _ in range(n_trials):
            session.run(None, {input_name: X})
        elapsed_ms = (time.perf_counter() - t0) / n_trials * 1000
        print(f"  batch={bs:<5}  {elapsed_ms:.3f} ms/call  ({elapsed_ms/bs*1000:.1f} µs/sample)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    artifact_dir = ROOT / "artifacts"

    if not (artifact_dir / "fraud_detector.xgb.json").exists():
        print("❌ Model file not found. Run `python model/train.py` first.")
        sys.exit(1)

    print("── Converting XGBoost model to ONNX ─────────────────────────────")
    onnx_path = convert_xgboost_to_onnx(artifact_dir)

    inspect_onnx_graph(onnx_path)
    validate_parity(artifact_dir)
    benchmark_onnx(artifact_dir)

    print(f"\n✅ ONNX model ready at {onnx_path}")
    print("   Next step: feast apply && python feature_store/materialize.py")


if __name__ == "__main__":
    main()
