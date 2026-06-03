"""Tests for COLMAP feature-matching strategy dispatch.

Regression guard for the bug where selecting "Vocab Tree" (or any matcher other
than "sequential") silently ran ``exhaustive_matcher``. ``feature_matching`` must
map every ``matcher_type`` to the matching COLMAP subcommand, and a Vocab Tree
run with no available vocabulary tree must fail visibly instead of falling back.

Runnable directly (``python tests/test_feature_matching.py``) or via pytest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.engine import ColmapEngine
from app.core.params import ColmapParams


def _make_engine(matcher_type):
    """Build an engine whose run_command is stubbed to capture the command."""
    params = ColmapParams(matcher_type=matcher_type)
    engine = ColmapEngine(
        params, "/tmp/in", "/tmp/out", "images", 5, "proj",
        logger_callback=lambda _msg: None,
    )
    captured = {}

    def fake_run(cmd, description, status_prefix=None):
        captured["cmd"] = cmd
        captured["description"] = description
        captured["ran"] = True
        return True

    engine.run_command = fake_run
    captured["ran"] = False
    return engine, captured


def test_exhaustive_uses_exhaustive_matcher():
    engine, captured = _make_engine("exhaustive")
    assert engine.feature_matching("/tmp/db.db") is True
    assert captured["cmd"][1] == "exhaustive_matcher"


def test_sequential_uses_sequential_matcher():
    engine, captured = _make_engine("sequential")
    assert engine.feature_matching("/tmp/db.db") is True
    cmd = captured["cmd"]
    assert cmd[1] == "sequential_matcher"
    # Sequential-specific options must still be present.
    assert "--SequentialMatching.overlap" in cmd
    assert "--SequentialMatching.quadratic_overlap" in cmd


def test_vocab_tree_uses_vocab_tree_matcher():
    engine, captured = _make_engine("vocab_tree")
    engine._ensure_vocab_tree = lambda: Path("/fake/engines/vocab.bin")
    assert engine.feature_matching("/tmp/db.db") is True
    cmd = captured["cmd"]
    assert cmd[1] == "vocab_tree_matcher"
    idx = cmd.index("--VocabTreeMatching.vocab_tree_path")
    assert cmd[idx + 1] == "/fake/engines/vocab.bin"


def test_vocab_tree_without_tree_fails_visibly():
    """Missing vocab tree must NOT silently fall back to exhaustive matching."""
    engine, captured = _make_engine("vocab_tree")
    engine._ensure_vocab_tree = lambda: None
    assert engine.feature_matching("/tmp/db.db") is False
    assert captured["ran"] is False  # run_command was never reached


def test_unknown_matcher_falls_back_to_exhaustive():
    engine, captured = _make_engine("does_not_exist")
    assert engine.feature_matching("/tmp/db.db") is True
    assert captured["cmd"][1] == "exhaustive_matcher"


def test_common_matching_args_shared_across_matchers():
    """All matchers must carry the shared Sift/FeatureMatching arguments."""
    shared = [
        "--database_path",
        "--FeatureMatching.num_threads",
        "--SiftMatching.max_ratio",
        "--SiftMatching.max_distance",
        "--SiftMatching.cross_check",
        "--FeatureMatching.guided_matching",
    ]
    for matcher in ("exhaustive", "sequential", "vocab_tree"):
        engine, captured = _make_engine(matcher)
        engine._ensure_vocab_tree = lambda: Path("/fake/vocab.bin")
        engine.feature_matching("/tmp/db.db")
        for arg in shared:
            assert arg in captured["cmd"], f"{arg} missing for {matcher}"


def test_ensure_vocab_tree_returns_existing_file(tmp_path, monkeypatch):
    """An existing vocab tree in engines/ is reused without any download."""
    import app.core.engine as engine_mod

    engines_dir = tmp_path / "engines"
    engines_dir.mkdir()
    vocab = engines_dir / engine_mod._VOCAB_TREE_FILENAME
    vocab.write_bytes(b"not empty")

    monkeypatch.setattr(engine_mod, "resolve_project_root", lambda: tmp_path)

    engine, _ = _make_engine("vocab_tree")
    assert engine._ensure_vocab_tree() == vocab


if __name__ == "__main__":
    # Minimal standalone runner so the suite works without pytest installed.
    import tempfile
    import types

    failures = 0
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        argcount = fn.__code__.co_argcount
        try:
            if argcount == 0:
                fn()
            else:
                # Provide tmp_path + a tiny monkeypatch shim for the fixture-based test.
                with tempfile.TemporaryDirectory() as d:
                    patches = []

                    class _MP:
                        def setattr(self, obj, name, value):
                            patches.append((obj, name, getattr(obj, name)))
                            setattr(obj, name, value)

                    try:
                        fn(Path(d), _MP())
                    finally:
                        for obj, name, old in reversed(patches):
                            setattr(obj, name, old)
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"ERROR {fn.__name__}: {e}")

    sys.exit(1 if failures else 0)
