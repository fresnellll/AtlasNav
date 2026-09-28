import hashlib
import json
from pathlib import Path

from atlasnav.artifacts import verify_artifact_root


def test_verify_artifact_root(tmp_path: Path) -> None:
    value = b"atlasnav\n"
    (tmp_path / "asset.bin").write_bytes(value)
    (tmp_path / "release_manifest.json").write_text(
        json.dumps(
            {
                "schema": "atlasnav_artifact_release_v2",
                "files": [
                    {
                        "path": "asset.bin",
                        "bytes": len(value),
                        "sha256": hashlib.sha256(value).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert verify_artifact_root(tmp_path)["valid"] is True


def test_verify_artifact_root_detects_tamper(tmp_path: Path) -> None:
    (tmp_path / "asset.bin").write_bytes(b"wrong")
    (tmp_path / "release_manifest.json").write_text(
        json.dumps(
            {
                "schema": "atlasnav_artifact_release_v2",
                "files": [
                    {"path": "asset.bin", "bytes": 5, "sha256": "0" * 64}
                ],
            }
        ),
        encoding="utf-8",
    )
    report = verify_artifact_root(tmp_path)
    assert report["valid"] is False
    assert report["hash_mismatches"]


def test_deep_verification_scans_declared_text_files(tmp_path: Path) -> None:
    value = b'{"source":"DCI-Agent-Lite/outputs/portable/internal"}\n'
    (tmp_path / "record.json").write_bytes(value)
    files = [{
        "path": "record.json",
        "bytes": len(value),
        "sha256": hashlib.sha256(value).hexdigest(),
    }]
    (tmp_path / "MANIFEST.sha256.json").write_text(json.dumps(files), encoding="utf-8")
    (tmp_path / "release_manifest.json").write_text(
        json.dumps({
            "schema": "atlasnav_artifact_release_v2",
            "compatible_code": ">=0.2.0.dev0,<0.3",
            "files": files,
        }),
        encoding="utf-8",
    )
    report = verify_artifact_root(tmp_path, deep=True)
    assert report["valid"] is False
    assert report["deep"]["sensitive_findings"]["record.json"]["legacy_repository"] == 1
