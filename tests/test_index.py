import numpy as np
import os
import subprocess
import pytest
from conftest import make_doc
from rag_index import save_index, load_index, assert_current_index


def test_roundtrip_and_model_source_provenance(sample_repo):
    index_dir = sample_repo / ".repoquest_index"
    save_index([make_doc("leaf")], [[1,2]], index_dir, root=sample_repo, model_name="test")
    docs, vectors = load_index(index_dir)
    assert vectors.shape == (1,2)
    assert docs[0].metadata["name"] == "leaf"
    assert_current_index(sample_repo, index_dir, "test")
    with pytest.raises(ValueError, match="模型"):
        assert_current_index(sample_repo, index_dir, "different")
    (sample_repo / "lib.py").write_text("def changed(): pass")
    with pytest.raises(ValueError, match="变化"):
        assert_current_index(sample_repo, index_dir, "test")


def test_corrupt_index_fails_checksum(tmp_path):
    save_index([make_doc("leaf")], [[1,2]], tmp_path)
    (tmp_path / "documents.json").write_text("[]")
    with pytest.raises(ValueError, match="校验"):
        load_index(tmp_path)


@pytest.mark.parametrize("matrix", [[], [[1, float('nan')]], [[1],[2]]])
def test_bad_index_shapes_fail(tmp_path, matrix):
    with pytest.raises(ValueError):
        save_index([make_doc("leaf")], matrix, tmp_path)


@pytest.mark.skipif(os.name != "nt", reason="Windows DACL regression")
def test_saved_index_inherits_repository_permissions(tmp_path):
    save_index([make_doc("leaf")], [[1, 2]], tmp_path)
    # Protected DACLs from TemporaryDirectory survive a move. The final files
    # must remain inheritable so another normal repository user can read them.
    script = """
$ErrorActionPreference = 'Stop'
foreach ($name in @('manifest.json', 'documents.json', 'embeddings.npy')) {
    $acl = Get-Acl -LiteralPath (Join-Path $args[0] $name)
    if ($acl.AreAccessRulesProtected) { throw "$name has a protected DACL" }
    if (-not ($acl.Access | Where-Object { $_.IsInherited })) {
        throw "$name has no inherited access rules"
    }
}
"""
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                    "& { " + script + " }", str(tmp_path)], check=True, capture_output=True, text=True)


def test_staging_files_are_removed_on_save_failure(tmp_path, monkeypatch):
    import rag_index
    def fail_replace(*args):
        raise OSError("simulated commit failure")
    monkeypatch.setattr(rag_index.os, "replace", fail_replace)
    with pytest.raises(OSError, match="commit failure"):
        save_index([make_doc("leaf")], [[1, 2]], tmp_path)
    assert list(tmp_path.iterdir()) == []
