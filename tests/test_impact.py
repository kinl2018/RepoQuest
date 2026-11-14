import pytest
from call_analysis import build_call_index
from impact_analysis import analyze_impact


def test_transitive_impact_includes_aliases_and_evidence(sample_repo):
    result = analyze_impact(sample_repo, "lib.py", "leaf", max_depth=3)
    affected = {(a["file"], a["symbol"]): a for a in result["affected"]}
    assert affected[("lib.py", "middle")]["direct"]
    assert affected[("app.py", "entry")]["depth"] == 2
    assert affected[("app.py", "entry")]["path"] == ["app.py::entry", "lib.py::middle", "lib.py::leaf"]
    assert any(s["file"] == "app.py" and s["line"] == 4 for s in result["call_sites"])
    assert ("other.py", "Worker.work") not in affected
    assert "app.py" in result["module_importers"]


def test_methods_and_relative_imports(sample_repo):
    package = sample_repo / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "a.py").write_text("def target(): pass\n")
    (package / "b.py").write_text("from .a import target as go\ndef main(): go()\n")
    result = analyze_impact(sample_repo, "pkg/a.py", "target")
    assert any(a["file"] == "pkg/b.py" and a["symbol"] == "main" for a in result["affected"])
    methods = analyze_impact(sample_repo, "other.py", "Worker.leaf")
    assert any(a["symbol"] == "Worker.work" for a in methods["affected"])


def test_cycles_do_not_revisit_target_and_depth_is_observable(sample_repo):
    (sample_repo / "cycle.py").write_text("def a(): b()\ndef b(): a()\ndef c(): b()\n")
    result = analyze_impact(sample_repo, "cycle.py", "a", max_depth=1)
    assert result["truncated"]
    assert [a["symbol"] for a in result["affected"]] == ["b"]
    result = analyze_impact(sample_repo, "cycle.py", "a", max_depth=10)
    assert {a["symbol"] for a in result["affected"]} == {"b", "c"}


def test_parameter_shadows_global_and_broken_files_reported(sample_repo):
    (sample_repo / "shadow.py").write_text("from lib import leaf\ndef f(leaf): leaf()\n")
    (sample_repo / "broken.py").write_text("def broken(")
    result = analyze_impact(sample_repo, "lib.py", "leaf")
    assert not any(a["file"] == "shadow.py" for a in result["affected"])
    assert "broken.py" in result["parse_errors"]


def test_bare_name_in_method_does_not_resolve_as_method(sample_repo):
    (sample_repo / "methods.py").write_text("class A:\n    def leaf(self): pass\n    def wrong(self): leaf()\n    def right(self): self.leaf()\n")
    result = analyze_impact(sample_repo, "methods.py", "A.leaf")
    assert {a["symbol"] for a in result["affected"]} == {"A.right"}


def test_dotted_module_import(sample_repo):
    pkg = sample_repo / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "a.py").write_text("def target(): pass\n")
    (sample_repo / "use.py").write_text("import pkg.a\ndef main(): pkg.a.target()\n")
    result = analyze_impact(sample_repo, "pkg/a.py", "target")
    assert any(a["symbol"] == "main" for a in result["affected"])


@pytest.mark.parametrize("file,symbol,depth", [("../outside.py", "leaf", 3), ("lib.py", "missing", 3), ("lib.py", "leaf", 0)])
def test_invalid_impact_target(sample_repo, file, symbol, depth):
    with pytest.raises(ValueError):
        analyze_impact(sample_repo, file, symbol, depth)
