import json
from pathlib import Path

from cv.pipeline import canonical_runner

def test_every_pipeline_module_has_an_explicit_status() -> None:
    pipeline = Path(__file__).resolve().parent
    registry = json.loads((pipeline / "implementations.json").read_text())
    modules = registry["modules"]
    expected = {
        path.name
        for path in pipeline.glob("*.py")
        if path.name != "__init__.py" and not path.name.startswith("test_")
    }

    assert set(modules) == expected
    assert all(record["status"] in registry["statuses"] for record in modules.values())


def test_canonical_stage_registry_has_no_duplicate_capabilities() -> None:
    capabilities = [name for name, _ in canonical_runner.CANONICAL_STAGE_REGISTRY]

    assert len(capabilities) == len(set(capabilities))


def test_canonical_artifacts_have_a_downstream_consumer() -> None:
    pipeline = Path(__file__).resolve().parent
    lineage = json.loads((pipeline / "artifact_lineage.json").read_text())

    assert all(record["consumed_by"] for record in lineage["artifacts"].values())
    assert all("legacy" not in name for name in lineage["artifacts"])
