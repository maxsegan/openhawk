import pytest

from cv.pipeline.resolution import NATIVE_SIZE, coordinate_columns_and_size


@pytest.mark.parametrize(
    "columns,manifest,error",
    [
        (["x_native", "x", "y"], {"artifact_size": {"width": 1920, "height": 1080}}, "partial"),
        (
            ["x_native", "y_native"],
            {"artifact_size": {"width": 960, "height": 540}},
            "native coordinate",
        ),
        (
            ["x", "y"],
            {
                "artifact_size": {"width": 1920, "height": 1080},
                "interface_space": "native_1920x1080",
            },
            "no declared legacy",
        ),
        (["frame"], {"artifact_size": {"width": 1920, "height": 1080}}, "missing coordinate"),
    ],
)
def test_inconsistent_coordinate_declarations_fail(columns, manifest, error):
    with pytest.raises(ValueError, match=error):
        coordinate_columns_and_size(
            columns, manifest, native_columns=("x_native", "y_native"), legacy_columns=("x", "y")
        )


def test_undecorated_columns_can_be_declared_native():
    columns, size = coordinate_columns_and_size(
        ["x", "y"],
        {"artifact_size": {"width": 1920, "height": 1080}},
        native_columns=("x_native", "y_native"),
        legacy_columns=("x", "y"),
    )
    assert columns == ("x", "y")
    assert size == NATIVE_SIZE
