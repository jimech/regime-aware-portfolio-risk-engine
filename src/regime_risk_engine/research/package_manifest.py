import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


class AdvancedPackageManifestError(ValueError):
    """Raised when an advanced research package manifest cannot be inspected."""


@dataclass(frozen=True, slots=True)
class AdvancedPackageManifestInspection:
    """Validated advanced research package manifest inspection result."""

    package_dir: Path
    manifest_path: Path
    memo_path: Path
    table_paths: dict[str, Path]

    def to_dict(self) -> dict[str, object]:
        return {
            "package_dir": str(self.package_dir),
            "manifest_path": str(self.manifest_path),
            "memo_path": str(self.memo_path),
            "tables": {
                name: str(path) for name, path in sorted(self.table_paths.items())
            },
        }


@dataclass(frozen=True, slots=True)
class AdvancedResearchPackage:
    """Loaded advanced research package contents."""

    package_dir: Path
    manifest_path: Path
    memo: str
    tables: dict[str, pd.DataFrame]

    def table_names(self) -> list[str]:
        return sorted(self.tables)


def inspect_advanced_package_manifest(
    package_dir: str | Path,
) -> AdvancedPackageManifestInspection:
    """Inspect and validate an advanced research package manifest."""
    clean_package_dir = Path(package_dir).expanduser().resolve()

    if not clean_package_dir.exists():
        raise AdvancedPackageManifestError(
            f"Package directory does not exist: {clean_package_dir}"
        )

    if not clean_package_dir.is_dir():
        raise AdvancedPackageManifestError(
            f"Package path is not a directory: {clean_package_dir}"
        )

    manifest_path = clean_package_dir / "manifest.json"

    if not manifest_path.exists():
        raise AdvancedPackageManifestError(
            f"Manifest file does not exist: {manifest_path}"
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AdvancedPackageManifestError(
            f"Manifest file is not valid JSON: {manifest_path}"
        ) from exc

    memo_name = manifest.get("memo")
    tables = manifest.get("tables")

    if not isinstance(memo_name, str) or not memo_name:
        raise AdvancedPackageManifestError(
            "Manifest must contain a non-empty memo field"
        )

    if Path(memo_name).is_absolute():
        raise AdvancedPackageManifestError("Manifest memo path must be relative")

    if not isinstance(tables, dict) or not tables:
        raise AdvancedPackageManifestError(
            "Manifest must contain a non-empty tables mapping"
        )

    memo_path = clean_package_dir / memo_name

    if not memo_path.exists():
        raise AdvancedPackageManifestError(
            f"Manifest memo file does not exist: {memo_name}"
        )

    table_paths: dict[str, Path] = {}

    for name, filename in tables.items():
        if not isinstance(name, str) or not name:
            raise AdvancedPackageManifestError(
                "Manifest table names must be non-empty strings"
            )

        if not isinstance(filename, str) or not filename:
            raise AdvancedPackageManifestError(
                f"Manifest table filename for {name} must be a non-empty string"
            )

        if Path(filename).is_absolute():
            raise AdvancedPackageManifestError(
                f"Manifest table path for {name} must be relative"
            )

        table_path = clean_package_dir / filename

        if not table_path.exists():
            raise AdvancedPackageManifestError(
                f"Manifest table file does not exist for {name}: {filename}"
            )

        table_paths[name] = table_path

    return AdvancedPackageManifestInspection(
        package_dir=clean_package_dir,
        manifest_path=manifest_path,
        memo_path=memo_path,
        table_paths=table_paths,
    )


@dataclass(frozen=True, slots=True)
class AdvancedResearchPackageSummary:
    """Dashboard-ready summary of a loaded advanced research package."""

    memo_title: str
    table_count: int
    table_names: list[str]
    has_factor_significance: bool
    has_rolling_factor_exposure: bool
    has_scenario_simulation: bool
    has_stress_test: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "memo_title": self.memo_title,
            "table_count": self.table_count,
            "table_names": self.table_names,
            "has_factor_significance": self.has_factor_significance,
            "has_rolling_factor_exposure": self.has_rolling_factor_exposure,
            "has_scenario_simulation": self.has_scenario_simulation,
            "has_stress_test": self.has_stress_test,
        }


def format_advanced_package_manifest_inspection(
    inspection: AdvancedPackageManifestInspection,
) -> str:
    """Format an advanced package manifest inspection for CLI output."""
    lines = [
        "Advanced research package inspection",
        f"Package directory: {inspection.package_dir}",
        f"Manifest: {inspection.manifest_path}",
        f"Memo: {inspection.memo_path}",
        "Tables:",
    ]

    for name, path in sorted(inspection.table_paths.items()):
        lines.append(f"- {name}: {path}")

    return "\n".join(lines)


def load_advanced_research_package(
    package_dir: str | Path,
) -> AdvancedResearchPackage:
    """Load an advanced research package from its manifest."""
    inspection = inspect_advanced_package_manifest(package_dir)

    memo = inspection.memo_path.read_text(encoding="utf-8")
    tables = {
        name: pd.read_csv(path) for name, path in sorted(inspection.table_paths.items())
    }

    return AdvancedResearchPackage(
        package_dir=inspection.package_dir,
        manifest_path=inspection.manifest_path,
        memo=memo,
        tables=tables,
    )


def summarize_advanced_research_package(
    package: AdvancedResearchPackage,
) -> AdvancedResearchPackageSummary:
    """Build a dashboard-ready summary of a loaded advanced research package."""
    table_names = package.table_names()

    return AdvancedResearchPackageSummary(
        memo_title=_extract_markdown_title(package.memo),
        table_count=len(table_names),
        table_names=table_names,
        has_factor_significance="factor_significance" in package.tables,
        has_rolling_factor_exposure=(
            "rolling_factor_exposures" in package.tables
            or "rolling_factor_exposure_summary" in package.tables
        ),
        has_scenario_simulation=any(
            name.startswith("scenario_") for name in package.tables
        ),
        has_stress_test="stress_test_summary" in package.tables,
    )


def _extract_markdown_title(markdown: str) -> str:
    for line in markdown.splitlines():
        stripped = line.strip()

        if stripped.startswith("# "):
            return stripped.removeprefix("# ").strip()

    return "Untitled research package"
