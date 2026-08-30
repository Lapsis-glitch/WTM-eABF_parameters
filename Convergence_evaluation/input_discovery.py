"""General input discovery and normalized simulation-run records.

The analysis programs consume :class:`RunRecord` objects rather than making
assumptions about experiment-specific directory names.  Legacy folder-name
inference is deliberately kept small and optional; explicit files and
manifests are the canonical interfaces.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunRecord:
    """Normalized description of one simulation/run."""

    run_id: str
    pmf_file: Path
    count_file: Path | None = None
    group: str | None = None
    seed: str | int | float | None = None
    parameter_values: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def parameter_name(self) -> str | None:
        return next(iter(self.parameter_values), None)

    def parameter_value(self) -> Any:
        name = self.parameter_name()
        return self.parameter_values.get(name) if name else None


@dataclass(frozen=True)
class DiscoveryResult:
    """Runs plus diagnostics emitted during discovery."""

    runs: tuple[RunRecord, ...]
    diagnostics: tuple[str, ...] = ()


def parse_value(value: Any) -> Any:
    """Convert manifest scalar values to numbers when that is unambiguous."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    return int(number) if number.is_integer() else number


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _infer_legacy_metadata(path: Path, metadata_regex: str | None = None) -> dict[str, Any]:
    """Best-effort compatibility metadata; never required for a valid run."""

    name = path.parent.name
    metadata: dict[str, Any] = {}
    if metadata_regex:
        match = re.search(metadata_regex, name)
        if match:
            metadata.update({key: parse_value(value) for key, value in match.groupdict().items()})

    # Compatibility for the historical ``parameter_value_seed_N`` convention.
    seed_match = re.search(r"(?:^|_)seed_([^_]+)$", name)
    base_name = name
    if seed_match:
        metadata.setdefault("seed", parse_value(seed_match.group(1)))
        base_name = name[:seed_match.start()]
    parts = base_name.split("_")
    if len(parts) >= 2:
        value = parse_value(parts[-1])
        if isinstance(value, (int, float)):
            metadata.setdefault("parameter_name", "_".join(parts[:-1]))
            metadata.setdefault("parameter_value", value)
            # Legacy sweep folders are grouped by the parameter name so all
            # values can share one semantic plot panel. Values remain in the
            # normalized parameter_values field and are never re-parsed later.
            metadata.setdefault("group", "_".join(parts[:-1]))
    metadata.setdefault("group", base_name)
    return metadata


def record_from_mapping(row: Mapping[str, Any], manifest_path: Path) -> RunRecord:
    """Build a run record from a CSV/TSV row."""

    base = manifest_path.parent.resolve()
    pmf_value = row.get("pmf_file") or row.get("pmf")
    if not pmf_value:
        raise ValueError("manifest row is missing pmf_file")
    pmf = _resolve_path(str(pmf_value), base)
    count_value = row.get("count_file") or row.get("count")
    count = _resolve_path(str(count_value), base) if count_value else None
    run_id = str(row.get("run_id") or pmf.stem)
    group = row.get("group") or None
    seed = parse_value(row.get("seed"))
    parameters: dict[str, Any] = {}
    parameter_name = row.get("parameter_name")
    parameter_value = row.get("parameter_value")
    if parameter_name and parameter_value not in (None, ""):
        parameters[str(parameter_name)] = parse_value(parameter_value)
    if row.get("parameter_values"):
        raw = row["parameter_values"]
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(parsed, dict):
                parameters.update({str(k): parse_value(v) for k, v in parsed.items()})
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid parameter_values JSON: {exc}") from exc
    for key, value in row.items():
        if key.startswith("parameter_") and key not in {"parameter_name", "parameter_value", "parameter_values"}:
            if value not in (None, ""):
                parameters[key[len("parameter_"):]] = parse_value(value)
    metadata = {
        str(key): value for key, value in row.items()
        if key not in {"run_id", "pmf_file", "pmf", "count_file", "count", "group", "seed",
                       "parameter_name", "parameter_value", "parameter_values"}
        and not key.startswith("parameter_")
        and value not in (None, "")
    }
    return RunRecord(run_id, pmf, count, str(group) if group else None, seed, parameters, metadata)


def read_manifest(path: str | Path) -> DiscoveryResult:
    """Read a CSV or TSV manifest, resolving file paths relative to it."""

    manifest = Path(path).expanduser().resolve()
    delimiter = "\t" if manifest.suffix.lower() in {".tsv", ".tab"} else ","
    runs: list[RunRecord] = []
    diagnostics: list[str] = []
    with manifest.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames or "pmf_file" not in reader.fieldnames:
            raise ValueError("manifest must contain a pmf_file column")
        for line_number, row in enumerate(reader, start=2):
            try:
                record = record_from_mapping(row, manifest)
                if not record.pmf_file.is_file():
                    diagnostics.append(f"run {record.run_id}: PMF does not exist: {record.pmf_file}")
                if record.count_file and not record.count_file.is_file():
                    diagnostics.append(f"run {record.run_id}: count file does not exist: {record.count_file}")
                runs.append(record)
            except ValueError as exc:
                diagnostics.append(f"manifest line {line_number}: {exc}")
    return DiscoveryResult(tuple(runs), tuple(diagnostics))


def explicit_run(pmf_file: str | Path, count_file: str | Path | None = None, *,
                 run_id: str | None = None, group: str | None = None,
                 seed: Any = None, parameter_values: Mapping[str, Any] | None = None,
                 metadata: Mapping[str, Any] | None = None) -> RunRecord:
    """Create a record from explicit file paths."""

    pmf = Path(pmf_file).expanduser().resolve()
    count = Path(count_file).expanduser().resolve() if count_file else None
    return RunRecord(run_id or pmf.stem, pmf, count, group, parse_value(seed),
                     dict(parameter_values or {}), dict(metadata or {}))


def _compatible_counts(pmf: Path, counts: Iterable[Path]) -> list[Path]:
    """Return counts sharing a directory and logical stem with a PMF."""

    pmf_name = pmf.name
    stems = {
        pmf_name.removesuffix(".czar.pmf"),
        pmf_name.removesuffix(".pmf"),
        pmf.stem,
    }
    candidates = []
    for count in counts:
        if count.parent != pmf.parent:
            continue
        count_stem = count.name
        for suffix in (".zcount", ".count", ".hist.count", ".hist.zcount"):
            count_stem = count_stem.removesuffix(suffix)
        if count_stem in stems or any(count_stem.startswith(stem) or stem.startswith(count_stem) for stem in stems):
            candidates.append(count)
    return sorted(set(candidates))


def discover_runs(root: str | Path, *, pmf_pattern: str = "**/*czar.pmf",
                  count_pattern: str | None = None, metadata_regex: str | None = None,
                  require_count: bool = True) -> DiscoveryResult:
    """Recursively discover PMFs and pair them with compatible count files."""

    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"input root does not exist or is not a directory: {base}")
    pmfs = sorted(path for path in base.glob(pmf_pattern) if path.is_file())
    count_glob = count_pattern or "**/*count*"
    counts = sorted(path for path in base.glob(count_glob) if path.is_file())
    diagnostics: list[str] = []
    records: list[RunRecord] = []
    matched_counts: set[Path] = set()
    if not pmfs:
        diagnostics.append(f"zero PMF matches for pattern {pmf_pattern!r} under {base}")
    for pmf in pmfs:
        candidates = _compatible_counts(pmf, counts)
        if len(candidates) == 0 and require_count:
            diagnostics.append(f"unmatched PMF: {pmf}")
            continue
        if len(candidates) > 1:
            diagnostics.append(f"multiple plausible count files for PMF {pmf}: {candidates}")
            continue
        if candidates:
            matched_counts.add(candidates[0])
        metadata = _infer_legacy_metadata(pmf, metadata_regex)
        records.append(RunRecord(
            run_id=str(pmf.relative_to(base)), pmf_file=pmf,
            count_file=candidates[0] if candidates else None,
            group=str(metadata.get("group")) if metadata.get("group") else None,
            seed=metadata.get("seed"),
            parameter_values=({str(metadata["parameter_name"]): metadata["parameter_value"]}
                              if "parameter_name" in metadata and "parameter_value" in metadata else {}),
            metadata=metadata,
        ))
    if not counts and require_count and pmfs:
        diagnostics.append(f"zero count matches for pattern {count_glob!r} under {base}")
    for count in counts:
        if count not in matched_counts:
            diagnostics.append(f"unmatched count file: {count}")
    for diagnostic in diagnostics:
        LOGGER.warning(diagnostic)
    return DiscoveryResult(tuple(records), tuple(diagnostics))


def records_from_inputs(*, root: str | Path | None = None, manifest: str | Path | None = None,
                        pmf_file: str | Path | None = None, count_file: str | Path | None = None,
                        pmf_pattern: str = "**/*czar.pmf", count_pattern: str | None = None,
                        metadata_regex: str | None = None, require_count: bool = True) -> DiscoveryResult:
    """Select explicit, manifest, or automatic-discovery input mode."""

    selected = sum(value is not None for value in (manifest, pmf_file, root))
    if selected > 1:
        raise ValueError("choose only one of root, manifest, or pmf_file input modes")
    if manifest:
        return read_manifest(manifest)
    if pmf_file:
        return DiscoveryResult((explicit_run(pmf_file, count_file),))
    if root:
        return discover_runs(root, pmf_pattern=pmf_pattern, count_pattern=count_pattern,
                             metadata_regex=metadata_regex, require_count=require_count)
    raise ValueError("one of root, manifest, or pmf_file is required")
