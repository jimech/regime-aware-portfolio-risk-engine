# Release checklist

Use this checklist before tagging or publishing a release.

## 1. Confirm repository health

Run the full local quality suite:

```bash
ruff format .
ruff check .
ruff format --check .
mypy src
pytest --cov=regime_risk_engine --cov-report=term-missing
```

Confirm GitHub Actions passes on `main`.

## 2. Review version number

The package version is defined in:

```text
src/regime_risk_engine/__init__.py
```

Use semantic versioning:

```text
MAJOR.MINOR.PATCH
```

Guidance:

- Increment PATCH for bug fixes and documentation-only improvements.
- Increment MINOR for backward-compatible features.
- Increment MAJOR for breaking changes.

## 3. Review documentation

Confirm that:

- `README.md` is accurate.
- `docs/cli.md` reflects current CLI commands.
- `docs/methodology.md` reflects major analytical workflow changes.
- New architectural decisions have ADRs.
- The quality checklist is still accurate.

## 4. Review generated artifacts

Confirm generated local artifacts are not committed accidentally.

Common generated paths include:

- `demo/`
- `reports/`
- `htmlcov/`
- `.coverage`
- `.pytest_cache/`
- `.mypy_cache/`
- `.ruff_cache/`

## 5. Create release commit

Example:

```bash
git add .
git commit -m "chore: prepare release v0.1.0"
git push
```

## 6. Tag the release

Create an annotated tag:

```bash
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

## 7. Draft release notes

Release notes should include:

- Summary of major changes
- New CLI commands
- Testing and coverage status
- Known limitations
- Upgrade notes, if any

## Current release readiness

For the initial project release, confirm:

- CLI commands work locally.
- CI passes on GitHub Actions.
- Coverage is above the configured threshold.
- Report demo workflow runs successfully.
