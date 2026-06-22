# Examples

This folder documents example outputs and walkthroughs for the regime-aware portfolio risk engine.

## Rendered example memo

A rendered example advanced research memo is available at:

```text
docs/examples/advanced_research_memo_example.md
```

This file shows the type of investment research output produced by the advanced demo workflow.

## Notebook walkthrough

The main walkthrough is:

```text
notebooks/advanced_demo_walkthrough.ipynb
```

It demonstrates the full advanced research workflow:

```text
create demo inputs
→ run advanced research export
→ generate investment memo
→ inspect supporting tables
```

## Generate an example research package

From the repository root:

```bash
python -m regime_risk_engine run-advanced-demo \
  --output-dir outputs/advanced_demo \
  --analyst "Jimena Chinchilla"
```

The generated memo will be written to:

```text
outputs/advanced_demo/package/advanced_research_memo.md
```

Supporting tables will also be written under:

```text
outputs/advanced_demo/package/
```

## Important note

Generated outputs are not committed by default. They can be recreated deterministically using the CLI command above.
