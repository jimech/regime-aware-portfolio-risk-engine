python -m regime_risk_engine run-advanced-demo \
  --output-dir outputs/advanced_demo \
  --analyst "Jimena Chinchilla"

python -m regime_risk_engine inspect-advanced-package \
  --package-dir outputs/advanced_demo/package

streamlit run src/regime_risk_engine/dashboard.py

## Quality checks

Run:

    ruff format .
    ruff check .
    ruff format --check .
    mypy src
    pytest --cov=regime_risk_engine --cov-report=term-missing

