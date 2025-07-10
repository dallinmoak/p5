quick setup script:

```bash
python3 -m venv .venv
source .venv/bin/activate
touch requirements.txt
pip install jupyter pandas lets_plot numpy scikit-learn
pip freeze > requirements.txt
```
install and run the notebook:
```bash
pip install -r requirements.txt
quarto render {notebook}.qmd --out index.html