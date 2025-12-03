
### `notebooks/README.md`
```markdown
# notebooks

Aquest directori conté notebooks Jupyter utilitzats per explorar les dades i provar idees. No formen part del pipeline de producció, però poden ser útils per comprendre el procés.

## Contingut

- `explore.ipynb`: exploració inicial del dataset, anàlisi de distribucions i valors nuls.
- `imputation.ipynb`: demostració dels mètodes d’imputació definits a `src/data_prep/imputation.py`.
- `sel_trams.ipynb`: anàlisi per seleccionar els trams més representatius.
- `arima.ipynb`: cerca i entrenament de models ARIMA; mostra els paràmetres trobats i les mètriques.

## Execució

Assegura’t de tenir l’entorn preparat i executa:

```bash
jupyter notebook
