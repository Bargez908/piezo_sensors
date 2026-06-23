# Changelog

Questo file traccia le modifiche principali per commit.

## 2026-05-05 - Refactor code structure for improved readability and maintainability

- Commit: `5c268162c594335734a0cc46ad2c599fdd39d2fe`
- Author: `Bargez908`
- Date: `2026-05-05 15:16:32 +0200`

### Modifiche principali

- Refactor esteso di `def_all_classification_new_resume.py` (409 inserimenti, 197 rimozioni).
- Introdotta logica più robusta per `resume` su AR-SVM, con skip dei fold già completati leggendo il CSV.
- Introdotto checkpoint incrementale dei risultati fold (`CM .png`, `CM .npy`, riga CSV) durante l'esecuzione.
- Aggiornato il flusso parallelizzato SVM per usare streaming non ordinato (`generator_unordered`) e gestione progressiva dei risultati.
- Migliorata la validazione task-risultato con controllo a chiavi (rilevazione mismatch/duplicati).
- Rifatta la fase di `AR FINAL EVALUATION` con:
  - ricostruzione/merge risultati da CSV esistente + run corrente,
  - ricalcolo robusto delle mean confusion matrix,
  - riscrittura CSV deduplicata e ordinata.
- Introdotta naming logic dipendente dalla modalità AR:
  - titolo CM: `SPECTRAL AR-SVM` in modalità spettrale,
  - file CSV: `SPECTRAL_AR_SVM.csv` in modalità spettrale (altrimenti `AR_SVM.csv`).
- Aggiunte metriche per classe e metadati SVM nei risultati (`precision`, `recall`, `f1`, support vectors, `gamma_eff`).

### File toccati nel commit

- `M def_all_classification_new_resume.py`
- `A __pycache__/def_all_classification_new_resume.cpython-312.pyc`

