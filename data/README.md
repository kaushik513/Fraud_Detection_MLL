# data/

Put the dataset CSV here, e.g. `data/credit_card.csv`.

Expected: one row per record, a binary `Target` column (1 = fraud, 0 = legitimate),
categorical columns as text and the rest numeric. An `ID` column is dropped
automatically. The reference study [1] describes 25,134 records, 19 features and
420 positives, downloaded from the Kaggle repository cited in the paper — use the
exact link from the paper's data-availability section so the reproduction matches.

If the target column has a different name, pass `--target NAME` (CLI) or type it in
the GUI's Data tab.
