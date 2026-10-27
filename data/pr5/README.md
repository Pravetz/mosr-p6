# Practical Work 5 preserved results

These CSV files are the inputs and derived tables used by the original Practical Work 5
notebook. They were preserved from `pr5_data.zip` without changing their values.

Source archive SHA-256:
`ffc55d925e8002114061764b85a293579f5ed784b17aaaa7e8ba3e39f3830188`.

Important provenance limitation: the archive did not contain the code that originally
generated the latency and classification measurements. The repository can reproduce the
statistical analysis of these files, but it cannot claim to regenerate the measurements.

The ten stored repetitions in `pr5_three_algorithms_5datasets.csv` are identical within
every dataset/algorithm pair. They are therefore not independent observations. The corrected
analysis in `src/analyze_pr5.py` collapses them to five independent dataset blocks before the
Friedman and Nemenyi tests. The original derived rank, Nemenyi and Cohen-d tables are retained
only as historical outputs.
