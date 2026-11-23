# KALANTAR — Data Limitations

## 1. Delay Prediction Dataset
- Contains 1.5 million synthetic journey records.
- Covers 2018–2024.
- Used to develop and evaluate the preliminary journey-delay prediction model.
- Model performance on this dataset should not be interpreted as validated real-world railway performance.

## 2. Railway Timetable
- Contains 186,114 records after initial cleaning.
- Represents a 2017 timetable and may not reflect current services.
- Used to construct scheduled station-to-station connections.
- Some segment distances are flagged for review.

## 3. Dataset Compatibility
- 1,820 train numbers appear in both datasets.
- Train-number agreement alone does not establish that the records describe the same service or route.
- A comparison of Kaggle median journey distances against maximum timetable distances showed substantial discrepancies for many trains.
- The datasets will remain separate modules until reliable route-level compatibility can be established.

## 4. Delay Propagation
- The timetable describes scheduled connectivity, not observed delay propagation.
- Any propagation feature developed without operational delay-event data will be presented as a simulation, with its assumptions and limitations documented.