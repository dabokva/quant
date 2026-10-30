Fixed-threshold reference comparison

comparison.csv: pooled results; per_acquisition.csv: acquisition-specific results.
plots/: static PDF and PNG figures. tables/: LaTeX tables (requires booktabs).
comparison.json: methods, settings, numerical results, and input hashes.

Calibration-selection rows exclude the temperature-fitting half. Test rows are held-out shot comparisons.
Gate fitting used a separate inference pass. Its differences from saved calibration predictions are in calibration_replay.
All configured thresholds are reported; the script does not select a winner using test results.
This is a follow-up exploratory analysis of existing studies. Keep it distinct from the original 1% gate.
Original gate respects its saved enabled flag. Fixed thresholds override it for this analysis only.
Corrected errors: classical wrong, neural correct on accepted shots. Introduced errors: the reverse.
Hybrid error = classical error + introduced errors - corrected errors.
Error-difference bounds cover all reported global comparisons, assuming independent shots.
They do not guarantee future accuracy or account for correlation between shots.
An empty accepted set has undefined accepted accuracy. Calls avoided are not measured time saved.
There is no training, new neural inference, gate fitting, or latency measurement in this script.
