# Bouc–Wen modelling of an unreinforced elastomeric isolator (UEI)

This project fits hysteresis models to cyclic force–displacement data for a
low-cost unreinforced elastomeric isolator (UEI). The workflow includes:

- parameter identification for multiple Bouc–Wen variants,
- identifiability analysis using profile likelihood,
- reduced-model fitting with fixed sigma and n values,
- leave-one-out prediction across vertical load levels,
- generation of figures and Excel summaries for the study.

## Repository files

- `uei_boucwen.py` — core models, objective functions, identification routines,
  profile likelihood, reduction, and a command-line example driver.
- `main.py` — full analysis pipeline that reads the raw data and writes all plots
  and diagnostics output.
- `requirements.txt` — Python dependencies.
- `Analytical_Raw.xlsx` — user-provided experimental data. The script expects
  three side-by-side blocks, one per vertical load (20, 30, and 40 kN), with
  columns `Displacement (mm)` and `Force (kN)`.

## Models implemented

- Normalized Bouc–Wen model
- Pinching Bouc–Wen model
- Amplitude-dependent Bouc–Wen model

These are implemented in `uei_boucwen.py` as classes:

- `NormalizedBoucWen`
- `PinchingBoucWen`
- `AmplitudeDependentBoucWen`

## Analysis workflow

The study follows the sequence implemented in the scripts:

1. Load experimental displacement/force data.
2. Identify model parameters for each load using differential evolution
   (with an L-BFGS-B local polish).
3. Compare model accuracy using RMSE, NRMSE, and R² metrics.
4. Perform profile-likelihood checks for parameter identifiability.
5. Fit a reduced model with fixed `sigma = 1` and `n = 1`.
6. Interpolate reduced parameters to a held-out load for prediction.
7. Save plots and a summary Excel file.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the library/example driver:

```bash
python uei_boucwen.py
```

This reads `Analytical_Raw.xlsx` in the project folder and prints the full
pipeline diagnostics, including full-fit results, reduced-model results, profile
likelihood checks, and the held-out-load prediction summary.

Run the plotting / manuscript pipeline:

```bash
python main.py
```

This generates the figure set next to the script, including:

- `fig_model_fit_<load>kN.svg`
- `fig_parameters_vs_load_<param>.svg`
- `fig_prediction.svg`
- `fig_profile_likelihood_<param>_<load>kN.svg`
- `hysteresis_fitting_results_<load>kN_<panel>.svg`
- `best_models_comparison_<load>kN.svg`
- `diagnostic_comparison_<metric>.svg`
- `bouc_wen_results.xlsx`

## Example Python usage

```python
import uei_boucwen as U

data = U.load_data("Analytical_Raw.xlsx", loads=(20, 30, 40))
params, metrics = U.identify(
    U.NormalizedBoucWen,
    {"k_x": (0.01, 5.0), "k_w": (0.001, 3.0), "rho": (0.01, 5.0),
     "sigma": (0.5, 1.5), "n": (0.5, 5.0)},
    data[20],
    maxiter=500,
    popsize=10,
)
```

## Notes

- Differential evolution is stochastic; pass `seed=...` to `identify()` for
  repeatable runs.
- Units: `k_x` is in kN/mm, `k_w` is in kN, and `rho`, `sigma`, `n`, `xi`,
  and `x_c` are dimensionless or as noted in the model definitions.
- The plotting script expects the input spreadsheet to be present in the same
  folder as the scripts.

## License

MIT
