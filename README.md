# Tokamak Plasma Visualizing Tool Kit

A web-based teaching tool for designing a tokamak and computing its plasma equilibrium with TokaMaker from the OpenFUSIONToolkit (OFT). It is aimed at high-school students and is an alternative to the course's Jupyter notebook (`MAIN_AOE_tokamaker.ipynb`, distributed separately).

## Quick start

```
git clone <this repository> && cd SHP-Tokamaker
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

`requirements.txt` installs OpenFUSIONToolkit from PyPI (`openfusiontoolkit`; tested with v26.6 and v26.9). Wheels exist for Linux and macOS only, so on Windows use WSL. To use your own OFT build instead, leave the PyPI package out and set `OFT_ROOTPATH` to its install directory (default `~/OpenFUSIONToolkit/install_release`).

## Using the GUI

1. **Shape the plasma.** Use the sliders to set the target major radius, minor radius, elongation and triangularity.
2. **Draw the vessel and coils.** Edit the vacuum-vessel outline as a table of (R, Z) points, and edit the positions of the poloidal-field (PF) coils in a second table (add or delete rows to change how many there are). A live cross-section updates as you edit, with warnings such as a coil overlapping the vessel or the plasma not fitting inside it.
3. **Set the physics.** Choose the toroidal field B0, the plasma current target (MA) and the core pressure target (kPa). Set the shape exponents (alpha, gamma) of the FF' and P' profiles. A live profile plot shows the result.
4. **Solve.** Press "Solve equilibrium". TokaMaker runs and the app reports whether the solve converged.
5. **Read the results.** The app shows the flux-surface plot with the target boundary, equilibrium statistics (achieved vs. target shape, q95, beta_pol, l_i and others) with hints on what to change, the PF coil currents against the 2 x Ip limit, and the vertical-stability feedback capability parameter, which should be below 2.

## How it works

- `streamlit_app.py` is the GUI. It collects the design, validates it, draws the live plots and displays the results.
- `tokamak_solver.py` is the solver backend. For every solve the GUI launches it as a separate process:

  ```
  python tokamak_solver.py design.json outdir
  ```

  A separate process is needed because OpenFUSIONToolkit can only be initialised once per process. The solver builds the mesh, solves the Grad-Shafranov equilibrium and writes `results.json`, `mesh.png` and `equilibrium.png` into `outdir`. You can also run it by hand from the command line.

## Legacy files

`AOE_tokamaker.ipynb`, `inspiration_code/` and `examples/` come from the earlier version of this project, which focused on vertical displacement events. The GUI no longer uses them.

## Contributing

For contributions or questions, please email nl2951@columbia.edu

## Acknowledgments

Built using the OpenFUSIONToolkit and TokaMaker for Grad-Shafranov equation solving.
Initial AOE Tokamaker and CUTE VDE code was provided by Jamie Xia at the Columbia Plasma Physics Lab. She also mentored Nikas Lukyanov and Matthew Long, the two contributors of this repository.
