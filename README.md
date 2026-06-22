[README.md](https://github.com/user-attachments/files/29221494/README.md)
# Age-Structured Measles SEIR/SIRVD Simulator

A browser-based, single-page web app for simulating measles transmission with age structure, vaccination, seasonality, vital dynamics, and disease-induced mortality.

This project ports the original Python ODE workflow into a static HTML/CSS/JavaScript app that can be hosted directly on GitHub Pages. The app runs entirely in the browser and does not require a backend server.

## Features

- Age-structured SEIR/SIRVD-style measles model with four age groups:
  - 0–4 years
  - 5–14 years
  - 15–44 years
  - 45+ years
- Susceptible, exposed, infectious, recovered, one-dose vaccinated, two-dose vaccinated, and cumulative death compartments
- Approximate POLYMOD-derived contact matrix
- Seasonal transmission forcing
- R0-targeted baseline transmission calibration
- First-dose and second-dose vaccination parameters
- Case fatality rate and natural birth/death dynamics
- Interactive charts for:
  - Total SEIR compartments
  - Infectious individuals by age group
  - Vaccination and cumulative deaths
  - Susceptible fraction by age group
  - Effective reproductive number Rt
  - Seasonal beta(t)
  - 30-day smoothed annualized incidence
- CSV export of the full simulation time series
- GitHub Pages ready

## Project structure

```text
.
├── index.html              # Single-page web app markup
├── styles.css              # Responsive UI styling
├── app.js                  # JavaScript model, solver, charts, and CSV export
├── SIRVD_age_struc.py      # Original Python source model
├── README.md               # Project documentation
├── LICENSE                 # MIT license
└── .gitignore              # Basic Git ignore rules
```

## How to run locally

Because the app uses Chart.js from a CDN, the easiest way to run it locally is with a simple static file server.

### Option 1: Python static server

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

### Option 2: Open directly

You can also open `index.html` directly in a browser, but a local server is recommended.

## How to deploy to GitHub Pages

1. Create a new GitHub repository.
2. Upload all files in this folder to the root of the repository.
3. Go to **Settings → Pages**.
4. Under **Build and deployment**, choose:
   - Source: `Deploy from a branch`
   - Branch: `main`
   - Folder: `/root`
5. Save the settings.
6. GitHub will provide a Pages URL after the deployment completes.

## Model notes

This app is based on an age-structured measles model with:

- Vital dynamics: births enter the youngest age group and natural mortality applies to living compartments.
- Disease-induced mortality: cumulative disease deaths are tracked through the D compartment.
- Vaccination: one-dose and two-dose vaccination compartments are included with reduced susceptibility.
- Seasonality: beta(t) follows sinusoidal seasonal forcing.
- Age mixing: force of infection is calculated with an approximate age-group contact matrix.

The original Python script used `scipy.integrate.odeint`, which performs adaptive numerical integration. The browser version uses a fourth-order Runge-Kutta method with small internal substeps to keep the fast measles dynamics numerically stable.

## Important limitations

This project is intended for educational, exploratory, and proof-of-concept public health modeling use. It is not a validated forecasting or outbreak-response decision tool.

Model results are sensitive to assumptions about:

- Contact matrix values
- Initial immunity
- Vaccination rates
- Case fatality rate
- R0
- Seasonality
- Population age distribution

Before using this for operational public health analysis, parameters should be reviewed, validated, and calibrated against local epidemiologic data.

## Suggested repository description

Browser-based age-structured measles SEIR/SIRVD simulator with vaccination, seasonality, Rt tracking, and CSV export.

## License

MIT License. See `LICENSE` for details.
