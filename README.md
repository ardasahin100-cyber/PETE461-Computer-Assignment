# PETE 461 - Computer Assignment 1

Middle East Technical University, Department of Petroleum and Natural Gas Engineering

PETE 461 Natural Gas Engineering, 2026 Fall

**Group members**
- Fırat Kartal - 2515070
- Arda Şahin - 2515245

## What the code does

The program calculates gas density, viscosity, formation volume factor, isothermal compressibility
and pseudo-pressure every 10 ft along a wellbore. Pressure and temperature change linearly from the
wellhead to the bottomhole. The input data are generated from student ID 2515070.

Correlations used:
- Pseudo-critical properties: Standing, Kay's mixing rule, Wichert-Aziz correction
- Z-factor: Redlich-Kwong equation of state
- Viscosity: Lee-Gonzales-Eakin (compared with Carr-Kobayashi-Burrows charts)
- Pseudo-pressure: trapezoidal integration from 14.7 psia

## How to run

Requirements: Python 3, `numpy`, `matplotlib`

```
pip install numpy matplotlib
python PETE461_Computer_Assignment1.py
```

The student ID and the gas inputs are at the top of the script. The results are written to the
`results` folder (one CSV file and the plots).

## Output

- `results/wellbore_properties.csv`: all properties at 211 depth points
- `results/*.png`: plots of density, viscosity, formation volume factor, compressibility,
  pseudo-pressure, Z-factor, Z-factor isotherms and viscosity versus gas gravity
