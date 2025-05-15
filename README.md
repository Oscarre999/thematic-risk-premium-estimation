# Risk Premium Estimation of Thematic Indices: An Application of the Heston-Nandi GARCH Model with Traditional and Explicit Factor Approaches
## Abstract
This thesis presents an analysis of the risk premium of financial instruments that track thematic indices. The aim of this thesis is to analyze and compare the risk premium of different categories of thematic investments (MSCI Environment, Alternative Energy, Energy Efficiency, ecc.) with a global index, considered as a benchmark (MSCI ACWI Index). 

Problem Formulation: **"Do Thematic Investments offer a higher risk premium compared to Traditional Investments, or are the latter still the most efficient choice in terms of risk-return trade-off?"**

## Repository
- `HNGARCH_Thesis.pdf`: PDF of the thesis documentation (*italian version*)
- `HNGARCH_ppt_Presentation.pdf`: PowerPoint presentation of the thesis (*italian version*)
- `Heston-Nandi Garch Files` : Folder containing code and Excel results:
  - `HNGARCH_ANALYSIS.py`: Python script for the Heston-Nandi GARCH analysis using both Original and Explicit Factor approaches
  - `HNGARCH_PREPROCESSING.ipynb`: Jupyter Notebook file for the preprocessing of the series
  -  `HNGARCH_Results.xlsx`: Results file including inference, annualized volatility, correlation Index-Volatility and estimated Risk Premium series
  -  `Risk_Premium_Analysis_DB.xlsx`: Database for the analysis

## Model, Libraries and Technologies
- **Python** - for the Heston-Nandi Garch analysis 
- **Jupyter Notebook** - for the Preprocessing
- **Heston-Nandi Garch** - a specific subset of the GARCH model that allows for the estimation of a parameter (lambda), which is interpreted as the market risk premium of the underlying index.
- **The Explicit Factors Heson-Nandi Garch** - a modified version of the Heston-Nandi GARCH model capable of estimating a time-varying risk premium, influenced by global risk factors such as inflation risk, interest rate risk, and others.
- **Libraries** - `numpy`, `pandas`, `scipy.optimize`

## Results
- **Original Heston-Nandi Garch model** provides a reliable estimate of the risk premium for thematic indices.
- **The Explicit Factors Heston-Nandi GARCH model** generates a time-varying risk premium, enabling us to analyze its temporal evolution across thematic indices and to compare them with the reference benchmark. Moreover, the model provides estimates for the explicit factor parameters, allowing us to assess the individual contribution of each factor to the dynamics of the time-varying risk premium.
- Both the Original and the Explicit Factors models yield comparable estimates of the risk premium. In both cases, the risk premium of the MSCI ACWI benchmark is higher than those of the thematic indices.
- The findings support the conclusion that, from a risk-return perspective, thematic investments tend to be less appealing compared to traditional investments.

## Author
Oscar Maria Bolletta

## Contact
Email: bollettaoscar@gmail.com
