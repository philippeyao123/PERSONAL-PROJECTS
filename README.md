#  PERSONAL PROJECTS

Welcome to my portfolio of quantitative finance and data science projects. These projects were developed during my MSc in Financial Mathematics at the London School of Economics and my MSc in Data Science at the University of Essex. They demonstrate practical applications of statistical modeling, machine learning, and financial theory in real-world scenarios. Most were designed to support internship and job applications in quantitative research, trading, and risk management.

---

##  Project Index

###  Derivatives & Volatility

- **[Heston Calibration with Smile Fitting](https://github.com/philippeyao123/PERSONAL-PROJECTS/tree/main/Heston_Calibration_with_Smile_Fitting)**   
  Calibrated the Heston stochastic volatility model to market smiles using the semi-analytical pricing formula.  
  • Performed least squares minimization on implied volatilities.  
  • Compared Heston model results to Black-Scholes and local volatility models.

- **[FX Option Pricing – Garman-Kohlhagen, Local Volatility & Barrier Options](https://github.com/philippeyao123/PERSONAL-PROJECTS/tree/main/FX_Option_Pricing_GK_LocalVol_Barriers)**   
  Priced FX options using the Garman-Kohlhagen framework extended with a Dupire local volatility surface.  
  • Simulated price paths via Monte Carlo and priced FX barrier options using finite difference PDEs.  
  • Analyzed FX volatility smiles and interest rate differentials.

- **[FX Options Pricing with Bergomi Local–Stochastic Volatility](https://github.com/philippeyao123/PERSONAL-PROJECTS/tree/main/Bergomi_Local_Volatility_FX)**   
  Implemented a stochastic-local volatility model combining a Dupire surface with a Bergomi-type volatility factor.  
  • Compared FX option prices under Garman–Kohlhagen, local volatility, and SLV models.  
  • Demonstrated dynamic smile behavior and calibrated volatility surfaces.
  
- **[RL-Based Hedging of FX Barrier Options under Bergomi + Dupire Local Volatility](https://github.com/philippeyao123/PERSONAL-PROJECTS/tree/main/RL_Based_Hedging_of_FX_Barrier_Options_under_Bergomi_%2B_Dupire_Local_Volatility)** 
  Used reinforcement learning to hedge barrier options under a hybrid stochastic-local volatility model.
  • Simulated FX price paths under Bergomi + Dupire framework and trained agents using deep Q-networks.
  • Evaluated hedge performance against traditional Delta and Vega hedging techniques.
  
- **[Machine Learning for Option Pricing](https://github.com/philippeyao123/PERSONAL-PROJECTS/tree/main/MACHINE%20LEARNING%20FOR%20OPTION%20PRICING)**  
  Implemented GBM, LSTM, and GRU models to approximate European call option prices using historical data.  
  • Assessed model performance across different moneyness levels.

- **[Equity Implied Volatility](https://github.com/philippeyao123/PERSONAL-PROJECTS/tree/main/EQUITY%20IMPLIED%20VOLATILITY)**   
  Constructed implied volatility surfaces from market data and examined smile/skew patterns across strikes and maturities.

- **[Local Volatility Model](https://github.com/philippeyao123/PERSONAL-PROJECTS/tree/main/LOCAL%20VOLATILITY%20MODEL)** 🆕  
  Calibrated local volatility surfaces using Dupire’s PDE and compared model outputs to market-implied volatilities.


---

###  Portfolio & Market Microstructure

- **[Portfolio Optimization](https://github.com/philippeyao123/PERSONAL-PROJECTS/tree/main/PORTFOLIO%20OPTIMIZATION)**  
  Explored Mean-Variance, Black-Litterman, and Michaud Resampling for portfolio construction and allocation robustness.

- **[High-Frequency Trading Analysis](https://github.com/philippeyao123/PERSONAL-PROJECTS/tree/main/HIGH-FREQUENCY%20TRADING%20ANALYSIS)**  
  Analyzed tick data to detect short-term alpha signals, spread patterns, and order flow imbalances.

- **[Baraka Portfolio App](https://github.com/philippeyao123/PERSONAL-PROJECTS/tree/main/baraka-portfolio-app)**  
  Built a Streamlit dashboard for portfolio construction, visualization, and performance attribution using custom allocation logic.

 - **[Alpha Research Notebook – Backtesting Framework](https://github.com/philippeyao123/PERSONAL-PROJECTS/tree/main/Alpha%20Research)** 
  Designed a modular backtesting platform to evaluate alpha factors and strategy P&L performance.
  • Incorporated signal generation, position sizing, transaction costs, and rolling statistics.
  • Applied to momentum and volatility-based strategies with dynamic portfolio rebalancing.

---

###  Risk, Credit & Forecasting

- **[Credit Risk Modeling](https://github.com/philippeyao123/PERSONAL-PROJECTS/tree/main/CREDIT%20RISK%20MODELING)**  
  Modeled Probability of Default (PD) using logistic regression and tree-based models. Validated models using ROC, AUC, and K-S statistics.

- **[Risk Forecasting Projects](https://github.com/philippeyao123/PERSONAL-PROJECTS/tree/main/RISK%20FORECASTING%20PROJECTS)**  
  Estimated Value-at-Risk (VaR) and Expected Shortfall (ES) using GARCH models, historical simulation, and EVT tail modeling.

- **[Risk Management Simulation](https://github.com/philippeyao123/PERSONAL-PROJECTS/tree/main/RISK%20MANAGEMENT%20SIMULATION)**  
  Simulated capital adequacy under stress scenarios and modeled dynamic capital requirements.

---

###  Interest Rate Derivatives & American Options

- **[CMS Spread Options Pricing](https://github.com/philippeyao123/PERSONAL-PROJECTS/tree/main/CMS_Spread_Options_Pricing_Analytics)**   
  Priced CMS spread options using a copula-based approach to model the joint distribution of swap rates.  
  • Applied Gaussian copulas and Monte Carlo simulations to model spread distributions and payoff profiles.

- **[CMS Convexity & Option Pricing](https://github.com/philippeyao123/PERSONAL-PROJECTS/tree/main/CMS_Derivatives_Pricing_Analytics)**   
  Computed CMS convexity adjustments and priced CMS options.  
  • Incorporated convexity correction techniques for swap rates and analyzed option payoff sensitivity.

- **[American Options Pricing](https://github.com/philippeyao123/PERSONAL-PROJECTS/tree/main/American_Options_Pricing_Analytics)**   
  Compared three valuation techniques for American-style derivatives:  
  • Binomial tree, Barone-Adesi & Whaley approximation, and Least-Squares Monte Carlo (LSMC).  
  • Assessed exercise boundary dynamics and early exercise premiums.

- **[Hull-White One-Factor Model](https://github.com/philippeyao123/PERSONAL-PROJECTS/tree/main/Hull_White_1F_Model)**   
  Implemented the Hull & White one-factor interest rate model to price Bermudan swaptions.  
  • Applied PDE techniques and lattice models to analyze early exercise features and calibration results.

---

###  Quantitative Research & Macro Modeling

- **[Quantitative Forecasting for High-Yield Bonds](https://github.com/philippeyao123/PERSONAL-PROJECTS/tree/main/QUANTITATIVE%20FORECASTING%20FOR%20HIGH-YIELD%20BONDS)**  
  Dissertation project comparing LSTM, GRU, and GBM for high-yield bond pricing and risk estimation.  
  • Integrated macroeconomic and sentiment variables into predictive frameworks.

- **[Sentiment Analysis on Financial News](https://github.com/philippeyao123/PERSONAL-PROJECTS/tree/main/SENTIMENT%20ANALYSIS%20ON%20FINANCIAL%20NEWS)**  
  Applied NLP techniques (TF-IDF, Vader) to extract sentiment signals from news headlines.  
  • Developed sentiment-based factor signals for return prediction.

- **[Stochastic Processes & Monte Carlo](https://github.com/philippeyao123/PERSONAL-PROJECTS/tree/main/STOCHASTIC%20PROCESSES%20%20AND%20MONTE%20CARLO%20SIMULATIONS)**  
  Simulated stochastic differential equations and diffusion processes using Euler-Maruyama.  
  • Used Monte Carlo pricing for exotic options and path-dependent derivatives.

---

###  Systematic Trading Strategies

- **[Statistical Arbitrage – Cointegration-Based Mean Reversion](https://github.com/philippeyao123/PERSONAL-PROJECTS/tree/main/STASTICAL%20ARBITRAGE%20–%20COINTEGRATION%20BASED%20MEAN%20REVERSION)**  
  Developed a pair trading strategy using cointegration, z-score signals, and rolling window regressions.  
  • Evaluated performance using Sharpe Ratio, drawdown, and turnover metrics.
- **[Volatility Arbitrage – Implied vs Realized Volatility ](https://github.com/philippeyao123/PERSONAL-PROJECTS/tree/main/Volatility%20Arbitrage%20)**  
  Simulated a volatility arbitrage strategy based on the spread between implied and realized volatility.
  • Generated GBM price paths and computed rolling 30-day realized volatility.
  • Modeled implied volatility as noisy RV and executed long/short variance trades based on a threshold.
  • Calculated P&L from variance differences and visualized strategy dynamics.
---

##  Tech Stack

- **Languages**: Python (99.8%), SQL  
- **Libraries**: NumPy, pandas, scikit-learn, statsmodels, yfinance, QuantLib, Plotly, Streamlit  
- **Tools**: Jupyter Notebook, Git, GitHub Actions

---

##  Contact

**Bathaix Philippe-Emmanuel Yao**  
📍 London, United Kingdom  
📧 [yaophilippeemmanuel@gmail.com](mailto:yaophilippeemmanuel@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/bathaix-philippe-emmanuel-yao-a302ab176/)  
🧠 FRM Level 1 Cleared | 📈 CFA Level 1 Candidate

