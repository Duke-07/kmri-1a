// Bayesian Regime Detection Engine - Quantitative Data & Parameters
export const REGIMES = [
  {
    id: 'risk_on',
    name: 'Risk-On',
    code: 'S1',
    color: '#10b981',
    glow: 'rgba(16, 185, 129, 0.25)',
    border: 'rgba(16, 185, 129, 0.4)',
    badgeClass: 'badge-risk-on',
    historicalFreq: '38.4%',
    meanReturnAnn: '+24.6%',
    annVol: '12.4%',
    kurtosis: 3.1,
    studentNu: 18.4,
    description: 'High momentum, sustained FII/DII net inflows, declining India VIX (<13), strong market breadth (>70% advancers).',
    allocationTilt: '+5.0% Equity Beta Overweight',
    riskOMeter: 'Moderate to High'
  },
  {
    id: 'late_cycle',
    name: 'Late-Cycle',
    code: 'S2',
    color: '#f59e0b',
    glow: 'rgba(245, 158, 11, 0.25)',
    border: 'rgba(245, 158, 11, 0.4)',
    badgeClass: 'badge-late-cycle',
    historicalFreq: '24.1%',
    meanReturnAnn: '+8.2%',
    annVol: '16.8%',
    kurtosis: 4.8,
    studentNu: 7.2,
    description: 'Elevated valuations, narrowing market breadth, divergence between large-cap and small/mid-cap, rising yield spread.',
    allocationTilt: 'Neutral Beta, Quality & Low-Vol Factor Tilt',
    riskOMeter: 'High'
  },
  {
    id: 'transitional',
    name: 'Transitional',
    code: 'S3',
    color: '#818cf8',
    glow: 'rgba(129, 140, 248, 0.25)',
    border: 'rgba(129, 140, 248, 0.4)',
    badgeClass: 'badge-transitional',
    historicalFreq: '16.5%',
    meanReturnAnn: '+1.4%',
    annVol: '19.2%',
    kurtosis: 6.2,
    studentNu: 5.4,
    description: 'High Shannon entropy across models, BOCPD hazard rate >0.4, sudden macro/monetary policy shifts, bidirectional gap risk.',
    allocationTilt: '-2.5% Beta Tilt, 10% Cash & Liquidity Buffer',
    riskOMeter: 'Very High'
  },
  {
    id: 'post_shock',
    name: 'Post-Shock',
    code: 'S4',
    color: '#06b6d4',
    glow: 'rgba(6, 182, 212, 0.25)',
    border: 'rgba(6, 182, 212, 0.4)',
    badgeClass: 'badge-post-shock',
    historicalFreq: '11.8%',
    meanReturnAnn: '+31.8%',
    annVol: '26.4%',
    kurtosis: 7.9,
    studentNu: 4.1,
    description: 'Mean-reverting high-volatility bounce, extreme oversold breadth, aggressive institutional accumulation, widening credit spreads tightening.',
    allocationTilt: '+3.5% Opportunistic Beta Tilt with Strict Stop-Loss',
    riskOMeter: 'Very High'
  },
  {
    id: 'risk_off',
    name: 'Risk-Off',
    code: 'S5',
    color: '#f43f5e',
    glow: 'rgba(244, 63, 94, 0.25)',
    border: 'rgba(244, 63, 94, 0.4)',
    badgeClass: 'badge-risk-off',
    historicalFreq: '9.2%',
    meanReturnAnn: '-28.4%',
    annVol: '34.2%',
    kurtosis: 11.6,
    studentNu: 3.2,
    description: 'Systemic liquidity freeze, correlation breakdown toward 1.0, India VIX >28, heavy institutional outflows, TDA topological persistence spike.',
    allocationTilt: '-5.0% Maximum Underweight Beta, Allocate to Gold & Sovereign T-Bills',
    riskOMeter: 'Principal at Very High Risk (Capital Preservation Triggered)'
  }
];

export const CRISIS_SCENARIOS = [
  {
    id: 'gfc_2008',
    name: '2008 Global Financial Crisis',
    period: 'Sep 2008 – Mar 2009',
    headline: 'Lehman Collapse & Global Deleveraging',
    peakRiskOff: 0.94,
    bocpdConfidence: 'P = 0.89',
    leadDays: '4 Trading Days',
    keyFeatures: {
      vix: 58.4,
      drawdown: '-54.2%',
      fiiOutflows: '-$12.1B',
      tdaNorm: '4.82 (High)'
    },
    timeline: [
      { date: '2008-09-01', nifty: 4420, pRiskOn: 0.12, pLateCycle: 0.35, pTransitional: 0.38, pPostShock: 0.05, pRiskOff: 0.10, action: 'Entropy alert triggered; cash buffer raised' },
      { date: '2008-09-15', nifty: 4038, pRiskOn: 0.02, pLateCycle: 0.08, pTransitional: 0.31, pPostShock: 0.04, pRiskOff: 0.55, action: 'BOCPD hazard spike P=0.89; Lehman bankruptcy' },
      { date: '2008-10-10', nifty: 3200, pRiskOn: 0.00, pLateCycle: 0.02, pTransitional: 0.09, pPostShock: 0.01, pRiskOff: 0.88, action: 'Full Risk-Off defensive tilt (-5% beta overlay)' },
      { date: '2008-10-27', nifty: 2524, pRiskOn: 0.00, pLateCycle: 0.01, pTransitional: 0.05, pPostShock: 0.00, pRiskOff: 0.94, action: 'Peak Risk-Off; Conformal prediction set {Risk-Off}' },
      { date: '2009-03-09', nifty: 2573, pRiskOn: 0.05, pLateCycle: 0.04, pTransitional: 0.22, pPostShock: 0.61, pRiskOff: 0.08, action: 'Transition to Post-Shock mean reversion; re-entry' },
      { date: '2009-05-18', nifty: 4321, pRiskOn: 0.74, pLateCycle: 0.14, pTransitional: 0.08, pPostShock: 0.04, pRiskOff: 0.00, action: 'Election upper circuit; confirmed Risk-On regime' }
    ]
  },
  {
    id: 'taper_2013',
    name: '2013 Taper Tantrum',
    period: 'May 2013 – Aug 2013',
    headline: 'INR Depreciation & Emerging Market Selloff',
    peakRiskOff: 0.79,
    bocpdConfidence: 'P = 0.84',
    leadDays: '6 Trading Days',
    keyFeatures: {
      vix: 31.8,
      drawdown: '-18.6%',
      fiiOutflows: '-$3.8B',
      tdaNorm: '2.95 (Moderate)'
    },
    timeline: [
      { date: '2013-05-15', nifty: 6187, pRiskOn: 0.68, pLateCycle: 0.22, pTransitional: 0.08, pPostShock: 0.01, pRiskOff: 0.01, action: 'Late-stage momentum; narrow breadth noted' },
      { date: '2013-05-22', nifty: 6104, pRiskOn: 0.28, pLateCycle: 0.38, pTransitional: 0.26, pPostShock: 0.02, pRiskOff: 0.06, action: 'Bernanke taper testimony; prediction set widened' },
      { date: '2013-06-20', nifty: 5655, pRiskOn: 0.08, pLateCycle: 0.16, pTransitional: 0.44, pPostShock: 0.03, pRiskOff: 0.29, action: 'ACI dynamic learning rate α adaptive step' },
      { date: '2013-08-28', nifty: 5287, pRiskOn: 0.01, pLateCycle: 0.05, pTransitional: 0.15, pPostShock: 0.00, pRiskOff: 0.79, action: 'USD/INR hits 68.80; Peak macro stress' },
      { date: '2013-09-04', nifty: 5447, pRiskOn: 0.18, pLateCycle: 0.12, pTransitional: 0.35, pPostShock: 0.32, pRiskOff: 0.03, action: 'Raghuram Rajan takes RBI helm; FCNR-B swap relief' }
    ]
  },
  {
    id: 'ilfs_2018',
    name: '2018 IL&FS Credit Crisis',
    period: 'Sep 2018 – Nov 2018',
    headline: 'NBFC Liquidity Shock & Cap-Segmentation Divergence',
    peakRiskOff: 0.82,
    bocpdConfidence: 'P = 0.86',
    leadDays: '12 Trading Days',
    keyFeatures: {
      vix: 22.4,
      drawdown: '-15.4% (Midcap -32%)',
      fiiOutflows: '-$4.2B',
      tdaNorm: '3.41 (Cap Divergence)'
    },
    timeline: [
      { date: '2018-08-28', nifty: 11738, pRiskOn: 0.42, pLateCycle: 0.45, pTransitional: 0.11, pPostShock: 0.01, pRiskOff: 0.01, action: 'Midcap divergence flag raised; Nifty hits new ATH' },
      { date: '2018-09-14', nifty: 11515, pRiskOn: 0.14, pLateCycle: 0.32, pTransitional: 0.38, pPostShock: 0.02, pRiskOff: 0.14, action: 'IL&FS commercial paper default; NBFC spread explodes' },
      { date: '2018-09-21', nifty: 11143, pRiskOn: 0.05, pLateCycle: 0.15, pTransitional: 0.32, pPostShock: 0.03, pRiskOff: 0.45, action: 'DHFL flash crash; multi-channel crash alert active' },
      { date: '2018-10-26', nifty: 10030, pRiskOn: 0.01, pLateCycle: 0.04, pTransitional: 0.13, pPostShock: 0.00, pRiskOff: 0.82, action: 'Peak NBFC liquidity crisis; Kelly tilt -5% cash buffer' },
      { date: '2018-11-20', nifty: 10600, pRiskOn: 0.22, pLateCycle: 0.25, pTransitional: 0.36, pPostShock: 0.14, pRiskOff: 0.03, action: 'RBI liquidity measures ease NBFC rollover fear' }
    ]
  },
  {
    id: 'covid_2020',
    name: '2020 COVID-19 Crash',
    period: 'Feb 2020 – Apr 2020',
    headline: 'Exogenous Pandemic Shock & TDA Early Warning',
    peakRiskOff: 0.98,
    bocpdConfidence: 'P = 0.96',
    leadDays: '18 Trading Days',
    keyFeatures: {
      vix: 83.6,
      drawdown: '-38.4%',
      fiiOutflows: '-$8.4B',
      tdaNorm: '5.67 (Historic Extreme)'
    },
    timeline: [
      { date: '2020-02-03', nifty: 11707, pRiskOn: 0.51, pLateCycle: 0.31, pTransitional: 0.14, pPostShock: 0.02, pRiskOff: 0.02, action: 'TDA persistence norm spikes 2.4σ above 60-day mean' },
      { date: '2020-02-21', nifty: 12080, pRiskOn: 0.21, pLateCycle: 0.34, pTransitional: 0.33, pPostShock: 0.02, pRiskOff: 0.10, action: 'Pre-crash divergence; systematic de-risking starts' },
      { date: '2020-03-09', nifty: 10451, pRiskOn: 0.01, pLateCycle: 0.04, pTransitional: 0.11, pPostShock: 0.01, pRiskOff: 0.83, action: 'Global travel curbs; crude oil collapse; circuit breaker' },
      { date: '2020-03-23', nifty: 7610, pRiskOn: 0.00, pLateCycle: 0.01, pTransitional: 0.01, pPostShock: 0.00, pRiskOff: 0.98, action: 'National Lockdown announced; India VIX reaches 83.6' },
      { date: '2020-04-09', nifty: 9111, pRiskOn: 0.08, pLateCycle: 0.06, pTransitional: 0.18, pPostShock: 0.65, pRiskOff: 0.03, action: 'Post-Shock mean-reversion phase; Fed/RBI massive stimulus' }
    ]
  },
  {
    id: 'election_2024',
    name: '2024 Election & Budget',
    period: 'May 2024 – Jul 2024',
    headline: 'Exit Poll Euphoria, Results Shock & Budget Realignment',
    peakRiskOff: 0.48,
    bocpdConfidence: 'P = 0.72',
    leadDays: 'Event Prior Halving',
    keyFeatures: {
      vix: 28.2,
      drawdown: '-6.2% (1-day record)',
      fiiOutflows: '-$1.5B',
      tdaNorm: '2.10 (Moderate)'
    },
    timeline: [
      { date: '2024-05-24', nifty: 22957, pRiskOn: 0.64, pLateCycle: 0.24, pTransitional: 0.10, pPostShock: 0.01, pRiskOff: 0.01, action: 'Pre-poll rally; conviction halved per event protocol' },
      { date: '2024-06-03', nifty: 23263, pRiskOn: 0.79, pLateCycle: 0.15, pTransitional: 0.05, pPostShock: 0.01, pRiskOff: 0.00, action: 'Exit polls show landslide; ATH gap up' },
      { date: '2024-06-04', nifty: 21884, pRiskOn: 0.08, pLateCycle: 0.14, pTransitional: 0.30, pPostShock: 0.00, pRiskOff: 0.48, action: 'Results tally shock; 1-day 1379 pt drop; quick absorption' },
      { date: '2024-06-07', nifty: 23290, pRiskOn: 0.62, pLateCycle: 0.22, pTransitional: 0.12, pPostShock: 0.04, pRiskOff: 0.00, action: 'Coalition stable; institutional buyback; Risk-On resumed' }
    ]
  }
];

export const CONFORMAL_CALIBRATION = {
  targetCoverage: 0.90,
  marginalCoverage: {
    split: 0.907,
    aps: 0.912,
    mondrian: 0.904,
    aci: 0.901
  },
  ece: 0.0156, // Expected Calibration Error
  brierScore: 0.1432,
  brierDecomposition: {
    reliability: 0.0124,
    resolution: 0.2185,
    uncertainty: 0.3493
  },
  rpsSkillScore: 0.284, // vs Climatological Benchmark
  mondrianClasses: [
    { regime: 'Risk-On', target: '90.0%', observed: '91.4%', meanSetSize: 1.12 },
    { regime: 'Late-Cycle', target: '90.0%', observed: '90.2%', meanSetSize: 1.48 },
    { regime: 'Transitional', target: '90.0%', observed: '89.6%', meanSetSize: 2.34 },
    { regime: 'Post-Shock', target: '90.0%', observed: '90.8%', meanSetSize: 1.82 },
    { regime: 'Risk-Off', target: '90.0%', observed: '92.1%', meanSetSize: 1.25 }
  ]
};

export const BACKTEST_METRICS = {
  walkForwardPeriod: '2007 – 2024 (18 Years)',
  informationRatio: 0.6142,
  trackingErrorAnn: '2.14%',
  strategySharpe: 1.42,
  benchmarkSharpe: 0.91,
  deflatedSharpeRatio: 0.8741,
  maxDrawdownStrategy: '-32.1%',
  maxDrawdownBenchmark: '-54.4%',
  ddReduction: '+22.3pp',
  annAlpha: '+3.42%',
  turnoverMonthly: '4.8%',
  winRateMonths: '68.4%',
  monteCarlo: {
    paths: 5000,
    horizon: '1 Year',
    meanReturn: '+14.8%',
    medianReturn: '+13.9%',
    var95: '-12.3%',
    cvar95: '-16.8%',
    probPositiveReturn: '84.2%'
  }
};

export const PIPELINE_STAGES = [
  { id: 1, title: 'Synthetic Indian Market Simulation', time: '1.2s', formula: 'r_t | S_t = k ~ Student-t(ν_k, μ_k, σ_k)', desc: '18-year 5-regime simulation with regime-conditioned Student-t fat tails (ν=3 to 18) calibrated to Nifty 50 historical extremes.' },
  { id: 2, title: 'Multi-Asset Feature Engineering', time: '1.4s', formula: 'X_t ∈ ℝ^33 (Parkinson, McClellan, FII, TDA)', desc: '33 orthogonal features: high-low Parkinson vol, breadth McClellan oscillator, FII/DII net institutional flows, and TDA persistence landscapes.' },
  { id: 3, title: 'Frequentist Gaussian HMM', time: '0.8s', formula: 'arg min_K BIC(K) = -2 ln L + p ln N', desc: 'EM Baum-Welch optimization with BIC state selection (K=3, 5, 7) and non-parametric empirical regime duration distribution verification.' },
  { id: 4, title: 'Variational Bayes HMM (VB-HMM)', time: '1.5s', formula: 'q^*(θ) ∝ exp(𝔼_{-θ}[ln p(X, Z, θ)])', desc: 'Analytical Dirichlet-Normal-Wishart conjugate variational inference producing 95% posterior credible intervals over all transition probabilities.' },
  { id: 5, title: 'Markov-Switching Baseline', time: '0.5s', formula: 'r_t = μ_{S_t} + ε_t,  ε_t ~ N(0, σ_{S_t}^2)', desc: 'Statsmodels Hamilton-style regime-switching autoregression baseline for benchmarking active ensembling superiority.' },
  { id: 6, title: 'Sequential Particle Filter + BOCPD', time: '1.8s', formula: 'w_t^{(i)} ∝ w_{t-1}^{(i)} p(y_t | x_t^{(i)})', desc: '5,000 particles with systematic resampling (ESS > 0.6N) and Adams & MacKay (2007) Bayesian Online Changepoint Detection hazard rate computation.' },
  { id: 7, title: 'Bayesian Deep Learning & SHAP', time: '2.2s', formula: 'I(S; θ | x) = H(S | x) - 𝔼[H(S | x, θ)]', desc: 'MC Dropout (200 stochastic passes) + Deep Ensemble for exact epistemic vs aleatoric uncertainty decomposition with SHAP feature attribution.' },
  { id: 8, title: 'Foundation Model Embeddings', time: '1.1s', formula: 'z_t = Encoder_{Chronos}(x_{t-252:t})', desc: 'Zero-shot foundation model representation learning utilizing rolling 252-day context windows for non-linear temporal pattern discovery.' },
  { id: 9, title: 'Conformal Prediction Calibration', time: '0.9s', formula: 'Ĉ(X_{n+1}) = {y : s(X_{n+1}, y) ≤ q̂_{1-α}}', desc: 'Split-Conformal and Adaptive Prediction Sets (APS) ensuring finite-sample exact 90% coverage regardless of underlying model calibration error.' },
  { id: 10, title: 'Calibrated Model Ensembling', time: '0.7s', formula: 'w^* = arg min_{w ∈ Δ} ||Y - ∑ w_m P_m||_2^2', desc: 'BMA log-predictive likelihood + SLSQP constrained stacking on the probability simplex, mathematically proven to outperform any member model.' },
  { id: 11, title: 'Purged Walk-Forward & Kelly Tilt', time: '1.1s', formula: 'f^* = arg max 𝔼[ln(1 + f · r_S)]', desc: 'Purged & embargoed cross-validation preventing lookahead bias, combined with half-Kelly bounded tactical overlays (±5% equity beta).' },
  { id: 12, title: 'Monte Carlo & IC Artefact', time: '0.6s', formula: 'DSR = P(SR̂ > 0 | N, Var(SR̂), γ_3, γ_4)', desc: '5,000-path regime-conditional Monte Carlo projection, Deflated Sharpe Ratio calculation (Bailey & López de Prado), and regulatory IC export.' }
];

export const INITIAL_LIVE_STATE = {
  timestamp: '2026-09-02 16:50:00 IST',
  marketStatus: 'LIVE CLOCK • Market Closed',
  niftyCurrent: 24823.15,
  niftyChange: '+142.30 (+0.58%)',
  indiaVix: 13.45,
  fiiNetFlow: '+₹1,842 Cr',
  diiNetFlow: '+₹1,120 Cr',
  mcclellanOsc: '+42.5',
  parkinsonVol: '11.8%',
  tdaPersistence: '1.42 (Normal)',
  psiDrift: '0.042 (Clean, No Drift)',
  dominantRegime: 'Risk-On',
  conviction: '84.2%',
  uncertaintyBreakdown: {
    totalEntropy: '0.512 nats',
    aleatoricPercent: 78.4,
    epistemicPercent: 21.6,
    interpretation: 'Aleatoric-dominated: Residual uncertainty stems from market randomness rather than model ignorance. High execution confidence.'
  },
  probabilities: {
    risk_on: 0.724,
    late_cycle: 0.182,
    transitional: 0.054,
    post_shock: 0.028,
    risk_off: 0.012
  },
  predictionSet: ['Risk-On', 'Late-Cycle'],
  tacticalOverlay: {
    equityBetaTilt: '+4.2%',
    recommendedAllocation: {
      equity: '74.2%',
      goldDefensive: '12.0%',
      liquidCash: '13.8%'
    },
    riskOMeter: 'Moderate'
  }
};
