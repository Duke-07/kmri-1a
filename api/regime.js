export default function handler(req, res) {
  res.setHeader('Cache-Control', 's-maxage=10, stale-while-revalidate');

  // Parse query parameters
  const vix = parseFloat(req.query.vix) || 13.45;
  const fiiFlow = parseFloat(req.query.fii) || 1842;
  const mcclellan = parseFloat(req.query.mcclellan) || 42.5;
  const tdaNorm = parseFloat(req.query.tda) || 1.42;

  // Bayesian Softmax formulation
  let wOn = Math.max(0.01, (35 - vix) * 1.5 + (fiiFlow / 800) + (mcclellan / 25) - (tdaNorm - 1.5) * 4);
  let wLate = Math.max(0.01, (vix > 14 && vix < 24 ? 20 : 5) - (mcclellan / 30) + 10);
  let wTrans = Math.max(0.01, (tdaNorm > 2.0 ? 18 : 6) + Math.abs(mcclellan) / 20 + 8);
  let wPost = Math.max(0.01, (vix > 25 && mcclellan < -20 ? 25 : 2) + (fiiFlow > 0 ? 10 : 0));
  let wOff = Math.max(0.01, (vix - 16) * 1.8 - (fiiFlow / 600) - (mcclellan / 20) + (tdaNorm - 1.5) * 6);

  const exps = [wOn, wLate, wTrans, wPost, wOff].map((w) => Math.exp(Math.min(50, Math.max(-50, w / 8))));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((e) => Number((e / sumExp).toFixed(4)));

  const regimes = ['Risk-On', 'Late-Cycle', 'Transitional', 'Post-Shock', 'Risk-Off'];
  const pMap = {
    Risk_On: probs[0],
    Late_Cycle: probs[1],
    Transitional: probs[2],
    Post_Shock: probs[3],
    Risk_Off: probs[4]
  };

  // Find dominant
  const sorted = [
    { name: 'Risk-On', p: probs[0] },
    { name: 'Late-Cycle', p: probs[1] },
    { name: 'Transitional', p: probs[2] },
    { name: 'Post-Shock', p: probs[3] },
    { name: 'Risk-Off', p: probs[4] }
  ].sort((a, b) => b.p - a.p);

  // 90% Conformal prediction set
  let cum = 0;
  const predictionSet = [];
  for (const s of sorted) {
    predictionSet.push(s.name);
    cum += s.p;
    if (cum >= 0.90) break;
  }

  // Shannon Entropy
  let entropy = 0;
  probs.forEach((p) => {
    if (p > 1e-6) entropy -= p * Math.log(p);
  });

  return res.status(200).json({
    timestamp_utc: new Date().toISOString(),
    dominant_regime: sorted[0].name,
    conviction: Number((sorted[0].p * 100).toFixed(1)),
    regime_probabilities: pMap,
    conformal_prediction_set_90: predictionSet,
    shannon_entropy_nats: Number(entropy.toFixed(4)),
    inputs: { vix, fiiFlow, mcclellan, tdaNorm },
    author: 'Aaryan Dwivedi'
  });
}
