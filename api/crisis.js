import { CRISIS_SCENARIOS } from '../src/engine/data.js';

export default function handler(req, res) {
  res.setHeader('Cache-Control', 's-maxage=3600, stale-while-revalidate');

  const { id } = req.query;

  if (id) {
    const scenario = CRISIS_SCENARIOS.find((c) => c.id === id);
    if (!scenario) {
      return res.status(404).json({ error: `Crisis scenario '${id}' not found.` });
    }
    return res.status(200).json(scenario);
  }

  return res.status(200).json({
    total_scenarios: CRISIS_SCENARIOS.length,
    scenarios: CRISIS_SCENARIOS
  });
}
