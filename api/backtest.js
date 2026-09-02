import { BACKTEST_METRICS } from '../src/engine/data.js';

export default function handler(req, res) {
  res.setHeader('Cache-Control', 's-maxage=3600, stale-while-revalidate');

  return res.status(200).json({
    metrics: BACKTEST_METRICS,
    author: 'Aaryan Dwivedi',
    engine_version: 'v1.5.0'
  });
}
