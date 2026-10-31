const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');
const SleeperEngine = require('./public/engine.js');

const PORT = process.env.PORT || 3457;
const PUBLIC_DIR = path.join(__dirname, 'public');

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.svg': 'image/svg+xml'
};

const server = http.createServer(async (req, res) => {
  const parsedUrl = url.parse(req.url, true);
  const pathname = parsedUrl.pathname;

  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  try {
    if (pathname === '/api/optimize') {
      const leagueId = parsedUrl.query.leagueId || SleeperEngine.DEFAULT_LEAGUE_ID;
      const rosterId = Number(parsedUrl.query.rosterId || 3);
      const week = parsedUrl.query.week ? Number(parsedUrl.query.week) : null;
      const marketWeight = parsedUrl.query.marketWeight !== undefined ? Number(parsedUrl.query.marketWeight) : 0.75;
      const objective = parsedUrl.query.objective || 'total_wins';

      const state = await SleeperEngine.getLeagueLiveState(leagueId, week, false, false, marketWeight);
      const result = SleeperEngine.runSimulationLineupOptimizer(state, rosterId, {
        numSims: 10000,
        objective,
        marketWeight
      });
      res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
      res.end(JSON.stringify(result));
      return;
    }

    if (pathname === '/api/backtest') {
      const leagueId = parsedUrl.query.leagueId || SleeperEngine.DEFAULT_LEAGUE_ID;
      const rosterId = Number(parsedUrl.query.rosterId || 3);
      const marketWeight = parsedUrl.query.marketWeight !== undefined ? Number(parsedUrl.query.marketWeight) : 0.75;
      const objective = parsedUrl.query.objective || 'total_wins';

      const bt = await SleeperEngine.runHistoricalBacktest(leagueId, rosterId, {
        marketWeight,
        objective
      });
      res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
      res.end(JSON.stringify(bt));
      return;
    }

    let filePath = pathname === '/' ? path.join(PUBLIC_DIR, 'index.html') : path.join(PUBLIC_DIR, pathname);
    if (!filePath.startsWith(PUBLIC_DIR)) {
      res.writeHead(403);
      res.end('Forbidden');
      return;
    }

    if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
      const ext = path.extname(filePath).toLowerCase();
      res.writeHead(200, { 'Content-Type': MIME_TYPES[ext] || 'application/octet-stream' });
      fs.createReadStream(filePath).pipe(res);
    } else {
      res.writeHead(404);
      res.end('Not Found');
    }
  } catch (err) {
    console.error('Server error:', err);
    res.writeHead(500, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify({ error: err.message }));
  }
});

server.listen(PORT, () => {
  console.log(`Sleeper Sportsbook & Simulation Lineup Optimizer running at http://localhost:${PORT}`);
});
