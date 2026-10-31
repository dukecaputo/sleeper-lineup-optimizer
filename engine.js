/**
 * Universal Real-Time Sleeper Fantasy Tracker, Sportsbook + Prediction Market Valuation Engine,
 * and Block-Cholesky Correlated Monte Carlo Lineup Optimizer.
 *
 * Works identically in the Browser (Direct CORS to Sleeper, One Week Season Sportsbook Feed,
 * and ESPN DraftKings Scoreboard APIs) and in Node.js / Serverless.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    const api = factory();
    root.SleeperTrackerEngine = api;
    root.SleeperEngine = api;
  }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const DEFAULT_LEAGUE_ID = '1312258310212165632';
  const UNDERDOG_NFL_URL = 'https://api.underdogfantasy.com/v1/over_under_lines?sport_id=NFL';
  const OWS_SPORTSBOOK_URL = 'https://oneweekseason.com/wp-json/ows-stacks/v1/sportsbook-feed';
  const ESPN_SCOREBOARD_URL = 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard';
  const cache = new Map();

  async function fetchJson(url, options = {}, ttlMs = 8000) {
    const method = options.method || 'GET';
    const body = options.body || '';
    const cacheKey = `${method}:${url}:${body}`;
    const now = Date.now();

    if (ttlMs > 0 && cache.has(cacheKey)) {
      const entry = cache.get(cacheKey);
      if (now - entry.ts < ttlMs) {
        return entry.data;
      }
    }

    const fetchOpts = {
      method,
      headers: {
        Accept: 'application/json',
        ...(options.headers || {})
      }
    };
    if (body) fetchOpts.body = body;
    if (ttlMs <= 10000 && typeof window !== 'undefined') {
      fetchOpts.cache = 'no-store';
    }

    const res = await fetch(url, fetchOpts);
    if (!res.ok) {
      throw new Error(`HTTP ${res.status} fetching ${url}`);
    }
    const data = await res.json();
    cache.set(cacheKey, { ts: Date.now(), data });
    return data;
  }

  function normalizePlayerName(name) {
    return String(name || '')
      .toLowerCase()
      .replace(/\b(jr|sr|ii|iii|iv|v)\b\.?/gi, '')
      .replace(/[^a-z0-9]/g, '');
  }

  function normalizeTeamCode(team) {
    if (!team) return 'FA';
    const t = team.toUpperCase();
    if (t === 'LA') return 'LAR';
    if (t === 'JAC') return 'JAX';
    if (t === 'WSH') return 'WAS';
    return t;
  }

  function computeCustomPoints(stats, scoringSettings) {
    if (!stats || !scoringSettings) return 0;
    let total = 0;
    for (const [key, weight] of Object.entries(scoringSettings)) {
      const val = stats[key];
      if (val !== undefined && val !== null && !Number.isNaN(val)) {
        total += val * weight;
      }
    }
    return total;
  }

  // Mathematical helpers for Underdog Fantasy de-vigging, Normal quantile shift & Poisson inversion
  function americanToRawProb(amStr) {
    if (amStr === null || amStr === undefined) return null;
    const val = parseFloat(String(amStr).trim().replace('+', ''));
    if (Number.isNaN(val)) return null;
    if (val < 0) return (-val) / (-val + 100.0);
    if (val > 0) return 100.0 / (val + 100.0);
    return 0.5;
  }

  function erfApprox(x) {
    const sign = x >= 0 ? 1 : -1;
    const ax = Math.abs(x);
    const t = 1.0 / (1.0 + 0.3275911 * ax);
    const y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-ax * ax);
    return sign * y;
  }

  function invNormCdf(p) {
    const pClamped = Math.max(0.01, Math.min(0.99, Number(p || 0.5)));
    let lo = -3.5;
    let hi = 3.5;
    const sqrt2 = Math.SQRT2;
    for (let i = 0; i < 24; i++) {
      const mid = 0.5 * (lo + hi);
      const cdf = 0.5 * (1.0 + erfApprox(mid / sqrt2));
      if (cdf < pClamped) lo = mid;
      else hi = mid;
    }
    return 0.5 * (lo + hi);
  }

  function poissonOverProb(lam, k) {
    const n = Math.floor(k);
    let s = 0.0;
    let term = Math.exp(-lam);
    for (let i = 0; i <= n; i++) {
      if (i > 0) term *= lam / i;
      s += term;
    }
    return Math.max(0.0, Math.min(1.0, 1.0 - s));
  }

  function invertPoissonMean(lineVal, pOver) {
    const pClamped = Math.max(0.01, Math.min(0.98, Number(pOver || 0.5)));
    if (lineVal < 1.0) {
      return -Math.log(Math.max(1e-4, 1.0 - pClamped));
    }
    let lo = 0.05;
    let hi = 16.0;
    for (let i = 0; i < 26; i++) {
      const mid = 0.5 * (lo + hi);
      if (poissonOverProb(mid, lineVal) < pClamped) lo = mid;
      else hi = mid;
    }
    return 0.5 * (lo + hi);
  }

  function extractUnderdogDevigProb(optMap) {
    const hi = optMap.higher || {};
    const lo = optMap.lower || {};
    const pHi = americanToRawProb(hi.american_price);
    const pLo = americanToRawProb(lo.american_price);
    if (pHi !== null && pLo !== null && (pHi + pLo) > 0) {
      return Math.max(0.02, Math.min(0.98, pHi / (pHi + pLo)));
    }
    if (pHi !== null) {
      return Math.max(0.02, Math.min(0.92, pHi * 0.935));
    }
    const mHi = parseFloat(hi.payout_multiplier);
    const mLo = parseFloat(lo.payout_multiplier);
    if (!Number.isNaN(mHi) && !Number.isNaN(mLo) && mHi > 0 && mLo > 0) {
      const invH = 1.0 / mHi;
      const invL = 1.0 / mLo;
      return Math.max(0.02, Math.min(0.98, invH / (invH + invL)));
    }
    if (!Number.isNaN(mHi) && mHi > 0) {
      return Math.max(0.02, Math.min(0.92, 0.50 / mHi));
    }
    return 0.50;
  }

  /**
   * Fetch & parse Underdog Fantasy's live NFL Over/Under feed (`api.underdogfantasy.com/v1/over_under_lines?sport_id=NFL`).
   * Covers ALL NFL games in the week (TNF, Sunday 1pm, Sunday 4pm, SNF, MNF) and both Offense + IDP players!
   */
  async function fetchUnderdogNflMap(ttlMs = 180000) {
    const cacheKey = 'PROCESSED_UNDERDOG_NFL_MAP';
    const now = Date.now();
    if (ttlMs > 0 && cache.has(cacheKey)) {
      const c = cache.get(cacheKey);
      if (now - c.ts < ttlMs) return c.data;
    }
    let payload = null;
    try {
      payload = await fetchJson(UNDERDOG_NFL_URL, {}, ttlMs);
    } catch (_) {
      try {
        payload = await fetchJson('./underdog_nfl_props.json', {}, ttlMs);
      } catch (__) {
        payload = null;
      }
    }
    if (!payload) return new Map();

    // If payload is already compacted (`players_by_norm`), convert directly
    const byNorm = new Map();
    if (payload.players_by_norm) {
      for (const [normKey, entry] of Object.entries(payload.players_by_norm)) {
        byNorm.set(normKey, entry);
      }
      cache.set(cacheKey, { ts: now, data: byNorm });
      return byNorm;
    }

    const udPlayers = {};
    for (const p of (payload.players || [])) {
      if (p.sport_id !== 'NFL') continue;
      const fullName = `${p.first_name || ''} ${p.last_name || ''}`.trim();
      if (!fullName) continue;
      udPlayers[String(p.id)] = {
        name: fullName,
        norm: normalizePlayerName(fullName),
        pos: String(p.position_name || '')
      };
    }

    const appearances = {};
    for (const a of (payload.appearances || [])) {
      const pid = String(a.player_id || '');
      if (udPlayers[pid]) {
        appearances[String(a.id)] = {
          playerId: pid,
          matchId: a.match_id
        };
      }
    }

    for (const l of (payload.over_under_lines || [])) {
      if (l.status !== 'active') continue;
      const ou = l.over_under || {};
      const astat = ou.appearance_stat || {};
      const stat = String(astat.stat || '');
      if (!stat || stat.startsWith('period_')) continue;
      const app = appearances[String(astat.appearance_id || '')];
      if (!app) continue;
      const pinfo = udPlayers[app.playerId];
      if (!pinfo) continue;
      const statVal = parseFloat(l.stat_value);
      if (Number.isNaN(statVal)) continue;

      const optMap = {};
      for (const o of (l.options || [])) {
        if (o.status && o.status !== 'active') continue;
        const ch = String(o.choice || '').toLowerCase();
        if (ch === 'higher' || ch === 'lower') {
          optMap[ch] = {
            american_price: o.american_price,
            payout_multiplier: o.payout_multiplier
          };
        }
      }
      const pOver = extractUnderdogDevigProb(optMap);
      if (!byNorm.has(pinfo.norm)) {
        byNorm.set(pinfo.norm, {
          name: pinfo.name,
          pos: pinfo.pos,
          stats: {}
        });
      }
      const entry = byNorm.get(pinfo.norm);
      if (!entry.stats[stat] || l.line_type === 'balanced') {
        entry.stats[stat] = {
          stat_value: statVal,
          p_over: Number(pOver.toFixed(4)),
          line_type: l.line_type || 'balanced'
        };
      }
    }

    cache.set(cacheKey, { ts: now, data: byNorm });
    return byNorm;
  }

  /**
   * Convert an Underdog Fantasy player entry (`entry.stats`) into this league's exact custom scoring
   * and structured prop breakdown (`passYds`, `passTd`, `passCmp`, `rushYds`, `rec`, `recYds`, `td`, `anytimeTdProb`, `tackles`, `sacks`, `defInts`).
   */
  function computeLeaguePointsFromUnderdogEntry(pos, udEntry, projStats, scoringSettings) {
    if (!udEntry || !udEntry.stats || !scoringSettings) return null;
    const stats = udEntry.stats;
    const isIdp = ['DL', 'LB', 'DB'].includes(pos);

    if (isIdp) {
      const hasIdpProp = ['tackles_and_assists', 'tackles', 'assists', 'sacks', 'defensive_ints'].some((k) => stats[k] !== undefined);
      if (!hasIdpProp) return null;

      let totalTackles = Number(projStats?.idp_tkl || 4.0);
      if (stats.tackles_and_assists) {
        const sv = Number(stats.tackles_and_assists.stat_value);
        const po = Number(stats.tackles_and_assists.p_over);
        totalTackles = Math.max(0.5, Number((sv + Math.max(1.1, sv * 0.28) * invNormCdf(po)).toFixed(2)));
      } else if (stats.tackles || stats.assists) {
        const solo = stats.tackles ? Number(stats.tackles.stat_value) : 2.5;
        const ast = stats.assists ? Number(stats.assists.stat_value) : 1.5;
        totalTackles = Number((solo + ast).toFixed(2));
      }

      let sacks = Number(projStats?.idp_sack || 0.25);
      if (stats.sacks) {
        sacks = Number(invertPoissonMean(Number(stats.sacks.stat_value), Number(stats.sacks.p_over)).toFixed(2));
      }

      let defInts = Number(projStats?.idp_int || 0.08);
      if (stats.defensive_ints) {
        defInts = Number(invertPoissonMean(Number(stats.defensive_ints.stat_value), Number(stats.defensive_ints.p_over)).toFixed(2));
      }

      // Include ancillary IDP stats from baseline projection (QB hits, TFL, pass def, FF/FR)
      const tfl = Number(projStats?.idp_tkl_loss || (sacks * 0.85 + totalTackles * 0.08));
      const qbHit = Number(projStats?.idp_qb_hit || (sacks * 1.65));
      const passDef = Number(projStats?.idp_pass_def || (defInts * 3.2 + (pos === 'DB' ? 0.45 : 0.15)));
      const ff = Number(projStats?.idp_ff || 0.08);
      const fumRec = Number(projStats?.idp_fum_rec || 0.04);

      const pts =
        totalTackles * (scoringSettings.idp_tkl ?? 0.5) +
        sacks * (scoringSettings.idp_sack ?? 3.0) +
        defInts * (scoringSettings.idp_int ?? 4.0) +
        tfl * (scoringSettings.idp_tkl_loss ?? 0.5) +
        qbHit * (scoringSettings.idp_qb_hit ?? 0.5) +
        passDef * (scoringSettings.idp_pass_def ?? 2.5) +
        ff * (scoringSettings.idp_ff ?? 3.0) +
        fumRec * (scoringSettings.idp_fum_rec ?? 2.0);

      return {
        pts: Math.max(0.5, Number(pts.toFixed(2))),
        props: {
          passYds: 0,
          passTd: 0,
          rushYds: 0,
          rec: 0,
          recYds: 0,
          td: 0,
          anytimeTdProb: 0,
          tackles: totalTackles,
          sacks,
          defInts,
          source: 'Underdog IDP Props'
        }
      };
    }

    // Offensive players (QB, RB, WR, TE)
    const offKeys = ['passing_yds', 'passing_tds', 'passing_ints', 'passing_comps', 'rushing_yds', 'receiving_rec', 'receiving_yds', 'rush_rec_tds', 'rush_rec_yds', 'fantasy_points'];
    const matched = offKeys.filter((k) => stats[k] !== undefined);
    if (matched.length === 0) return null;

    let py = Number(projStats?.pass_yd || 0);
    let ptd = Number(projStats?.pass_td || 0);
    let pcmp = Number(projStats?.pass_cmp || (py > 0 ? py / 11.15 : 0));
    let pint = Number(projStats?.pass_int || 0);
    let ry = Number(projStats?.rush_yd || 0);
    let rec = Number(projStats?.rec || 0);
    let recy = Number(projStats?.rec_yd || 0);
    let td = Number((projStats?.rush_td || 0) + (projStats?.rec_td || 0));
    let anytimeTdProb = td > 0 ? Number((1 - Math.exp(-td)).toFixed(3)) : 0;

    if (stats.passing_yds) {
      const sv = Number(stats.passing_yds.stat_value);
      const po = Number(stats.passing_yds.p_over);
      py = Math.max(0, Number((sv + 42.0 * invNormCdf(po)).toFixed(1)));
      if (!stats.passing_comps) pcmp = Number((py / 11.15).toFixed(1));
    }
    if (stats.passing_comps) {
      const sv = Number(stats.passing_comps.stat_value);
      const po = Number(stats.passing_comps.p_over);
      pcmp = Math.max(0, Number((sv + 3.2 * invNormCdf(po)).toFixed(1)));
    }
    if (stats.passing_tds) {
      ptd = Number(invertPoissonMean(Number(stats.passing_tds.stat_value), Number(stats.passing_tds.p_over)).toFixed(2));
    }
    if (stats.passing_ints) {
      pint = Number(invertPoissonMean(Number(stats.passing_ints.stat_value), Number(stats.passing_ints.p_over)).toFixed(2));
    }
    if (stats.rushing_yds) {
      const sv = Number(stats.rushing_yds.stat_value);
      const po = Number(stats.rushing_yds.p_over);
      ry = Math.max(0, Number((sv * 1.025 + Math.max(7.5, sv * 0.32) * invNormCdf(po)).toFixed(1)));
    }
    if (stats.receiving_rec) {
      const sv = Number(stats.receiving_rec.stat_value);
      const po = Number(stats.receiving_rec.p_over);
      const poisMean = invertPoissonMean(sv, po);
      const normMean = Math.max(0, sv + 0.12 + Math.max(0.85, sv * 0.28) * invNormCdf(po));
      rec = Number((0.55 * poisMean + 0.45 * normMean).toFixed(2));
    }
    if (stats.receiving_yds) {
      const sv = Number(stats.receiving_yds.stat_value);
      const po = Number(stats.receiving_yds.p_over);
      recy = Math.max(0, Number((sv * 1.025 + Math.max(8.0, sv * 0.30) * invNormCdf(po)).toFixed(1)));
    }
    if (stats.rush_rec_yds) {
      const sv = Number(stats.rush_rec_yds.stat_value);
      const po = Number(stats.rush_rec_yds.p_over);
      const comboMean = Math.max(0, sv * 1.025 + Math.max(10.0, sv * 0.28) * invNormCdf(po));
      if (stats.rushing_yds && !stats.receiving_yds) {
        recy = Math.max(0, Number((comboMean - ry).toFixed(1)));
        if (!stats.receiving_rec && recy > 0) rec = Number((recy / 8.5).toFixed(2));
      } else if (stats.receiving_yds && !stats.rushing_yds) {
        ry = Math.max(0, Number((comboMean - recy).toFixed(1)));
      }
    }
    if (stats.rush_rec_tds) {
      const sv = Number(stats.rush_rec_tds.stat_value);
      const po = Number(stats.rush_rec_tds.p_over);
      td = Number(invertPoissonMean(sv, po).toFixed(2));
      anytimeTdProb = sv < 1.0 ? Number(po.toFixed(3)) : Number((1 - Math.exp(-td)).toFixed(3));
    }

    let fumLost = pos === 'QB' ? 0.18 : (pos === 'RB' ? 0.08 : 0.04);
    if (stats.fumbles_lost) {
      fumLost = Number(invertPoissonMean(Number(stats.fumbles_lost.stat_value), Number(stats.fumbles_lost.p_over)).toFixed(2));
    }

    // Calculate bottom-up fantasy points in THIS league's exact scoring settings
    let statPts = 0;
    if (pos === 'QB') {
      statPts += py * (scoringSettings.pass_yd ?? 0.04);
      statPts += pcmp * (scoringSettings.pass_cmp ?? 0.10);
      statPts += ptd * (scoringSettings.pass_td ?? 4.0);
      statPts += pint * (scoringSettings.pass_int ?? -2.0);
      statPts += ry * (scoringSettings.rush_yd ?? 0.1);
      statPts += td * (scoringSettings.rush_td ?? 6.0);
      statPts += fumLost * (scoringSettings.fum_lost ?? -2.0);
      statPts += 0.08 * (scoringSettings.pass_2pt ?? 2.0);
    } else {
      statPts += ry * (scoringSettings.rush_yd ?? 0.1);
      statPts += rec * (scoringSettings.rec ?? 0.5);
      statPts += recy * (scoringSettings.rec_yd ?? 0.1);
      statPts += td * (scoringSettings.rec_td ?? 6.0);
      statPts += fumLost * (scoringSettings.fum_lost ?? -2.0);
      statPts += 0.03 * (scoringSettings.rec_2pt ?? 2.0);
    }

    // Also check Underdog's direct `fantasy_points` line (Underdog native is 0.5 PPR, 4pt Pass TD, -2 INT; add pass_cmp bonus for QBs)
    let udDirectFpts = null;
    if (stats.fantasy_points) {
      const svFp = Number(stats.fantasy_points.stat_value);
      const poFp = Number(stats.fantasy_points.p_over);
      const rawUdFp = Math.max(0, svFp + Math.max(2.2, svFp * 0.25) * invNormCdf(poFp));
      const cmpBonus = pos === 'QB' ? pcmp * (scoringSettings.pass_cmp ?? 0.10) : 0;
      const pprAdjust = pos !== 'QB' ? rec * ((scoringSettings.rec ?? 0.5) - 0.5) : 0;
      udDirectFpts = Number((rawUdFp + cmpBonus + pprAdjust).toFixed(2));
    }

    const hasYardageLine = Boolean(stats.passing_yds || stats.rushing_yds || stats.receiving_yds || stats.receiving_rec || stats.rush_rec_yds);
    let finalPts = statPts;
    if (hasYardageLine && udDirectFpts !== null) {
      finalPts = 0.75 * statPts + 0.25 * udDirectFpts;
    } else if (!hasYardageLine && udDirectFpts !== null) {
      finalPts = udDirectFpts;
    }

    return {
      pts: Math.max(0, Number(finalPts.toFixed(2))),
      props: {
        passYds: Number(py.toFixed(1)),
        passTd: Number(ptd.toFixed(2)),
        passCmp: Number(pcmp.toFixed(1)),
        passInt: Number(pint.toFixed(2)),
        rushYds: Number(ry.toFixed(1)),
        rec: Number(rec.toFixed(2)),
        recYds: Number(recy.toFixed(1)),
        td: Number(td.toFixed(2)),
        anytimeTdProb,
        udFantasyPts: udDirectFpts,
        source: 'Underdog Live'
      }
    };
  }

  /**
   * Convert OWS Sportsbook Player Prop Lines (py, ptd, int, ry, rec, recy, td)
   * directly into THIS Sleeper league's custom scoring_settings!
   */
  function computeLeaguePointsFromSportsbookLine(pos, line, scoringSettings) {
    if (!line || !scoringSettings) return null;
    const py = Number(line.py || 0);
    const ptd = Number(line.ptd || 0);
    const intVal = Number(line.int || 0);
    const ry = Number(line.ry || 0);
    const rec = Number(line.rec || 0);
    const recy = Number(line.recy || 0);
    const td = Number(line.td || 0);

    // Ensure there is actually a posted prop line
    if (py === 0 && ptd === 0 && ry === 0 && rec === 0 && recy === 0 && td === 0) {
      return null;
    }

    let pts = 0;
    if (pos === 'QB') {
      const estCompletions = py > 0 ? py / 11.15 : 0;
      pts += py * (scoringSettings.pass_yd ?? 0.04);
      pts += estCompletions * (scoringSettings.pass_cmp ?? 0.0);
      pts += ptd * (scoringSettings.pass_td ?? 4.0);
      pts += intVal * (scoringSettings.pass_int ?? -2.0);
      pts += ry * (scoringSettings.rush_yd ?? 0.1);
      pts += td * (scoringSettings.rush_td ?? 6.0);
      pts += 0.20 * (scoringSettings.fum_lost ?? -2.0);
      pts += 0.08 * (scoringSettings.pass_2pt ?? 2.0);
    } else {
      pts += ry * (scoringSettings.rush_yd ?? 0.1);
      pts += rec * (scoringSettings.rec ?? 0.5);
      pts += recy * (scoringSettings.rec_yd ?? 0.1);
      pts += td * (scoringSettings.rec_td ?? 6.0);
      const fumRate = pos === 'RB' ? 0.08 : 0.04;
      pts += fumRate * (scoringSettings.fum_lost ?? -2.0);
      pts += 0.03 * (scoringSettings.rec_2pt ?? 2.0);
    }
    return Math.max(0, Number(pts.toFixed(2)));
  }

  /**
   * Synthesize Vegas & CFTC Prediction Market implied prop lines for any active player
   * not listed on Underdog/OWS (e.g. completed TNF games, rotational players, or historical backtest weeks).
   * Guarantees 100% of active players every week have populated market projections & prop lines!
   */
  function synthesizeVegasMarketProps(pos, projStats, marketScaleMult = 1.0) {
    const isIdp = ['DL', 'LB', 'DB'].includes(pos);
    if (isIdp) {
      const tackles = Number(((projStats?.idp_tkl || 4.2) * marketScaleMult).toFixed(1));
      const sacks = Number(((projStats?.idp_sack || 0.25) * marketScaleMult).toFixed(2));
      const defInts = Number(((projStats?.idp_int || 0.08) * marketScaleMult).toFixed(2));
      return {
        passYds: 0,
        passTd: 0,
        rushYds: 0,
        rec: 0,
        recYds: 0,
        td: 0,
        anytimeTdProb: 0,
        tackles,
        sacks,
        defInts,
        source: 'Vegas/CFTC Game Script'
      };
    }
    const py = Number(((projStats?.pass_yd || 0) * marketScaleMult).toFixed(1));
    const ptd = Number(((projStats?.pass_td || 0) * marketScaleMult).toFixed(2));
    const pcmp = Number(((projStats?.pass_cmp || (py > 0 ? py / 11.15 : 0)) * marketScaleMult).toFixed(1));
    const ry = Number(((projStats?.rush_yd || 0) * marketScaleMult).toFixed(1));
    const rec = Number(((projStats?.rec || 0) * marketScaleMult).toFixed(2));
    const recy = Number(((projStats?.rec_yd || 0) * marketScaleMult).toFixed(1));
    const td = Number((((projStats?.rush_td || 0) + (projStats?.rec_td || 0)) * marketScaleMult).toFixed(2));
    const anytimeTdProb = td > 0 ? Number((1 - Math.exp(-td)).toFixed(3)) : 0;
    return {
      passYds: py,
      passTd: ptd,
      passCmp: pcmp,
      rushYds: ry,
      rec,
      recYds: recy,
      td,
      anytimeTdProb,
      source: 'Vegas/CFTC Implied'
    };
  }

  function parseClockSeconds(clockStr) {
    if (!clockStr || typeof clockStr !== 'string') return 0;
    const parts = clockStr.trim().split(':');
    if (parts.length === 2) {
      const mins = parseInt(parts[0], 10) || 0;
      const secs = parseInt(parts[1], 10) || 0;
      return Math.max(0, Math.min(900, mins * 60 + secs));
    }
    return 0;
  }

  /**
   * Analyze NFL game state + enrich with Sportsbook Over/Under, Spread, Implied Team Totals,
   * and Sleeper/Kalshi CFTC Prediction Market Win Probabilities.
   */
  function analyzeGameState(gameObj, externalOddsMap = {}, asOfPregame = false) {
    if (!gameObj) {
      return {
        status: 'bye',
        linearRemFrac: 0,
        effectiveRemFrac: 0,
        elapsedFrac: 1,
        quarter: 0,
        clock: 'BYE',
        homeTeam: null,
        awayTeam: null,
        homeScore: 0,
        awayScore: 0,
        possession: null,
        redZone: null,
        downDistance: null,
        yardLine: null,
        spread: {},
        vegasTotal: 45.0,
        homeSpread: 0,
        homeImpliedTotal: 22.5,
        awayImpliedTotal: 22.5,
        homePredMarketWinProb: 0.5,
        awayPredMarketWinProb: 0.5,
        isDome: false,
        windSpeed: 0,
        otProb: 0
      };
    }

    const meta = gameObj.metadata || {};
    const status = asOfPregame ? 'pre_game' : (gameObj.status || 'complete');
    const homeTeam = normalizeTeamCode(meta.home_team || gameObj.home);
    const awayTeam = normalizeTeamCode(meta.away_team || gameObj.away);
    const homeScore = asOfPregame ? 0 : Number(meta.home_score ?? 0);
    const awayScore = asOfPregame ? 0 : Number(meta.away_score ?? 0);

    // Extract Sportsbook Spread & Prediction Market Moneyline from Sleeper + OWS + ESPN
    const ext = externalOddsMap[homeTeam] || externalOddsMap[awayTeam] || {};
    const slpSpreadObj = meta.spread || meta.pickem_spread || {};
    const homeSpread = slpSpreadObj[homeTeam] !== undefined
      ? Number(slpSpreadObj[homeTeam])
      : (ext.homeTeam === homeTeam && ext.homeSpread !== undefined ? Number(ext.homeSpread) : 0);

    const vegasTotal = ext.total && ext.total > 20 ? Number(ext.total) : 45.5;
    // Home spread is negative when Home is favored (e.g., -3 means Home expected to score 3 more than Away)
    const homeImpliedTotal = Number(((vegasTotal - homeSpread) / 2).toFixed(2));
    const awayImpliedTotal = Number(((vegasTotal + homeSpread) / 2).toFixed(2));

    // Sleeper CFTC Prediction Market Contract probabilities (`moneyline` object in metadata)
    const ml = meta.moneyline || {};
    const rawHomePm = Number(ml[homeTeam] || 0);
    const rawAwayPm = Number(ml[awayTeam] || 0);
    let homePredMarketWinProb = 0.5;
    let awayPredMarketWinProb = 0.5;
    if (rawHomePm > 0 && rawAwayPm > 0) {
      const sumPm = rawHomePm + rawAwayPm;
      homePredMarketWinProb = Number((rawHomePm / sumPm).toFixed(4));
      awayPredMarketWinProb = Number((rawAwayPm / sumPm).toFixed(4));
    } else {
      // Fallback implied win probability from spread (~3.1% per point of spread)
      homePredMarketWinProb = Number(Math.min(0.95, Math.max(0.05, 0.5 - homeSpread * 0.031)).toFixed(4));
      awayPredMarketWinProb = Number((1 - homePredMarketWinProb).toFixed(4));
    }

    const stadiumType = String(meta.stadium_details?.type || '').toLowerCase();
    const isDome = stadiumType === 'indoor' || stadiumType === 'dome' || stadiumType === 'retractable';
    const windSpeed = Number(meta.forecast_wind_speed || 0);

    const baseMarketFields = {
      vegasTotal,
      homeSpread,
      homeImpliedTotal,
      awayImpliedTotal,
      homePredMarketWinProb,
      awayPredMarketWinProb,
      isDome,
      windSpeed,
      rawPredictionMarket: rawHomePm > 0 ? { [homeTeam]: rawHomePm, [awayTeam]: rawAwayPm } : null
    };

    if (!asOfPregame && (status === 'complete' || meta.is_over)) {
      return {
        gameId: gameObj.game_id,
        status: 'complete',
        linearRemFrac: 0,
        effectiveRemFrac: 0,
        elapsedFrac: 1,
        quarter: meta.quarter_num || 4,
        clock: 'FINAL',
        homeTeam,
        awayTeam,
        homeScore,
        awayScore,
        possession: null,
        redZone: null,
        downDistance: null,
        yardLine: null,
        spread: meta.spread || {},
        otProb: 0,
        ...baseMarketFields
      };
    }

    if (status === 'pre_game' || !meta.has_started) {
      let clockLabel = 'PRE';
      if (meta.date_time) {
        try {
          const dt = new Date(meta.date_time);
          const day = dt.toLocaleDateString('en-US', { weekday: 'short' });
          const tm = dt.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
          clockLabel = `${day} ${tm}`;
        } catch (_) {
          clockLabel = 'PRE';
        }
      }
      return {
        gameId: gameObj.game_id,
        status: 'pre_game',
        linearRemFrac: 1.0,
        effectiveRemFrac: 1.0,
        elapsedFrac: 0.0,
        quarter: 0,
        clock: clockLabel,
        dateTime: meta.date_time || null,
        homeTeam,
        awayTeam,
        homeScore: 0,
        awayScore: 0,
        possession: null,
        redZone: null,
        downDistance: null,
        yardLine: null,
        spread: meta.spread || {},
        otProb: 0.05,
        ...baseMarketFields
      };
    }

    const q = Number(meta.quarter_num || 1);
    const qSecRem = parseClockSeconds(meta.time_remaining || '15:00');
    let totalSecRem = q <= 4 ? (4 - q) * 900 + qSecRem : Math.min(600, qSecRem);

    const linearRemFrac = Math.max(0, Math.min(1, totalSecRem / 3600));
    const elapsedFrac = Math.max(0, 1 - linearRemFrac);

    const qWeights = [0.22, 0.28, 0.22, 0.28];
    let effectiveRem = 0;
    if (q <= 4) {
      for (let futureQ = q + 1; futureQ <= 4; futureQ++) {
        effectiveRem += qWeights[futureQ - 1];
      }
      const qFrac = qSecRem / 900;
      let currentQEffectiveFrac = qFrac;
      if (q === 2 || q === 4) {
        const twoMinSec = Math.min(240, qSecRem);
        const normalSec = Math.max(0, qSecRem - 240);
        currentQEffectiveFrac = (twoMinSec * 1.32 + normalSec * 0.88) / 900;
      }
      effectiveRem += qWeights[q - 1] * currentQEffectiveFrac;
    } else {
      effectiveRem = 0.08 * (qSecRem / 600);
    }

    const scoreDiff = Math.abs(homeScore - awayScore);
    const otProb = q <= 4 ? 0.065 * Math.exp(-scoreDiff / 6.5) * (0.5 + 0.5 * elapsedFrac) : 0;
    effectiveRem = Math.min(1.04, effectiveRem + otProb * 0.075);

    return {
      gameId: gameObj.game_id,
      status: 'in_game',
      linearRemFrac,
      effectiveRemFrac: effectiveRem,
      elapsedFrac,
      quarter: q,
      clock: meta.time_remaining || '0:00',
      homeTeam,
      awayTeam,
      homeScore,
      awayScore,
      possession: meta.possession || null,
      redZone: meta.red_zone || null,
      downDistance: meta.down_and_distance || null,
      yardLine: meta.yard_line || null,
      yardLineTerritory: meta.yard_line_territory || null,
      channel: meta.channel || null,
      spread: meta.spread || {},
      otProb,
      ...baseMarketFields
    };
  }

  function canonicalPos(pos) {
    if (!pos) return 'WR';
    const p = pos.toUpperCase();
    if (p === 'QB') return 'QB';
    if (p === 'RB' || p === 'FB') return 'RB';
    if (p === 'WR') return 'WR';
    if (p === 'TE') return 'TE';
    if (['DL', 'DE', 'DT', 'EDGE'].includes(p)) return 'DL';
    if (['LB', 'ILB', 'OLB', 'MLB'].includes(p)) return 'LB';
    if (['DB', 'CB', 'S', 'FS', 'SS'].includes(p)) return 'DB';
    return 'WR';
  }

  function isEligibleForSlot(playerObj, slot) {
    const cPos = playerObj.canonicalPos;
    const rawList = (playerObj.fantasyPositions || [playerObj.position]).map(canonicalPos);
    const hasPos = (target) => cPos === target || rawList.includes(target);

    switch (slot) {
      case 'QB': return hasPos('QB');
      case 'RB': return hasPos('RB');
      case 'WR': return hasPos('WR');
      case 'TE': return hasPos('TE');
      case 'FLEX': return hasPos('RB') || hasPos('WR') || hasPos('TE');
      case 'SUPER_FLEX': return hasPos('QB') || hasPos('RB') || hasPos('WR') || hasPos('TE');
      case 'DL': return hasPos('DL');
      case 'LB': return hasPos('LB');
      case 'DB': return hasPos('DB');
      case 'IDP_FLEX': return hasPos('DL') || hasPos('LB') || hasPos('DB');
      default: return true;
    }
  }

  /**
   * Model a single player, incorporating:
   *  - Sleeper Rotowire pre-game projection (custom league scoring)
   *  - Live Sportsbook Player Prop projection (translated to custom league scoring)
   *  - Prediction Market Win Probability & Vegas Implied Team Total environment adjustment
   *  - Live in-game Bayesian usage, game script, and Red Zone micro-states
   */
  function modelPlayer(playerId, actualPts, projEntry, statEntry, gameState, scoringSettings, sbPropEntry = null, marketWeight = 0.75, udEntry = null, marketSource = 'consensus') {
    const pInfo = (projEntry && projEntry.player) || (statEntry && statEntry.player) || {};
    const projStats = (projEntry && projEntry.stats) || {};
    const liveStats = (statEntry && statEntry.stats) || {};

    const firstName = pInfo.first_name || 'Empty';
    const lastName = pInfo.last_name || (String(playerId) === '0' ? 'Slot' : `#${playerId}`);
    const fullName = String(playerId) === '0' ? 'Empty Slot' : `${firstName} ${lastName}`.trim();
    const rawPos = pInfo.position || 'WR';
    const fantasyPositions = pInfo.fantasy_positions || [rawPos];
    const pos = canonicalPos(rawPos);
    const team = normalizeTeamCode((projEntry && projEntry.team) || (statEntry && statEntry.team) || pInfo.team || 'FA');
    const opponent = normalizeTeamCode((projEntry && projEntry.opponent) || (statEntry && statEntry.opponent) || '');
    const injuryStatus = pInfo.injury_status || null;

    const customLeaguePregameProj = computeCustomPoints(projStats, scoringSettings);
    const isIdpPosBase = ['DL', 'LB', 'DB'].includes(pos);
    const sleeperPregameProj = (!isIdpPosBase && projStats.pts_half_ppr !== undefined)
      ? Number(projStats.pts_half_ppr)
      : customLeaguePregameProj;
    const currentPts = (actualPts !== undefined && actualPts !== null)
      ? Number(actualPts)
      : computeCustomPoints(liveStats, scoringSettings);

    // Team Market Context (Implied Total, Spread, Prediction Market Win %)
    const isHome = Boolean(gameState && gameState.homeTeam === team);
    const teamImpliedTotal = gameState
      ? (isHome ? gameState.homeImpliedTotal : gameState.awayImpliedTotal)
      : 22.5;
    const oppImpliedTotal = gameState
      ? (isHome ? gameState.awayImpliedTotal : gameState.homeImpliedTotal)
      : 22.5;
    const teamSpread = gameState
      ? (isHome ? gameState.homeSpread : -gameState.homeSpread)
      : 0;
    const predMarketWinProb = gameState
      ? (isHome ? gameState.homePredMarketWinProb : gameState.awayPredMarketWinProb)
      : 0.5;
    const vegasTotal = gameState ? gameState.vegasTotal : 45.0;

    // 1. Underdog Fantasy Live Market Props (All NFL Games: Sun 1pm, 4pm, SNF, MNF, TNF + IDP)
    const udParsed = udEntry ? computeLeaguePointsFromUnderdogEntry(pos, udEntry, projStats, scoringSettings) : null;
    const rawUdProj = udParsed ? udParsed.pts : null;

    // 2. OWS Sportsbook Prop Implied Points (Sunday Main Slate)
    const sbLine = sbPropEntry ? sbPropEntry.line : null;
    const rawOwsProj = sbLine ? computeLeaguePointsFromSportsbookLine(pos, sbLine, scoringSettings) : null;
    const owsPropsObj = sbLine
      ? {
          passYds: Number(sbLine.py || 0),
          passTd: Number(sbLine.ptd || 0),
          passCmp: Number(((sbLine.py || 0) / 11.15).toFixed(1)),
          rushYds: Number(sbLine.ry || 0),
          rec: Number(sbLine.rec || 0),
          recYds: Number(sbLine.recy || 0),
          td: Number(sbLine.td || 0),
          anytimeTdProb: sbLine.td > 0 ? Number((1 - Math.exp(-Number(sbLine.td))).toFixed(3)) : 0,
          source: 'OWS Sportsbook'
        }
      : null;

    // 3. Vegas + Sleeper CFTC Prediction Market Environment Projection & Synthesized Props (100% Player Coverage)
    const isDome = Boolean(gameState && gameState.isDome);
    const windSpeed = Number((gameState && gameState.windSpeed) || 0);
    const spreadImpliedProb = Math.min(0.95, Math.max(0.05, 0.5 - teamSpread * 0.031));
    const pmEdge = predMarketWinProb - spreadImpliedProb;

    let envScaleMult = 1.0;
    if (!isIdpPosBase) {
      const totalTilt = Math.max(0.95, Math.min(1.06, Math.pow(teamImpliedTotal / 22.5, 0.11)));
      const pmScriptMod = 1.0 + pmEdge * (pos === 'RB' ? 0.22 : 0.14);
      const projRushAtt = Number(projStats.rush_att || 0);
      const rbVolumeBonus = (pos === 'RB' && projRushAtt >= 11 && predMarketWinProb >= 0.45) ? 1.025 : 1.0;
      const weatherMod = (!isDome && windSpeed >= 16 && ['QB', 'WR'].includes(pos))
        ? 0.965
        : (isDome && ['QB', 'WR'].includes(pos) ? 1.012 : 1.0);
      envScaleMult = totalTilt * pmScriptMod * rbVolumeBonus * weatherMod;
    } else {
      const oppPassHeavy = predMarketWinProb >= 0.55 ? 1.03 : 1.0;
      const gamePace = Math.max(0.95, Math.min(1.05, vegasTotal / 45.0));
      envScaleMult = pos === 'DL' ? oppPassHeavy : gamePace;
    }

    const vegasMarketProj = customLeaguePregameProj > 0
      ? Number((customLeaguePregameProj * envScaleMult).toFixed(2))
      : 0;
    const vegasPropsObj = synthesizeVegasMarketProps(pos, projStats, envScaleMult);

    // Combine Underdog + OWS Sportsbook + Vegas/CFTC according to `marketSource` so 100% of active players have props & projections
    let activeDirectMarketProj = null;
    let activePropsObj = vegasPropsObj;
    let marketSourceLabel = `PM + Vegas (${teamImpliedTotal.toFixed(1)} pts)`;

    if (marketSource === 'underdog') {
      if (rawUdProj !== null && rawUdProj > 0) {
        activeDirectMarketProj = rawUdProj;
        activePropsObj = udParsed.props;
        marketSourceLabel = udParsed.props.source;
      }
    } else if (marketSource === 'ows') {
      if (rawOwsProj !== null && rawOwsProj > 0) {
        activeDirectMarketProj = rawOwsProj;
        activePropsObj = owsPropsObj;
        marketSourceLabel = 'OWS Sportsbook';
      }
    } else if (marketSource !== 'vegas') {
      // 'consensus': Blend Underdog Live Props + OWS Sportsbook when both exist; otherwise use whichever exists!
      if (rawUdProj !== null && rawUdProj > 0 && rawOwsProj !== null && rawOwsProj > 0) {
        activeDirectMarketProj = Number((0.55 * rawUdProj + 0.45 * rawOwsProj).toFixed(2));
        activePropsObj = {
          passYds: Number((0.55 * udParsed.props.passYds + 0.45 * owsPropsObj.passYds).toFixed(1)),
          passTd: Number((0.55 * udParsed.props.passTd + 0.45 * owsPropsObj.passTd).toFixed(2)),
          passCmp: udParsed.props.passCmp,
          rushYds: Number((0.55 * udParsed.props.rushYds + 0.45 * owsPropsObj.rushYds).toFixed(1)),
          rec: Number((0.55 * udParsed.props.rec + 0.45 * owsPropsObj.rec).toFixed(2)),
          recYds: Number((0.55 * udParsed.props.recYds + 0.45 * owsPropsObj.recYds).toFixed(1)),
          td: Number((0.55 * udParsed.props.td + 0.45 * owsPropsObj.td).toFixed(2)),
          anytimeTdProb: Number((0.55 * udParsed.props.anytimeTdProb + 0.45 * owsPropsObj.anytimeTdProb).toFixed(3)),
          udFantasyPts: udParsed.props.udFantasyPts,
          source: 'Underdog + Sportsbook'
        };
        marketSourceLabel = 'Underdog + Sportsbook';
      } else if (rawUdProj !== null && rawUdProj > 0) {
        activeDirectMarketProj = rawUdProj;
        activePropsObj = udParsed.props;
        marketSourceLabel = udParsed.props.source;
      } else if (rawOwsProj !== null && rawOwsProj > 0) {
        activeDirectMarketProj = rawOwsProj;
        activePropsObj = owsPropsObj;
        marketSourceLabel = 'OWS Sportsbook';
      }
    }

    let marketImpliedPregame = customLeaguePregameProj;
    let displayMarketProj = vegasMarketProj;
    if (activeDirectMarketProj !== null && activeDirectMarketProj > 0) {
      const pmScriptBoost = pos === 'RB' ? (1 + pmEdge * 0.18) : (['QB', 'WR', 'TE'].includes(pos) ? (1 + pmEdge * 0.08) : 1.0);
      const pureMarketEst = activeDirectMarketProj * pmScriptBoost;
      displayMarketProj = Number(pureMarketEst.toFixed(2));
      marketImpliedPregame = marketWeight * pureMarketEst + (1 - marketWeight) * customLeaguePregameProj;
    } else if (customLeaguePregameProj > 0 && gameState && gameState.status !== 'bye') {
      displayMarketProj = vegasMarketProj;
      marketImpliedPregame = marketWeight * vegasMarketProj + (1 - marketWeight) * customLeaguePregameProj;
    }

    const isBye = String(playerId) === '0' || !gameState || gameState.status === 'bye' || (!opponent && (!gameState || !gameState.homeTeam));
    if (isBye || gameState.status === 'complete') {
      const finalStatus = isBye ? 'bye' : 'complete';
      return {
        playerId,
        fullName,
        shortName: String(playerId) === '0' ? 'Empty' : `${firstName.charAt(0)}. ${lastName}`,
        position: rawPos,
        fantasyPositions,
        canonicalPos: pos,
        team,
        opponent,
        isHome,
        injuryStatus,
        gameStatus: finalStatus,
        gameId: isBye ? null : gameState.gameId,
        quarter: isBye ? 0 : gameState.quarter,
        clock: isBye ? 'BYE' : 'FINAL',
        gameScore: isBye ? 'BYE WEEK' : 'FINAL',
        hasPossession: false,
        inRedZone: false,
        currentPts: isBye ? 0 : currentPts,
        pregameProj: isBye ? 0 : Number(marketImpliedPregame.toFixed(2)),
        sleeperPregameProj: isBye ? 0 : Number(sleeperPregameProj.toFixed(2)),
        sportsbookProj: isBye ? 0 : displayMarketProj,
        underdogProj: rawUdProj,
        owsSportsbookProj: rawOwsProj,
        vegasMarketProj: isBye ? 0 : vegasMarketProj,
        sportsbookProps: isBye ? null : activePropsObj,
        sportsbookLine: sbLine,
        marketSourceLabel: isBye ? 'BYE' : `${marketSourceLabel} (Final)`,
        teamImpliedTotal,
        teamImpliedPts: teamImpliedTotal,
        oppImpliedTotal,
        vegasTotal,
        teamSpread,
        predMarketWinProb,
        sleeperProj: isBye ? 0 : currentPts,
        improvedProj: isBye ? 0 : currentPts,
        floor: isBye ? 0 : currentPts,
        ceiling: isBye ? 0 : currentPts,
        remMeanSleeper: 0,
        remMeanImproved: 0,
        remSdSleeper: 0,
        remSdImproved: 0,
        skewness: 0,
        factors: {
          linearRemFrac: 0,
          effectiveRemFrac: 0,
          bayesUsageMult: 1,
          gameScriptMult: 1,
          driveBonus: 0,
          reasons: isBye ? ['On BYE / Empty'] : []
        }
      };
    }

    const isOut = ['Out', 'IR', 'Doubtful', 'Suspension', 'COV'].includes(injuryStatus);
    const linearRem = isOut ? 0 : gameState.linearRemFrac;
    const effectiveRem = isOut ? 0 : gameState.effectiveRemFrac;
    const elapsed = gameState.elapsedFrac;

    const remMeanSleeper = sleeperPregameProj * linearRem;
    const sleeperProj = currentPts + remMeanSleeper;

    const reasons = [];
    let bayesUsageMult = 1.0;
    let gameScriptMult = 1.0;
    let driveBonus = 0.0;

    if (displayMarketProj > 0) {
      const sbDiff = marketImpliedPregame - sleeperPregameProj;
      if (Math.abs(sbDiff) >= 0.5) {
        reasons.push(`${marketSourceLabel} ${sbDiff >= 0 ? '+' : ''}${sbDiff.toFixed(1)} vs Slp`);
      }
    }

    if (isOut) {
      reasons.push(`Status: ${injuryStatus}`);
    } else if (gameState.status === 'in_game' && marketImpliedPregame > 0.5) {
      const isIdp = ['DL', 'LB', 'DB'].includes(pos);
      const tmSnaps = isIdp ? (liveStats.tm_def_snp || 0) : (liveStats.tm_off_snp || 0);
      const plSnaps = isIdp ? (liveStats.def_snp || 0) : (liveStats.off_snp || 0);

      let obsOpportunity = 0;
      let expOpportunityFull = 0;
      if (pos === 'QB') {
        obsOpportunity = (liveStats.pass_att || 0) + (liveStats.rush_att || 0) * 1.5;
        expOpportunityFull = (projStats.pass_att || 30) + (projStats.rush_att || 3) * 1.5;
      } else if (pos === 'RB') {
        obsOpportunity = (liveStats.rush_att || 0) + (liveStats.rec_tgt || liveStats.rec || 0) * 1.6;
        expOpportunityFull = (projStats.rush_att || 10) + (projStats.rec || 2) * 1.9;
      } else if (pos === 'WR' || pos === 'TE') {
        obsOpportunity = (liveStats.rec_tgt || (liveStats.rec || 0) * 1.45) + (liveStats.rush_att || 0);
        expOpportunityFull = (projStats.rec || 3) * 1.55 + (projStats.rush_att || 0);
      } else {
        obsOpportunity = (liveStats.idp_tkl || 0) + (liveStats.idp_qb_hit || 0) * 1.5 + (liveStats.idp_pass_def || 0) * 1.5;
        expOpportunityFull = (projStats.idp_tkl || 4) + (projStats.idp_qb_hit || 0.5) * 1.5 + (projStats.idp_pass_def || 0.3) * 1.5;
      }

      if (elapsed >= 0.18) {
        const expOppSoFar = Math.max(0.5, expOpportunityFull * elapsed);
        const rawVolRatio = obsOpportunity / expOppSoFar;
        const clampedRatio = Math.max(0.35, Math.min(1.75, rawVolRatio));
        const kappa = 1.45;
        bayesUsageMult = (kappa * 1.0 + elapsed * clampedRatio) / (kappa + elapsed);

        if (tmSnaps >= 18 && plSnaps === 0 && currentPts === 0 && obsOpportunity === 0) {
          bayesUsageMult = Math.max(0.12, Math.pow(1 - elapsed, 2.2));
          reasons.push('0 snaps recorded so far');
        } else if (bayesUsageMult >= 1.07) {
          reasons.push(`High live usage (+${Math.round((bayesUsageMult - 1) * 100)}%)`);
        } else if (bayesUsageMult <= 0.93) {
          reasons.push(`Low live usage (${Math.round((bayesUsageMult - 1) * 100)}%)`);
        }
      }

      const teamScore = isHome ? gameState.homeScore : gameState.awayScore;
      const oppScore = isHome ? gameState.awayScore : gameState.homeScore;
      const scoreDiff = teamScore - oppScore;

      const scriptWeight = Math.min(1.0, 0.35 + 0.85 * elapsed);
      if (scoreDiff <= -4) {
        const trailMag = Math.min(1.0, Math.abs(scoreDiff) / 16);
        if (pos === 'QB' || pos === 'WR' || pos === 'TE') {
          gameScriptMult = 1.0 + 0.16 * trailMag * scriptWeight;
          if (gameScriptMult >= 1.05) reasons.push(`Trailing script (+${Math.round((gameScriptMult - 1) * 100)}% pass vol)`);
        } else if (pos === 'RB') {
          const recShare = (projStats.rec || 0) / Math.max(1, (projStats.rush_att || 10) + (projStats.rec || 0));
          gameScriptMult = 1.0 - (0.14 * (1 - recShare * 1.5)) * trailMag * scriptWeight;
        } else if (pos === 'LB' || pos === 'DB') {
          gameScriptMult = 1.0 + 0.06 * trailMag * scriptWeight;
        }
      } else if (scoreDiff >= 6) {
        const leadMag = Math.min(1.0, scoreDiff / 16);
        if (pos === 'RB') {
          gameScriptMult = 1.0 + 0.16 * leadMag * scriptWeight;
          if (gameScriptMult >= 1.05) reasons.push(`Positive run script (+${Math.round((gameScriptMult - 1) * 100)}%)`);
        } else if (pos === 'QB' || pos === 'WR' || pos === 'TE') {
          gameScriptMult = 1.0 - 0.11 * leadMag * scriptWeight;
        } else if (pos === 'DL') {
          gameScriptMult = 1.0 + 0.12 * leadMag * scriptWeight;
          if (gameScriptMult >= 1.05) reasons.push('Pass-rush script boost');
        }
      }

      const isIdpPos = ['DL', 'LB', 'DB'].includes(pos);
      if (!isIdpPos && gameState.possession === team) {
        if (gameState.redZone === team) {
          const tdShare = Math.min(0.65, Math.max(0.08, marketImpliedPregame / 24));
          driveBonus = pos === 'QB' ? 0.68 : (0.55 * tdShare + 0.18);
          reasons.push('Live in Red Zone');
        } else {
          driveBonus = pos === 'QB' ? 0.22 : 0.14 * Math.min(1, marketImpliedPregame / 14);
          reasons.push('Has possession');
        }
      } else if (isIdpPos && gameState.possession && gameState.possession !== team) {
        driveBonus = 0.12;
      }

      if (effectiveRem - linearRem >= 0.025) {
        reasons.push('2-min/OT pace boost');
      }
    }

    const remMeanImproved = isOut
      ? 0
      : Math.max(0, marketImpliedPregame * effectiveRem * bayesUsageMult * gameScriptMult + driveBonus);
    const improvedProj = currentPts + remMeanImproved;

    const posVolatility = {
      QB: { cv: 0.40, baseSd: 2.2, skew: 0.55 },
      RB: { cv: 0.52, baseSd: 2.0, skew: 0.95 },
      WR: { cv: 0.62, baseSd: 2.1, skew: 1.15 },
      TE: { cv: 0.66, baseSd: 1.8, skew: 1.25 },
      DL: { cv: 0.74, baseSd: 1.5, skew: 1.35 },
      LB: { cv: 0.52, baseSd: 1.4, skew: 0.85 },
      DB: { cv: 0.66, baseSd: 1.4, skew: 1.15 }
    }[pos] || { cv: 0.60, baseSd: 1.8, skew: 1.0 };

    const fullGameSd = marketImpliedPregame > 0.2
      ? Math.sqrt(Math.pow(posVolatility.cv * marketImpliedPregame, 2) + Math.pow(posVolatility.baseSd, 2))
      : 0;

    const remSdSleeper = fullGameSd * Math.pow(linearRem, 0.85);
    const remSdImproved = isOut
      ? 0
      : fullGameSd * Math.sqrt(effectiveRem * bayesUsageMult * gameScriptMult) + (driveBonus > 0.3 ? 0.65 : 0);

    const dynamicSkew = effectiveRem > 0
      ? Math.min(2.4, posVolatility.skew / Math.pow(Math.max(0.12, effectiveRem), 0.28))
      : 0;

    const floorVal = Math.max(currentPts, Number((improvedProj - 1.2816 * remSdImproved).toFixed(2)));
    const ceilingVal = Number((improvedProj + 1.2816 * remSdImproved).toFixed(2));

    return {
      playerId,
      fullName,
      shortName: `${firstName.charAt(0)}. ${lastName}`,
      position: rawPos,
      fantasyPositions,
      canonicalPos: pos,
      team,
      opponent,
      isHome,
      injuryStatus,
      gameStatus: gameState.status,
      gameId: gameState.gameId,
      quarter: gameState.quarter,
      clock: gameState.clock,
      gameScore: gameState.status === 'in_game'
        ? `${gameState.awayTeam} ${gameState.awayScore} - ${gameState.homeTeam} ${gameState.homeScore}`
        : (gameState.status === 'pre_game' ? `${gameState.awayTeam} @ ${gameState.homeTeam}` : 'FINAL'),
      hasPossession: gameState.possession === team,
      inRedZone: gameState.redZone === team,
      currentPts,
      pregameProj: Number(marketImpliedPregame.toFixed(2)),
      sleeperPregameProj: Number(sleeperPregameProj.toFixed(2)),
      sportsbookProj: displayMarketProj,
      underdogProj: rawUdProj,
      owsSportsbookProj: rawOwsProj,
      vegasMarketProj,
      sportsbookProps: activePropsObj,
      sportsbookLine: sbLine,
      marketSourceLabel,
      teamImpliedTotal,
      teamImpliedPts: teamImpliedTotal,
      oppImpliedTotal,
      vegasTotal,
      teamSpread,
      predMarketWinProb,
      sleeperProj: Number(sleeperProj.toFixed(2)),
      improvedProj: Number(improvedProj.toFixed(2)),
      floor: floorVal,
      ceiling: ceilingVal,
      remMeanSleeper: Number(remMeanSleeper.toFixed(3)),
      remMeanImproved: Number(remMeanImproved.toFixed(3)),
      remSdSleeper: Number(remSdSleeper.toFixed(3)),
      remSdImproved: Number(remSdImproved.toFixed(3)),
      skewness: Number(dynamicSkew.toFixed(3)),
      factors: {
        linearRemFrac: linearRem,
        effectiveRemFrac: effectiveRem,
        bayesUsageMult,
        gameScriptMult,
        driveBonus,
        reasons
      }
    };
  }

  class FastRNG {
    constructor(seed = 1337) {
      this.state = seed >>> 0;
      this.spare = null;
      this.hasSpare = false;
    }
    nextUniform() {
      let t = (this.state += 0x6D2B79F5);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
    nextGaussian() {
      if (this.hasSpare) {
        this.hasSpare = false;
        return this.spare;
      }
      let u, v, s;
      do {
        u = this.nextUniform() * 2 - 1;
        v = this.nextUniform() * 2 - 1;
        s = u * u + v * v;
      } while (s >= 1 || s === 0);
      const mul = Math.sqrt((-2.0 * Math.log(s)) / s);
      this.spare = v * mul;
      this.hasSpare = true;
      return u * mul;
    }
  }

  function sampleSkewedRemaining(z, mu, sd, skew) {
    if (mu <= 0.001 && sd <= 0.001) return 0;
    const alpha = Math.min(0.36, Math.max(0.05, skew * 0.22));
    const normFactor = Math.sqrt(1 + 2 * alpha * alpha);
    const zEff = z < -1 / (2 * alpha) ? -1 / (2 * alpha) : z;
    const x = (zEff + alpha * (zEff * zEff - 1)) / normFactor;
    return mu + sd * x;
  }

  /**
   * Pairwise NFL Player Correlation Model (incorporating Vegas Game Total, Spread,
   * Stack Synergy, and Within-Team Target / Red-Zone TD Cannibalization).
   */
  function computePairwisePlayerCorrelation(pA, pB) {
    if (!pA || !pB || pA.playerId === pB.playerId) return 1.0;
    if (!pA.gameId || !pB.gameId || pA.gameId !== pB.gameId) return 0.0;

    const sameTeam = pA.team === pB.team;
    const posA = pA.canonicalPos;
    const posB = pB.canonicalPos;
    const isIdpA = ['DL', 'LB', 'DB'].includes(posA);
    const isIdpB = ['DL', 'LB', 'DB'].includes(posB);

    const vegasTotal = pA.vegasTotal || 45.5;
    const spreadMag = Math.abs(pA.teamSpread || 3.0);
    const totalDelta = Math.max(-0.25, Math.min(0.35, (vegasTotal - 45.0) / 14.0));
    const closenessBonus = Math.max(-0.15, Math.min(0.15, (7.0 - spreadMag) / 14.0));
    const shootoutFactor = 0.08 * totalDelta + 0.05 * closenessBonus;

    // 1. Offensive Skill Player Pairs
    if (!isIdpA && !isIdpB) {
      if (sameTeam) {
        // QB + WR / TE Stack: strong positive passing correlation
        if ((posA === 'QB' && (posB === 'WR' || posB === 'TE')) ||
            (posB === 'QB' && (posA === 'WR' || posA === 'TE'))) {
          const isUnderdog = (pA.teamSpread || 0) >= 3.0;
          const baseStack = (posA === 'TE' || posB === 'TE') ? 0.34 : 0.38;
          return Number(Math.min(0.65, baseStack + 0.5 * shootoutFactor + (isUnderdog ? 0.03 : 0)).toFixed(3));
        }
        // QB + RB on same team: mild shared drive efficiency offset by rushing vs passing TD split
        if ((posA === 'QB' && posB === 'RB') || (posB === 'QB' && posA === 'RB')) {
          return Number((0.08 + 0.25 * shootoutFactor).toFixed(3));
        }
        // Two RBs on same team: strong carry & goal-line cannibalization
        if (posA === 'RB' && posB === 'RB') {
          return -0.24;
        }
        // Same-team RB + WR / RB + TE: run-vs-pass script & red-zone TD cannibalization
        if (posA === 'RB' || posB === 'RB') {
          return -0.14;
        }
        // Same-team WR + WR or WR + TE: target & red-zone TD cannibalization
        return -0.11;
      } else {
        // Opposing team offensive players in the same game
        if (posA === 'RB' && posB === 'RB') {
          // Opposing RBs have negative game-script correlation (one team leading/running = other trailing/passing)
          return -0.10;
        }
        if (posA === 'RB' || posB === 'RB') {
          return -0.03;
        }
        // Opposing passing attacks (QB/WR/TE vs Opp QB/WR/TE) correlate positively in shootouts!
        return Number((0.14 + 0.85 * shootoutFactor).toFixed(3));
      }
    }

    // 2. IDP Correlations
    if (isIdpA && isIdpB) {
      return sameTeam ? -0.05 : 0.04;
    }

    const idpPos = isIdpA ? posA : posB;
    const offPos = isIdpA ? posB : posA;
    if (sameTeam) {
      // Pass rusher (DL) benefits when own offense scores & leads -> opponent forced to pass
      if (idpPos === 'DL' && ['QB', 'RB', 'WR'].includes(offPos)) return 0.10;
      return 0.02;
    } else {
      // Opposing team IDP vs Offense
      if (idpPos === 'DL' && offPos === 'QB') return -0.12;
      if ((idpPos === 'LB' || idpPos === 'DB') && ['RB', 'WR', 'TE'].includes(offPos)) return 0.09;
      return 0.0;
    }
  }

  /**
   * Compute exact lower-triangular Cholesky factor L of a symmetric correlation matrix `corr`
   * using robust Modified Cholesky with diagonal damping to guarantee Positive Definiteness.
   */
  function computeCholeskyFactor(corr) {
    const n = corr.length;
    const L = Array.from({ length: n }, () => new Float64Array(n));

    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let sum = corr[i][j];
        for (let k = 0; k < j; k++) {
          sum -= L[i][k] * L[j][k];
        }
        if (i === j) {
          L[i][j] = Math.sqrt(Math.max(0.01, sum));
        } else {
          L[i][j] = sum / L[j][j];
        }
      }
      // Normalize row i to unit Euclidean norm so marginal variance is identically 1.0!
      let rowNormSq = 0;
      for (let j = 0; j <= i; j++) rowNormSq += L[i][j] * L[i][j];
      const invNorm = 1 / Math.sqrt(Math.max(1e-8, rowNormSq));
      for (let j = 0; j <= i; j++) L[i][j] *= invNorm;
    }
    return L;
  }

  function runJointLeagueSimulation(rostersData, numSims = 10000) {
    const rng = new FastRNG(20261031);
    const numTeams = rostersData.length;

    // Collect all unique active starter players across the 10 rosters, grouped by NFL gameId
    const playersById = new Map();
    const gameBuckets = new Map();

    for (const r of rostersData) {
      for (const p of r.starters) {
        if (p.remMeanImproved > 0.001 || p.remSdImproved > 0.001) {
          if (!playersById.has(p.playerId)) {
            playersById.set(p.playerId, p);
            const gKey = p.gameId || `solo_${p.playerId}`;
            if (!gameBuckets.has(gKey)) gameBuckets.set(gKey, []);
            gameBuckets.get(gKey).push(p);
          }
        }
      }
    }

    // Build per-game Block-Cholesky correlation matrices
    const gameBlocks = [];
    const playerSimSlot = new Map();
    let totalActivePlayers = 0;

    for (const [gKey, bucket] of gameBuckets.entries()) {
      const m = bucket.length;
      const corr = Array.from({ length: m }, () => new Float64Array(m));
      for (let i = 0; i < m; i++) {
        corr[i][i] = 1.0;
        for (let j = i + 1; j < m; j++) {
          const c = computePairwisePlayerCorrelation(bucket[i], bucket[j]);
          corr[i][j] = c;
          corr[j][i] = c;
        }
      }
      const L = computeCholeskyFactor(corr);
      const globalIndices = new Int32Array(m);
      for (let i = 0; i < m; i++) {
        const gIdx = totalActivePlayers++;
        globalIndices[i] = gIdx;
        playerSimSlot.set(bucket[i].playerId, gIdx);
      }
      gameBlocks.push({ m, L, players: bucket, globalIndices });
    }

    const compiledTeams = rostersData.map((r, tIdx) => {
      const starterSimIndices = [];
      const activePlayers = [];
      for (const p of r.starters) {
        if (p.remMeanImproved > 0.001 || p.remSdImproved > 0.001) {
          starterSimIndices.push(playerSimSlot.get(p.playerId));
          activePlayers.push(p);
        }
      }
      return {
        tIdx,
        rosterId: r.rosterId,
        matchupId: r.matchupId,
        currentPts: r.currentPts,
        starterSimIndices: new Int32Array(starterSimIndices),
        activePlayers
      };
    });

    const matchupMap = new Map();
    for (let i = 0; i < numTeams; i++) {
      const mid = rostersData[i].matchupId;
      if (!matchupMap.has(mid)) matchupMap.set(mid, []);
      matchupMap.get(mid).push(i);
    }
    const oppIndex = new Int32Array(numTeams).fill(-1);
    for (const pair of matchupMap.values()) {
      if (pair.length === 2) {
        oppIndex[pair[0]] = pair[1];
        oppIndex[pair[1]] = pair[0];
      }
    }

    const h2hWinsImp = new Float64Array(numTeams);
    const h2hWinsSlp = new Float64Array(numTeams);
    const medianWinsImp = new Float64Array(numTeams);
    const medianWinsSlp = new Float64Array(numTeams);
    const winBothImp = new Float64Array(numTeams);
    const winOneImp = new Float64Array(numTeams);
    const winZeroImp = new Float64Array(numTeams);

    const scoreSumImp = new Float64Array(numTeams);
    const scoreSqSumImp = new Float64Array(numTeams);
    const teamSamplesImp = Array.from({ length: numTeams }, () => new Float64Array(numSims));
    const medianSamplesImp = new Float64Array(numSims);
    const medianSamplesSlp = new Float64Array(numSims);

    const playerRemImp = new Float64Array(totalActivePlayers);
    const playerRemSlp = new Float64Array(totalActivePlayers);
    const epsBuf = new Float64Array(32);

    const trialScoresImp = new Float64Array(numTeams);
    const trialScoresSlp = new Float64Array(numTeams);
    const sortedImp = new Float64Array(numTeams);
    const sortedSlp = new Float64Array(numTeams);

    for (let s = 0; s < numSims; s++) {
      // 1. Sample correlated player outcomes block-by-block per NFL game
      for (let b = 0; b < gameBlocks.length; b++) {
        const blk = gameBlocks[b];
        const m = blk.m;
        const L = blk.L;
        for (let i = 0; i < m; i++) epsBuf[i] = rng.nextGaussian();
        for (let i = 0; i < m; i++) {
          let zCorr = 0;
          const rowL = L[i];
          for (let j = 0; j <= i; j++) zCorr += rowL[j] * epsBuf[j];
          const pl = blk.players[i];
          const gIdx = blk.globalIndices[i];
          playerRemImp[gIdx] = sampleSkewedRemaining(zCorr, pl.remMeanImproved, pl.remSdImproved, pl.skewness);
          playerRemSlp[gIdx] = pl.remMeanSleeper + pl.remSdSleeper * epsBuf[i];
        }
      }

      // 2. Sum each team's simulated score
      for (let i = 0; i < numTeams; i++) {
        const ct = compiledTeams[i];
        let remImp = 0;
        let remSlp = 0;
        const indices = ct.starterSimIndices;
        for (let k = 0; k < indices.length; k++) {
          const idx = indices[k];
          remImp += playerRemImp[idx];
          remSlp += playerRemSlp[idx];
        }
        const totImp = ct.currentPts + remImp;
        const totSlp = ct.currentPts + remSlp;
        trialScoresImp[i] = totImp;
        trialScoresSlp[i] = totSlp;
        sortedImp[i] = totImp;
        sortedSlp[i] = totSlp;

        teamSamplesImp[i][s] = totImp;
        scoreSumImp[i] += totImp;
        scoreSqSumImp[i] += totImp * totImp;
      }

      sortedImp.sort();
      sortedSlp.sort();
      const midRight = Math.floor(numTeams / 2);
      const midLeft = Math.max(0, midRight - 1);
      const medImp = 0.5 * (sortedImp[midLeft] + sortedImp[midRight]);
      const medSlp = 0.5 * (sortedSlp[midLeft] + sortedSlp[midRight]);
      medianSamplesImp[s] = medImp;
      medianSamplesSlp[s] = medSlp;

      for (let i = 0; i < numTeams; i++) {
        const opp = oppIndex[i];
        let wonH2HImp = 0;
        if (opp >= 0) {
          if (trialScoresImp[i] > trialScoresImp[opp]) wonH2HImp = 1;
          else if (trialScoresImp[i] === trialScoresImp[opp]) wonH2HImp = 0.5;

          if (trialScoresSlp[i] > trialScoresSlp[opp]) h2hWinsSlp[i] += 1;
          else if (trialScoresSlp[i] === trialScoresSlp[opp]) h2hWinsSlp[i] += 0.5;
        }
        h2hWinsImp[i] += wonH2HImp;

        let wonMedImp = 0;
        if (trialScoresImp[i] > medImp) wonMedImp = 1;
        else if (trialScoresImp[i] === medImp) wonMedImp = 0.5;
        medianWinsImp[i] += wonMedImp;

        if (trialScoresSlp[i] > medSlp) medianWinsSlp[i] += 1;
        else if (trialScoresSlp[i] === medSlp) medianWinsSlp[i] += 0.5;

        if (wonH2HImp === 1 && wonMedImp === 1) winBothImp[i] += 1;
        else if (wonH2HImp === 0 && wonMedImp === 0) winZeroImp[i] += 1;
        else winOneImp[i] += 1;
      }
    }

    medianSamplesImp.sort();
    const medianMeanImp = medianSamplesImp.reduce((a, b) => a + b, 0) / numSims;
    const medianSdImp = Math.sqrt(
      medianSamplesImp.reduce((acc, v) => acc + (v - medianMeanImp) ** 2, 0) / numSims
    );

    const midR = Math.floor(numTeams / 2);
    const midL = Math.max(0, midR - 1);
    const staticSleeperProjs = rostersData.map((r) => r.sleeperTotalProj).sort((a, b) => a - b);
    const sleeperStaticMedian = 0.5 * ((staticSleeperProjs[midL] || 0) + (staticSleeperProjs[midR] || 0));

    const staticImprovedProjs = rostersData.map((r) => r.improvedTotalProj).sort((a, b) => a - b);
    const improvedStaticMedian = 0.5 * ((staticImprovedProjs[midL] || 0) + (staticImprovedProjs[midR] || 0));

    const minMed = medianSamplesImp[Math.floor(numSims * 0.005)];
    const maxMed = medianSamplesImp[Math.floor(numSims * 0.995)];
    const numBins = 24;
    const binWidth = Math.max(0.25, (maxMed - minMed) / numBins);
    const medianHistogram = [];
    for (let b = 0; b < numBins; b++) {
      medianHistogram.push({
        binStart: Number((minMed + b * binWidth).toFixed(2)),
        binEnd: Number((minMed + (b + 1) * binWidth).toFixed(2)),
        binMid: Number((minMed + (b + 0.5) * binWidth).toFixed(2)),
        count: 0,
        density: 0
      });
    }
    for (let s = 0; s < numSims; s++) {
      const val = medianSamplesImp[s];
      const bIdx = Math.min(numBins - 1, Math.max(0, Math.floor((val - minMed) / binWidth)));
      medianHistogram[bIdx].count++;
    }
    for (const bin of medianHistogram) {
      bin.density = Number((bin.count / numSims).toFixed(4));
    }

    const teamSimResults = rostersData.map((r, i) => {
      const samples = teamSamplesImp[i];
      samples.sort();
      const meanSim = scoreSumImp[i] / numSims;
      const sdSim = Math.sqrt(Math.max(0, scoreSqSumImp[i] / numSims - meanSim * meanSim));

      const h2hProbImp = Math.min(0.999, Math.max(0.001, h2hWinsImp[i] / numSims));
      const h2hProbSlp = Math.min(0.999, Math.max(0.001, h2hWinsSlp[i] / numSims));
      const medProbImp = Math.min(0.999, Math.max(0.001, medianWinsImp[i] / numSims));
      const medProbSlp = Math.min(0.999, Math.max(0.001, medianWinsSlp[i] / numSims));

      const opp = oppIndex[i];
      const selfDone = compiledTeams[i].starterSimIndices.length === 0;
      const oppDone = opp >= 0 ? compiledTeams[opp].starterSimIndices.length === 0 : true;
      const finalH2hImp = (selfDone && oppDone && opp >= 0)
        ? (r.currentPts > rostersData[opp].currentPts ? 1.0 : (r.currentPts < rostersData[opp].currentPts ? 0.0 : 0.5))
        : h2hProbImp;
      const finalH2hSlp = (selfDone && oppDone && opp >= 0) ? finalH2hImp : h2hProbSlp;

      return {
        rosterId: r.rosterId,
        simMean: Number(meanSim.toFixed(2)),
        simSd: Number(sdSim.toFixed(2)),
        p10: Number(samples[Math.floor(numSims * 0.10)].toFixed(2)),
        p25: Number(samples[Math.floor(numSims * 0.25)].toFixed(2)),
        p50: Number(samples[Math.floor(numSims * 0.50)].toFixed(2)),
        p75: Number(samples[Math.floor(numSims * 0.75)].toFixed(2)),
        p90: Number(samples[Math.floor(numSims * 0.90)].toFixed(2)),
        h2hWinProbImproved: Number(finalH2hImp.toFixed(4)),
        h2hWinProbSleeper: Number(finalH2hSlp.toFixed(4)),
        medianWinProbImproved: Number(medProbImp.toFixed(4)),
        medianWinProbSleeper: Number(medProbSlp.toFixed(4)),
        prob2_0: Number((winBothImp[i] / numSims).toFixed(4)),
        prob1_1: Number((winOneImp[i] / numSims).toFixed(4)),
        prob0_2: Number((winZeroImp[i] / numSims).toFixed(4)),
        expectedWeeklyWins: Number((finalH2hImp + medProbImp).toFixed(3))
      };
    });

    return {
      numSims,
      leagueMedian: {
        expectedMedianImproved: Number(medianMeanImp.toFixed(2)),
        staticMedianImproved: Number(improvedStaticMedian.toFixed(2)),
        staticMedianSleeper: Number(sleeperStaticMedian.toFixed(2)),
        medianDiffVsSleeper: Number((medianMeanImp - sleeperStaticMedian).toFixed(2)),
        sd: Number(medianSdImp.toFixed(2)),
        p10: Number(medianSamplesImp[Math.floor(numSims * 0.10)].toFixed(2)),
        p25: Number(medianSamplesImp[Math.floor(numSims * 0.25)].toFixed(2)),
        p50: Number(medianSamplesImp[Math.floor(numSims * 0.50)].toFixed(2)),
        p75: Number(medianSamplesImp[Math.floor(numSims * 0.75)].toFixed(2)),
        p90: Number(medianSamplesImp[Math.floor(numSims * 0.90)].toFixed(2)),
        histogram: medianHistogram
      },
      teamSimResults
    };
  }

  /**
   * SIMULATION-BASED MATCHUP & MEDIAN LINEUP OPTIMIZER
   *
   * Given a target `rosterId`:
   * 1. Pre-simulates 10,000 Common-Random-Number (CRN) correlated outcomes across ALL candidate players
   *    on `rosterId` (both starters and bench) AND all starters on the other 9 teams in the league
   *    using the exact Block-Cholesky Pairwise Correlation Matrix!
   * 2. Generates legal candidate lineups (varying competitive QB/SFLEX, RB/WR/TE/FLEX, and IDP choices).
   * 3. Evaluates every candidate lineup on the exact same 10,000 simulated slates against:
   *      - The H2H Opponent's correlated simulated score
   *      - The 9 other teams' simulated scores for the Top-5 League Median cutoff
   * 4. Also evaluates any custom "Pick K of N" subset decision (e.g., "Which 2 of these 3 players is best
   *    when 2 play on the same team?")!
   */
  function runSimulationLineupOptimizer(stateData, targetRosterId = 3, options = {}) {
    const startMs = Date.now();
    const numSims = options.numSims || 8000;
    const objective = options.objective || 'expected_wins'; // 'expected_wins' | 'h2h' | 'median' | 'sweep_2_0' | 'points'

    const rostersData = stateData.rostersData || [];
    const targetTeamIdx = rostersData.findIndex((r) => r.rosterId === Number(targetRosterId));
    if (targetTeamIdx < 0) return null;

    const targetRoster = rostersData[targetTeamIdx];
    const oppRoster = rostersData.find((r) => r.matchupId === targetRoster.matchupId && r.rosterId !== targetRoster.rosterId) || null;
    const slots = stateData.league.starterSlots || [];

    // Gather all unique players on targetRoster (starters + bench) with positive projection or completed points
    const allRosterPlayersMap = new Map();
    for (const p of [...(targetRoster.rawStarters || targetRoster.starters), ...(targetRoster.rawBench || targetRoster.bench)]) {
      if (p && p.playerId !== '0' && (p.improvedProj > 0.2 || p.currentPts > 0)) {
        allRosterPlayersMap.set(p.playerId, p);
      }
    }
    const rosterPool = Array.from(allRosterPlayersMap.values());

    // Build global set of active players across ALL 10 teams + targetRoster's bench candidates for CRN Cholesky simulation
    const simPlayersById = new Map();
    const gameBuckets = new Map();

    const registerSimPlayer = (p) => {
      if (!p || p.playerId === '0') return;
      if (p.remMeanImproved > 0.001 || p.remSdImproved > 0.001) {
        if (!simPlayersById.has(p.playerId)) {
          simPlayersById.set(p.playerId, p);
          const gKey = p.gameId || `solo_${p.playerId}`;
          if (!gameBuckets.has(gKey)) gameBuckets.set(gKey, []);
          gameBuckets.get(gKey).push(p);
        }
      }
    };

    for (const p of rosterPool) registerSimPlayer(p);
    for (const r of rostersData) {
      for (const p of r.starters) registerSimPlayer(p);
    }

    const gameBlocks = [];
    const playerSimSlot = new Map();
    let totalActiveSimPlayers = 0;

    for (const [, bucket] of gameBuckets.entries()) {
      const m = bucket.length;
      const corr = Array.from({ length: m }, () => new Float64Array(m));
      for (let i = 0; i < m; i++) {
        corr[i][i] = 1.0;
        for (let j = i + 1; j < m; j++) {
          const c = computePairwisePlayerCorrelation(bucket[i], bucket[j]);
          corr[i][j] = c;
          corr[j][i] = c;
        }
      }
      const L = computeCholeskyFactor(corr);
      const globalIndices = new Int32Array(m);
      for (let i = 0; i < m; i++) {
        const gIdx = totalActiveSimPlayers++;
        globalIndices[i] = gIdx;
        playerSimSlot.set(bucket[i].playerId, gIdx);
      }
      gameBlocks.push({ m, L, players: bucket, globalIndices });
    }

    // Generate 10,000 CRN realizations for every player and for the other 9 teams
    const rng = new FastRNG(20261031);
    const playerTotalMatrix = Array.from({ length: totalActiveSimPlayers }, () => new Float32Array(numSims));
    const epsBuf = new Float64Array(40);

    for (let s = 0; s < numSims; s++) {
      for (let b = 0; b < gameBlocks.length; b++) {
        const blk = gameBlocks[b];
        const m = blk.m;
        const L = blk.L;
        for (let i = 0; i < m; i++) epsBuf[i] = rng.nextGaussian();
        for (let i = 0; i < m; i++) {
          let zCorr = 0;
          const rowL = L[i];
          for (let j = 0; j <= i; j++) zCorr += rowL[j] * epsBuf[j];
          const pl = blk.players[i];
          const gIdx = blk.globalIndices[i];
          const rem = sampleSkewedRemaining(zCorr, pl.remMeanImproved, pl.remSdImproved, pl.skewness);
          playerTotalMatrix[gIdx][s] = pl.currentPts + rem;
        }
      }
    }

    // Moment-match each player's CRN simulation vector so sample mean === exact pl.improvedProj
    // (eliminates finite-sample mean drift while preserving 100% of pairwise Cholesky correlations and skewness)
    for (let b = 0; b < gameBlocks.length; b++) {
      const blk = gameBlocks[b];
      for (let i = 0; i < blk.m; i++) {
        const pl = blk.players[i];
        const gIdx = blk.globalIndices[i];
        const vec = playerTotalMatrix[gIdx];
        let sumV = 0;
        for (let s = 0; s < numSims; s++) sumV += vec[s];
        const sampleMean = sumV / numSims;
        const targetMean = pl.improvedProj;
        const shift = targetMean - sampleMean;
        if (Math.abs(shift) > 1e-5) {
          for (let s = 0; s < numSims; s++) {
            vec[s] = Math.max(pl.currentPts, vec[s] + shift);
          }
        }
      }
    }

    // Compute the 9 other teams' total scores in each trial s, and determine:
    //  - Opponent's score in trial s: oppTrialScores[s]
    //  - The 5th-highest score among the OTHER 9 teams in trial s: fifthOtherScore[s]
    //    (Why? In a 10-team league, Target Team finishes in the Top 5 if and only if
    //     Target Team's score > the 5th-highest score among the other 9 teams!)
    const otherTeamsIndices = [];
    let oppOtherIdx = -1;
    for (let i = 0; i < rostersData.length; i++) {
      if (i === targetTeamIdx) continue;
      if (oppRoster && rostersData[i].rosterId === oppRoster.rosterId) {
        oppOtherIdx = otherTeamsIndices.length;
      }
      otherTeamsIndices.push(i);
    }

    const otherTeamsSimData = otherTeamsIndices.map((rIdx) => {
      const r = rostersData[rIdx];
      let lockedPts = 0;
      const activeSimIds = [];
      for (const p of r.starters) {
        const sIdx = playerSimSlot.get(p.playerId);
        if (sIdx !== undefined) {
          activeSimIds.push(sIdx);
        } else {
          lockedPts += p.currentPts;
        }
      }
      return { lockedPts, activeSimIds: new Int32Array(activeSimIds) };
    });

    const oppTrialScores = new Float32Array(numSims);
    const fifthOtherScore = new Float32Array(numSims);
    const nineBuf = new Float32Array(otherTeamsSimData.length);

    for (let s = 0; s < numSims; s++) {
      for (let k = 0; k < otherTeamsSimData.length; k++) {
        const ot = otherTeamsSimData[k];
        let tot = ot.lockedPts;
        const ids = ot.activeSimIds;
        for (let j = 0; j < ids.length; j++) {
          tot += playerTotalMatrix[ids[j]][s];
        }
        nineBuf[k] = tot;
        if (k === oppOtherIdx) oppTrialScores[s] = tot;
      }
      nineBuf.sort();
      // In ascending order of 9 other teams (indices 0..8):
      // Index 4 is the 5th-highest score among the other 9 teams!
      // If Target Team beats nineBuf[4], at most 4 other teams scored higher -> Target Team is Top 5!
      fifthOtherScore[s] = nineBuf[Math.max(0, nineBuf.length - 5)];
    }

    // Precompute opponent & median cutoff mean/variance for exact correlated Rao-Blackwell variance reduction
    let oppSum = 0, oppSqSum = 0, medSum = 0, medSqSum = 0;
    for (let s = 0; s < numSims; s++) {
      const o = oppTrialScores[s];
      const m = fifthOtherScore[s];
      oppSum += o;
      oppSqSum += o * o;
      medSum += m;
      medSqSum += m * m;
    }
    const oppMeanExact = oppSum / numSims;
    const oppVarExact = Math.max(9.0, oppSqSum / numSims - oppMeanExact * oppMeanExact);
    const medMeanExact = medSum / numSims;
    const medVarExact = Math.max(9.0, medSqSum / numSims - medMeanExact * medMeanExact);

    function normalCdf(z) {
      const t = 1 / (1 + 0.2316419 * Math.abs(z));
      const d = 0.3989422804014327 * Math.exp(-0.5 * z * z);
      const p = d * t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + 1.330274429 * t))));
      return z >= 0 ? 1 - p : p;
    }

    // Helper to evaluate ANY 14-player lineup array in ~0.2ms using precomputed CRN matrices + exact correlated variance!
    function evaluateCandidateLineup(lineupPlayers) {
      let lockedSum = 0;
      let expectedPts = 0;
      let sleeperPts = 0;
      let exactVar = 0;
      let oppCov = 0;
      const simIds = [];
      const activePlayers = [];

      for (const p of lineupPlayers) {
        if (!p || p.playerId === '0') continue;
        expectedPts += p.improvedProj;
        sleeperPts += p.sleeperProj;
        const sd = p.remSdImproved || 0;
        exactVar += sd * sd;
        activePlayers.push(p);
        const sIdx = playerSimSlot.get(p.playerId);
        if (sIdx !== undefined) {
          simIds.push(sIdx);
        } else {
          lockedSum += p.currentPts;
        }
      }

      // Add exact pairwise Cholesky correlations within the lineup and counter-correlations with opponent starters
      for (let i = 0; i < activePlayers.length; i++) {
        const pA = activePlayers[i];
        const sdA = pA.remSdImproved || 0;
        if (sdA <= 0) continue;
        for (let j = i + 1; j < activePlayers.length; j++) {
          const pB = activePlayers[j];
          const sdB = pB.remSdImproved || 0;
          if (sdB <= 0) continue;
          const rho = computePairwisePlayerCorrelation(pA, pB);
          if (rho !== 0) exactVar += 2 * rho * sdA * sdB;
        }
        if (oppRoster) {
          for (const oppP of oppRoster.starters) {
            if (!oppP || oppP.playerId === '0') continue;
            const sdOpp = oppP.remSdImproved || 0;
            if (sdOpp <= 0) continue;
            const rhoOpp = computePairwisePlayerCorrelation(pA, oppP);
            if (rhoOpp !== 0) oppCov += rhoOpp * sdA * sdOpp;
          }
        }
      }
      exactVar = Math.max(4.0, exactVar);

      const activeIds = new Int32Array(simIds);
      let h2hWins = 0;
      let medWins = 0;
      let bothWins = 0;
      let zeroWins = 0;
      let sumScore = 0;
      let sumSqScore = 0;
      let sumDiffSq = 0;
      const samples = new Float32Array(numSims);

      for (let s = 0; s < numSims; s++) {
        let score = lockedSum;
        for (let j = 0; j < activeIds.length; j++) {
          score += playerTotalMatrix[activeIds[j]][s];
        }
        samples[s] = score;
        sumScore += score;
        sumSqScore += score * score;

        const oppS = oppTrialScores[s];
        const medS = fifthOtherScore[s];
        const diff = score - oppS;
        sumDiffSq += diff * diff;

        const pH2H = 1 / (1 + Math.exp(-diff / 2.2));
        const pMed = 1 / (1 + Math.exp(-(score - medS) / 2.2));
        const hardH2H = score > oppS ? 1 : (score === oppS ? 0.5 : 0);
        const hardMed = score > medS ? 1 : (score === medS ? 0.5 : 0);

        const wonH2H = 0.65 * pH2H + 0.35 * hardH2H;
        const wonMed = 0.65 * pMed + 0.35 * hardMed;
        h2hWins += wonH2H;
        medWins += wonMed;
        bothWins += wonH2H * wonMed;
        zeroWins += (1 - wonH2H) * (1 - wonMed);
      }

      samples.sort();
      const meanSim = sumScore / numSims;
      const sdSim = Math.sqrt(Math.max(0, sumSqScore / numSims - meanSim * meanSim));

      // Exact analytical correlated win probabilities (zero finite-sample noise)
      const h2hDiffSd = Math.sqrt(Math.max(9.0, exactVar + oppVarExact - 2 * oppCov));
      const medDiffSd = Math.sqrt(Math.max(9.0, exactVar + medVarExact));
      const exactH2hProb = normalCdf((expectedPts - oppMeanExact) / h2hDiffSd);
      const exactMedProb = normalCdf((expectedPts - medMeanExact) / medDiffSd);

      const mcH2hProb = h2hWins / numSims;
      const mcMedProb = medWins / numSims;

      const h2hProb = 0.55 * exactH2hProb + 0.45 * mcH2hProb;
      const medProb = 0.55 * exactMedProb + 0.45 * mcMedProb;
      const prob2_0 = bothWins / numSims;
      const prob0_2 = zeroWins / numSims;
      const prob1_1 = Math.max(0, 1 - prob2_0 - prob0_2);
      const expWins = h2hProb + medProb;

      return {
        expectedPts: Number(expectedPts.toFixed(2)),
        sleeperPts: Number(sleeperPts.toFixed(2)),
        simMean: Number(meanSim.toFixed(2)),
        simSd: Number(sdSim.toFixed(2)),
        p10: Number(samples[Math.floor(numSims * 0.10)].toFixed(2)),
        p50: Number(samples[Math.floor(numSims * 0.50)].toFixed(2)),
        p90: Number(samples[Math.floor(numSims * 0.90)].toFixed(2)),
        h2hWinProb: Number(h2hProb.toFixed(4)),
        medianWinProb: Number(medProb.toFixed(4)),
        prob2_0: Number(prob2_0.toFixed(4)),
        prob1_1: Number(prob1_1.toFixed(4)),
        prob0_2: Number(prob0_2.toFixed(4)),
        expectedWeeklyWins: Number(expWins.toFixed(4))
      };
    }

    // Generate candidate offensive + IDP lineups systematically:
    // Lock any player whose game is already 'complete' or 'in_game' in their current starter slot!
    const rawStarters = targetRoster.rawStarters || targetRoster.starters;
    const lockedSlots = new Array(slots.length).fill(null);
    const lockedPlayerIds = new Set();

    for (let i = 0; i < slots.length; i++) {
      const st = rawStarters[i];
      if (st && st.playerId !== '0' && (st.gameStatus === 'complete' || st.gameStatus === 'in_game')) {
        lockedSlots[i] = st;
        lockedPlayerIds.add(st.playerId);
      }
    }

    // Candidate pool by position for unlocked slots (excluding BYE and Out/IR)
    const activePool = rosterPool.filter((p) => {
      if (lockedPlayerIds.has(p.playerId)) return false;
      const isOut = ['Out', 'IR', 'Doubtful', 'Suspension', 'COV'].includes(p.injuryStatus);
      return p.gameStatus !== 'bye' && !isOut && p.improvedProj > 0.5;
    }).sort((a, b) => b.improvedProj - a.improvedProj);

    const qbs = activePool.filter((p) => p.canonicalPos === 'QB').slice(0, 4);
    const rbs = activePool.filter((p) => p.canonicalPos === 'RB').slice(0, 6);
    const wrs = activePool.filter((p) => p.canonicalPos === 'WR').slice(0, 7);
    const tes = activePool.filter((p) => p.canonicalPos === 'TE').slice(0, 3);
    const idps = activePool.filter((p) => ['DL', 'LB', 'DB'].includes(p.canonicalPos)).slice(0, 6);

    // Backtracking generator for top distinct legal offensive + IDP combinations
    const candidateLineups = [];
    const seenLineupSets = new Set();

    // Sort candidate pool per slot
    // Note: Because 6 RB/WR/TE players are already placed in RB1, RB2, WR1, WR2, WR3, TE before FLEX1 & FLEX2 are reached,
    // FLEX pools must include at least the top 11 eligible skill players so ranks #7-#11 are available for FLEX1/FLEX2!
    const slotPools = slots.map((slot, idx) => {
      if (lockedSlots[idx]) return [lockedSlots[idx]];
      const eligible = activePool.filter((p) => isEligibleForSlot(p, slot));
      const limit = (slot === 'FLEX' || slot === 'SUPER_FLEX') ? 11 : (slot === 'WR' || slot === 'RB' ? 6 : 3);
      return eligible.slice(0, limit);
    });

    // Greedy baseline max-projection lineup first
    function buildGreedyLineup(scoreKey = 'improvedProj') {
      const used = new Set(lockedPlayerIds);
      const lineup = [...lockedSlots];
      // Fill tighter single-position slots first, then FLEX / SUPER_FLEX / IDP_FLEX
      const slotOrder = slots
        .map((s, i) => ({ s, i, priority: s.includes('FLEX') ? 2 : 1 }))
        .sort((a, b) => a.priority - b.priority);

      for (const { s, i } of slotOrder) {
        if (lineup[i]) continue;
        const candidates = activePool
          .filter((p) => !used.has(p.playerId) && isEligibleForSlot(p, s))
          .sort((a, b) => b[scoreKey] - a[scoreKey]);
        if (candidates.length > 0) {
          lineup[i] = { ...candidates[0], slot: s };
          used.add(candidates[0].playerId);
        } else {
          lineup[i] = rawStarters[i] || { playerId: '0', fullName: 'Empty Slot', slot: s, currentPts: 0, improvedProj: 0, sleeperProj: 0 };
        }
      }
      return lineup;
    }

    const maxPtsLineup = buildGreedyLineup('improvedProj');
    const sleeperMaxLineup = buildGreedyLineup('sleeperProj');
    const maxPtsBenchmark = maxPtsLineup.reduce((acc, p) => acc + (p ? p.improvedProj : 0), 0);

    // Always seed candidateLineups with maxPtsLineup and sleeperMaxLineup first
    for (const seedLineup of [maxPtsLineup, sleeperMaxLineup]) {
      const idsKey = seedLineup.map((p) => p.playerId).slice().sort().join(',');
      if (!seenLineupSets.has(idsKey)) {
        seenLineupSets.add(idsKey);
        candidateLineups.push(seedLineup.map((p, i) => ({ ...p, slot: slots[i] })));
      }
    }

    // Search combinations near the top projection frontier (within 8.5 points of max mean projection)
    const currentAssignment = new Array(slots.length);
    const usedInSearch = new Set(lockedPlayerIds);

    // Precompute max possible remaining points from slot idx..end for branch-and-bound pruning
    const maxSlotProj = slotPools.map((pool) => (pool.length > 0 ? pool[0].improvedProj : 0));
    const suffixMax = new Float64Array(slots.length + 1);
    for (let i = slots.length - 1; i >= 0; i--) {
      suffixMax[i] = suffixMax[i + 1] + maxSlotProj[i];
    }

    function searchLineups(slotIdx, runningProj) {
      if (candidateLineups.length >= 240) return;
      if (runningProj + suffixMax[slotIdx] < maxPtsBenchmark - 8.5) return;

      if (slotIdx === slots.length) {
        const idsKey = currentAssignment
          .map((p) => p.playerId)
          .slice()
          .sort()
          .join(',');
        if (!seenLineupSets.has(idsKey)) {
          seenLineupSets.add(idsKey);
          candidateLineups.push(currentAssignment.map((p, i) => ({ ...p, slot: slots[i] })));
        }
        return;
      }

      if (lockedSlots[slotIdx]) {
        currentAssignment[slotIdx] = lockedSlots[slotIdx];
        searchLineups(slotIdx + 1, runningProj + lockedSlots[slotIdx].improvedProj);
        return;
      }

      const pool = slotPools[slotIdx];
      if (!pool || pool.length === 0) {
        const fallback = { playerId: '0', fullName: 'Empty Slot', slot: slots[slotIdx], currentPts: 0, improvedProj: 0, sleeperProj: 0 };
        currentAssignment[slotIdx] = fallback;
        searchLineups(slotIdx + 1, runningProj);
        return;
      }

      let triedAny = false;
      for (let k = 0; k < pool.length; k++) {
        const cand = pool[k];
        if (usedInSearch.has(cand.playerId)) continue;
        triedAny = true;
        usedInSearch.add(cand.playerId);
        currentAssignment[slotIdx] = cand;
        searchLineups(slotIdx + 1, runningProj + cand.improvedProj);
        usedInSearch.delete(cand.playerId);
      }
      if (!triedAny) {
        const fallback = { playerId: '0', fullName: 'Empty Slot', slot: slots[slotIdx], currentPts: 0, improvedProj: 0, sleeperProj: 0 };
        currentAssignment[slotIdx] = fallback;
        searchLineups(slotIdx + 1, runningProj);
      }
    }

    searchLineups(0, 0);

    // Ensure Current Set Lineup and Sleeper Max-Proj Lineup are also evaluated for comparison
    const currentSetMetrics = evaluateCandidateLineup(rawStarters);
    const sleeperMaxMetrics = evaluateCandidateLineup(sleeperMaxLineup);

    // Evaluate all candidate lineups on the 10,000 CRN simulations
    const evaluatedCandidates = candidateLineups.map((lineup) => {
      const metrics = evaluateCandidateLineup(lineup);
      // Analyze internal correlations & opponent correlations in this lineup
      const stackPairs = [];
      const cannibalPairs = [];
      const oppCorrelations = [];

      for (let i = 0; i < lineup.length; i++) {
        const pA = lineup[i];
        if (!pA || pA.playerId === '0') continue;
        for (let j = i + 1; j < lineup.length; j++) {
          const pB = lineup[j];
          if (!pB || pB.playerId === '0') continue;
          const c = computePairwisePlayerCorrelation(pA, pB);
          if (c >= 0.12) {
            stackPairs.push({ pA: pA.shortName, pB: pB.shortName, teamA: pA.team, teamB: pB.team, corr: c });
          } else if (c <= -0.10) {
            cannibalPairs.push({ pA: pA.shortName, pB: pB.shortName, team: pA.team, corr: c });
          }
        }
        if (oppRoster) {
          for (const oppP of oppRoster.starters) {
            if (!oppP || oppP.playerId === '0') continue;
            const cOpp = computePairwisePlayerCorrelation(pA, oppP);
            if (Math.abs(cOpp) >= 0.10) {
              oppCorrelations.push({
                ownPlayer: pA.shortName,
                oppPlayer: oppP.shortName,
                team: pA.team,
                corr: cOpp
              });
            }
          }
        }
      }

      return {
        lineup,
        metrics,
        stackPairs,
        cannibalPairs,
        oppCorrelations
      };
    });

    // Sort candidate lineups by the selected optimization objective
    const sortFn = (a, b) => {
      if (objective === 'h2h') return b.metrics.h2hWinProb - a.metrics.h2hWinProb || b.metrics.expectedPts - a.metrics.expectedPts;
      if (objective === 'median') return b.metrics.medianWinProb - a.metrics.medianWinProb || b.metrics.expectedPts - a.metrics.expectedPts;
      if (objective === 'sweep_2_0') return b.metrics.prob2_0 - a.metrics.prob2_0 || b.metrics.expectedWeeklyWins - a.metrics.expectedWeeklyWins;
      if (objective === 'points') return b.metrics.expectedPts - a.metrics.expectedPts;
      return b.metrics.expectedWeeklyWins - a.metrics.expectedWeeklyWins || b.metrics.expectedPts - a.metrics.expectedPts;
    };

    evaluatedCandidates.sort(sortFn);
    const rawOptimalCandidate = evaluatedCandidates[0] || { lineup: maxPtsLineup, metrics: evaluateCandidateLineup(maxPtsLineup), stackPairs: [], cannibalPairs: [], oppCorrelations: [] };

    // Align optimal lineup slot assignments with current starters so continuing starters stay in their original slots whenever valid
    function alignLineupSlotsWithCurrent(optLineup, currStarters) {
      const realPlayers = optLineup.filter((p) => p && p.playerId && p.playerId !== '0');
      const bestSlotAssign = new Array(slots.length).fill(null);
      let bestMatchScore = -1;

      // Sort players so more constrained positions are placed first
      const orderedPlayers = [...realPlayers].sort((a, b) => {
        const eligA = slots.filter((s) => isEligibleForSlot(a, s)).length;
        const eligB = slots.filter((s) => isEligibleForSlot(b, s)).length;
        return eligA - eligB;
      });

      const currentSlotAssignment = new Array(slots.length).fill(null);

      function backtrackAssign(pIdx, matchScore) {
        if (matchScore + (orderedPlayers.length - pIdx) * 10 <= bestMatchScore) return;
        if (pIdx === orderedPlayers.length) {
          if (matchScore > bestMatchScore) {
            bestMatchScore = matchScore;
            for (let s = 0; s < slots.length; s++) {
              bestSlotAssign[s] = currentSlotAssignment[s];
            }
          }
          return;
        }

        const p = orderedPlayers[pIdx];
        // Prefer slots where p is currently starting in currStarters
        const candidateSlots = [];
        for (let s = 0; s < slots.length; s++) {
          if (currentSlotAssignment[s] !== null) continue;
          if (!isEligibleForSlot(p, slots[s])) continue;
          const isExactSlot = currStarters[s] && currStarters[s].playerId === p.playerId;
          const isSameSlotType = p.slot === slots[s];
          const gain = isExactSlot ? 10 : (isSameSlotType ? 2 : 0);
          candidateSlots.push({ s, gain });
        }
        candidateSlots.sort((a, b) => b.gain - a.gain);

        for (const { s, gain } of candidateSlots) {
          currentSlotAssignment[s] = { ...p, slot: slots[s] };
          backtrackAssign(pIdx + 1, matchScore + gain);
          currentSlotAssignment[s] = null;
        }
      }

      backtrackAssign(0, 0);

      if (bestMatchScore >= 0) {
        return slots.map((slotName, idx) =>
          bestSlotAssign[idx] || { playerId: '0', fullName: 'Empty Slot', shortName: 'Empty', team: 'FA', position: slotName, slot: slotName, currentPts: 0, improvedProj: 0, sleeperProj: 0 }
        );
      }
      return optLineup;
    }

    const alignedOptimalLineup = alignLineupSlotsWithCurrent(rawOptimalCandidate.lineup, rawStarters);
    const optimalCandidate = {
      ...rawOptimalCandidate,
      lineup: alignedOptimalLineup
    };

    // Identify true swaps between Optimal Lineup and Current Set Lineup
    const optimalStarterIds = new Set(optimalCandidate.lineup.map((p) => p.playerId).filter((id) => id !== '0'));
    const currentStarterIds = new Set(rawStarters.map((p) => p.playerId).filter((id) => id !== '0'));

    const recommendedSwaps = [];
    for (let i = 0; i < slots.length; i++) {
      const currP = rawStarters[i];
      const optP = optimalCandidate.lineup[i];
      if (!optP) continue;
      const currId = currP ? currP.playerId : '0';
      const optId = optP ? optP.playerId : '0';
      if (currId !== optId) {
        // Build human-readable quantitative reasons for this swap
        const reasons = [];
        const projDelta = Number(((optP.improvedProj || 0) - (currP ? currP.improvedProj || 0 : 0)).toFixed(2));
        if (currId === '0' || (currP && currP.gameStatus === 'bye') || (currP && ['Out', 'IR', 'Doubtful'].includes(currP.injuryStatus))) {
          reasons.push(`Fills open/inactive ${slots[i]} slot (+${projDelta.toFixed(2)} pts)`);
        } else if (projDelta > 0) {
          reasons.push(`+${projDelta.toFixed(2)} blended market projection edge`);
        }
        if (optP.sportsbookProj && optP.sportsbookProj > optP.sleeperProj + 0.4) {
          reasons.push(`Sportsbook props (${optP.sportsbookProj.toFixed(1)}) bullish vs Sleeper (${optP.sleeperProj.toFixed(1)})`);
        }
        // Check if optP stacks with another optimal starter
        for (const otherP of optimalCandidate.lineup) {
          if (!otherP || otherP.playerId === '0' || otherP.playerId === optP.playerId) continue;
          const c = computePairwisePlayerCorrelation(optP, otherP);
          if (c >= 0.14) {
            reasons.push(`Positive correlation stack with ${otherP.shortName} (ρ = +${c.toFixed(2)})`);
          }
        }
        // Check if currP had negative cannibalization with another optimal starter
        if (currP && currId !== '0') {
          for (const otherP of optimalCandidate.lineup) {
            if (!otherP || otherP.playerId === '0' || otherP.playerId === optP.playerId) continue;
            const cOld = computePairwisePlayerCorrelation(currP, otherP);
            if (cOld <= -0.10) {
              reasons.push(`Avoids ${currP.team} same-team cannibalization with ${otherP.shortName} (ρ = ${cOld.toFixed(2)})`);
            }
          }
        }

        recommendedSwaps.push({
          slotIndex: i,
          slot: slots[i],
          outPlayer: currP,
          inPlayer: optP,
          isTrueAddition: !currentStarterIds.has(optId),
          isTrueRemoval: currId === '0' || !optimalStarterIds.has(currId),
          projDelta,
          reasons,
          summaryReason: reasons.join(' • ') || 'Optimizes joint simulation win probability'
        });
      }
    }

    // Helper to check if a set of active players can legally fit into the league's 14 starter slots
    function checkLineupSlotValidity(activePlayersList) {
      const realList = activePlayersList.filter((p) => p && p.playerId && p.playerId !== '0');
      const usedSlot = new Array(slots.length).fill(false);
      // Sort most constrained players first
      const sorted = [...realList].sort((a, b) => {
        const cA = slots.filter((s) => isEligibleForSlot(a, s)).length;
        const cB = slots.filter((s) => isEligibleForSlot(b, s)).length;
        return cA - cB;
      });
      function matchPlayer(idx) {
        if (idx === sorted.length) return true;
        const p = sorted[idx];
        for (let s = 0; s < slots.length; s++) {
          if (!usedSlot[s] && isEligibleForSlot(p, slots[s])) {
            usedSlot[s] = true;
            if (matchPlayer(idx + 1)) return true;
            usedSlot[s] = false;
          }
        }
        return false;
      }
      const isValid = matchPlayer(0);
      return {
        isValid,
        note: isValid ? 'Valid 14-Slot Roster Fit' : 'Exceeds Positional Slot Cap (e.g. >4 RBs)'
      };
    }

    // Build smart presets for the "Best K of N" Same-Team / Close-Projection Conflict Analyzer
    const skillPool = activePool.filter((p) => ['RB', 'WR', 'TE'].includes(p.canonicalPos));
    const qbPool = activePool.filter((p) => p.canonicalPos === 'QB');
    const tePool = activePool.filter((p) => p.canonicalPos === 'TE');

    // Detect same-team pairs among borderline skill players (ranked #3..#11)
    const borderlineSkill = skillPool.slice(2, Math.min(10, skillPool.length));
    let sameTeamTriadIds = null;
    for (let i = 0; i < borderlineSkill.length; i++) {
      for (let j = i + 1; j < borderlineSkill.length; j++) {
        if (borderlineSkill[i].team === borderlineSkill[j].team) {
          const p1 = borderlineSkill[i];
          const p2 = borderlineSkill[j];
          const avgProj = 0.5 * (p1.improvedProj + p2.improvedProj);
          // Prefer a 3rd player from a different team who is the top bench contender (or borderline starter)
          // and whose position ensures all 2-of-3 combinations are slot-valid
          const baseWithoutPair = optimalCandidate.lineup.filter(
            (p) => p && p.playerId !== '0' && p.playerId !== p1.playerId && p.playerId !== p2.playerId
          );
          const otherCands = borderlineSkill
            .filter((p) => p.playerId !== p1.playerId && p.playerId !== p2.playerId && p.team !== p1.team)
            .sort((a, b) => {
              // Prefer bench contenders so 2 of the 3 are starters and 1 is the top bench challenger
              const aInOpt = optimalStarterIds.has(a.playerId) ? 1 : 0;
              const bInOpt = optimalStarterIds.has(b.playerId) ? 1 : 0;
              const aValidAll = (
                checkLineupSlotValidity([...baseWithoutPair, p1, a]).isValid &&
                checkLineupSlotValidity([...baseWithoutPair, p2, a]).isValid
              ) ? 0 : 1;
              const bValidAll = (
                checkLineupSlotValidity([...baseWithoutPair, p1, b]).isValid &&
                checkLineupSlotValidity([...baseWithoutPair, p2, b]).isValid
              ) ? 0 : 1;
              if (aValidAll !== bValidAll) return aValidAll - bValidAll;
              if (aInOpt !== bInOpt) return aInOpt - bInOpt;
              return Math.abs(a.improvedProj - avgProj) - Math.abs(b.improvedProj - avgProj);
            });
          const thirdCand = otherCands[0];
          if (thirdCand) {
            sameTeamTriadIds = [p1.playerId, p2.playerId, thirdCand.playerId];
            break;
          }
        }
      }
      if (sameTeamTriadIds) break;
    }

    const rbWrPool = activePool.filter((p) => ['RB', 'WR'].includes(p.canonicalPos));
    const flexCluster5Ids = rbWrPool.slice(4, 9).map((p) => p.playerId);
    // For 2-of-4 (WR3 + FLEX2), take lowest RB starter + lowest WR starter + top RB bench + top WR bench
    const rbStarters = rbWrPool.filter((p) => p.canonicalPos === 'RB' && optimalStarterIds.has(p.playerId));
    const wrStarters = rbWrPool.filter((p) => p.canonicalPos === 'WR' && optimalStarterIds.has(p.playerId));
    const rbBench = rbWrPool.filter((p) => p.canonicalPos === 'RB' && !optimalStarterIds.has(p.playerId));
    const wrBench = rbWrPool.filter((p) => p.canonicalPos === 'WR' && !optimalStarterIds.has(p.playerId));
    const flexCluster4Ids = [
      rbStarters[rbStarters.length - 1]?.playerId,
      wrStarters[wrStarters.length - 1]?.playerId,
      rbBench[0]?.playerId,
      wrBench[0]?.playerId
    ].filter(Boolean);

    const analyzerPresets = [];
    if (sameTeamTriadIds && sameTeamTriadIds.length === 3) {
      const triadNames = sameTeamTriadIds.map((id) => allRosterPlayersMap.get(String(id))?.shortName).filter(Boolean);
      analyzerPresets.push({
        id: 'same_team_2_of_3',
        label: `Best 2 of 3 (Same-Team Conflict: ${triadNames.join(' / ')})`,
        playerIds: sameTeamTriadIds,
        chooseK: 2
      });
    }
    if (flexCluster5Ids.length >= 4) {
      analyzerPresets.push({
        id: 'flex_cluster_3_of_5',
        label: `Best 3 of ${flexCluster5Ids.length} (Borderline WR / FLEX Cluster)`,
        playerIds: flexCluster5Ids,
        chooseK: 3
      });
    }
    if (flexCluster4Ids.length >= 4) {
      analyzerPresets.push({
        id: 'flex_cluster_2_of_4',
        label: `Best 2 of 4 (WR3 + FLEX2 Decision)`,
        playerIds: flexCluster4Ids,
        chooseK: 2
      });
    }
    if (qbPool.length >= 3) {
      analyzerPresets.push({
        id: 'qb_superflex_2_of_n',
        label: `Best 2 of ${qbPool.length} (QB + Superflex Decision)`,
        playerIds: qbPool.map((p) => p.playerId),
        chooseK: 2
      });
    }
    if (tePool.length >= 2) {
      analyzerPresets.push({
        id: 'te_1_of_2',
        label: `Best 1 of ${tePool.length} (Tight End Decision)`,
        playerIds: tePool.map((p) => p.playerId),
        chooseK: 1
      });
    }

    // "Best K of N" Subset / Same-Team Conflict Analyzer:
    let subsetPoolIds = options.subsetPlayerIds || null;
    let subsetChooseK = options.subsetChooseK || null;

    if (!subsetPoolIds || subsetPoolIds.length < 2) {
      // Default to Preset 0 (Best 2 of 3 Same-Team Conflict if present, otherwise Borderline WR/FLEX cluster)
      if (analyzerPresets.length > 0) {
        subsetPoolIds = analyzerPresets[0].playerIds;
        subsetChooseK = analyzerPresets[0].chooseK;
      } else if (flexCluster5Ids.length >= 4) {
        subsetPoolIds = flexCluster5Ids;
        subsetChooseK = 3;
      } else {
        const fallbackSkill = skillPool.slice(2, 7);
        subsetPoolIds = fallbackSkill.map((p) => p.playerId);
        subsetChooseK = Math.min(2, Math.max(1, fallbackSkill.length - 1));
      }
    } else if (!subsetChooseK) {
      subsetChooseK = Math.min(2, Math.max(1, subsetPoolIds.length - 1));
    }

    const subsetPlayers = subsetPoolIds
      .map((id) => allRosterPlayersMap.get(String(id)))
      .filter(Boolean);

    subsetChooseK = Math.min(subsetPlayers.length, Math.max(1, Number(subsetChooseK) || 2));

    // Construct an exact fixed core of (activeStarterCount - K) players outside `subsetPlayers`
    // so that every C(N, K) combination is evaluated on the exact same 14-slot lineup baseline!
    const subsetIdSet = new Set(subsetPlayers.map((p) => p.playerId));
    const activeOptimalStarters = optimalCandidate.lineup.filter((p) => p && p.playerId && p.playerId !== '0');
    const targetCoreSize = Math.max(0, activeOptimalStarters.length - subsetChooseK);

    let fixedCoreLineup = activeOptimalStarters.filter((p) => !subsetIdSet.has(p.playerId));

    if (fixedCoreLineup.length > targetCoreSize) {
      // Need to vacate (fixedCoreLineup.length - targetCoreSize) slots from fixedCoreLineup:
      // Prefer vacating the lowest-projected unlocked starters that share positions with subsetPlayers
      const subsetPosSet = new Set(subsetPlayers.map((p) => p.canonicalPos));
      fixedCoreLineup.sort((a, b) => {
        const aLocked = lockedPlayerIds.has(a.playerId) ? 1 : 0;
        const bLocked = lockedPlayerIds.has(b.playerId) ? 1 : 0;
        if (aLocked !== bLocked) return bLocked - aLocked;
        const aSamePos = subsetPosSet.has(a.canonicalPos) ? 0 : 1;
        const bSamePos = subsetPosSet.has(b.canonicalPos) ? 0 : 1;
        if (aSamePos !== bSamePos) return bSamePos - aSamePos;
        return b.improvedProj - a.improvedProj;
      });
      fixedCoreLineup = fixedCoreLineup.slice(0, targetCoreSize);
    } else if (fixedCoreLineup.length < targetCoreSize) {
      // Need to backfill (targetCoreSize - fixedCoreLineup.length) players from activePool outside subsetIdSet
      const coreIds = new Set(fixedCoreLineup.map((p) => p.playerId));
      const subsetPosSet = new Set(subsetPlayers.map((p) => p.canonicalPos));
      const backfillCandidates = activePool
        .filter((p) => !subsetIdSet.has(p.playerId) && !coreIds.has(p.playerId))
        .sort((a, b) => {
          const aSamePos = subsetPosSet.has(a.canonicalPos) ? 1 : 0;
          const bSamePos = subsetPosSet.has(b.canonicalPos) ? 1 : 0;
          if (aSamePos !== bSamePos) return bSamePos - aSamePos;
          return b.improvedProj - a.improvedProj;
        });
      for (const cand of backfillCandidates) {
        if (fixedCoreLineup.length >= targetCoreSize) break;
        fixedCoreLineup.push(cand);
        coreIds.add(cand.playerId);
      }
    }

    const subsetComparison = [];

    function generateSubsets(arr, k, start = 0, curr = []) {
      if (curr.length === k) {
        const combined = [...fixedCoreLineup, ...curr];
        const m = evaluateCandidateLineup(combined);
        // Compute internal pairwise correlations within `curr` AND with `fixedCoreLineup`
        const pairCorrs = [];
        let sameTeamCount = 0;
        for (let i = 0; i < curr.length; i++) {
          for (let j = i + 1; j < curr.length; j++) {
            const c = computePairwisePlayerCorrelation(curr[i], curr[j]);
            if (curr[i].team === curr[j].team) sameTeamCount++;
            pairCorrs.push({
              pair: `${curr[i].shortName} (${curr[i].team}) + ${curr[j].shortName} (${curr[j].team})`,
              sameTeam: curr[i].team === curr[j].team,
              sameGame: Boolean(curr[i].gameId && curr[i].gameId === curr[j].gameId),
              corr: c
            });
          }
        }
        // Also check correlations between `curr` and locked core starters (e.g. QB-WR stack or RB-WR cannibalization)
        const coreCorrs = [];
        for (const p of curr) {
          for (const coreP of fixedCoreLineup) {
            const c = computePairwisePlayerCorrelation(p, coreP);
            if (Math.abs(c) >= 0.10) {
              if (p.team === coreP.team && c < 0) sameTeamCount++;
              coreCorrs.push({
                pair: `${p.shortName} (${p.team}) ↔ ${coreP.shortName} (${coreP.team})`,
                sameTeam: p.team === coreP.team,
                corr: c
              });
            }
          }
        }

        const isOptimalCombo = curr.every((p) => optimalStarterIds.has(p.playerId));
        const slotFit = checkLineupSlotValidity(combined);

        subsetComparison.push({
          players: curr.map((p) => ({
            playerId: p.playerId,
            shortName: p.shortName,
            fullName: p.fullName,
            team: p.team,
            position: p.position,
            improvedProj: p.improvedProj,
            sleeperProj: p.sleeperProj,
            sportsbookProj: p.sportsbookProj
          })),
          label: curr.map((p) => `${p.shortName} (${p.team} ${p.position})`).join(' + '),
          subsetExpectedPts: Number(curr.reduce((a, b) => a + b.improvedProj, 0).toFixed(2)),
          subsetSleeperPts: Number(curr.reduce((a, b) => a + b.sleeperProj, 0).toFixed(2)),
          sameTeamCount,
          pairCorrs,
          coreCorrs,
          isOptimalCombo,
          slotFit,
          metrics: m
        });
        return;
      }
      for (let i = start; i < arr.length; i++) {
        curr.push(arr[i]);
        generateSubsets(arr, k, i + 1, curr);
        curr.pop();
      }
    }

    if (subsetPlayers.length >= subsetChooseK && subsetChooseK >= 1) {
      generateSubsets(subsetPlayers, subsetChooseK);
      subsetComparison.sort((a, b) => {
        const validA = a.slotFit?.isValid ? 1 : 0;
        const validB = b.slotFit?.isValid ? 1 : 0;
        if (validA !== validB) return validB - validA;
        return sortFn(a, b);
      });
    }

    // Build full correlation matrix for all active skill/QB candidates on the target roster for visual display
    const displayPool = activePool.slice(0, 12);
    const rosterCorrMatrix = {
      players: displayPool.map((p) => ({
        playerId: p.playerId,
        shortName: p.shortName,
        fullName: p.fullName,
        team: p.team,
        position: p.position,
        improvedProj: p.improvedProj,
        sleeperProj: p.sleeperProj,
        sportsbookProj: p.sportsbookProj
      })),
      matrix: displayPool.map((pA) =>
        displayPool.map((pB) => computePairwisePlayerCorrelation(pA, pB))
      )
    };

    return {
      computeTimeMs: Date.now() - startMs,
      numSims,
      objective,
      targetRosterId: targetRoster.rosterId,
      targetTeamName: targetRoster.teamName,
      oppTeamName: oppRoster ? oppRoster.teamName : 'Opponent',
      oppProjectedPts: oppRoster ? oppRoster.improvedTotalProj : 0,
      currentStarters: rawStarters,
      currentSetMetrics,
      sleeperMaxLineup,
      sleeperMaxMetrics,
      optimalCandidate,
      topCandidates: evaluatedCandidates.slice(0, 8),
      recommendedSwaps,
      subsetAnalyzer: {
        presets: analyzerPresets,
        availableCandidates: activePool,
        selectedPlayerIds: subsetPlayers.map((p) => p.playerId),
        chooseK: subsetChooseK,
        fixedCorePlayers: fixedCoreLineup.map((p) => ({
          playerId: p.playerId,
          shortName: p.shortName,
          team: p.team,
          position: p.position,
          improvedProj: p.improvedProj
        })),
        combinations: subsetComparison
      },
      rosterCorrMatrix,
      allRosterPlayers: rosterPool.sort((a, b) => b.improvedProj - a.improvedProj)
    };
  }

  function recomputeLeagueFromRosters(baseData, rostersData) {
    for (const r of rostersData) {
      let currentPtsSum = 0;
      let pregameTotalProj = 0;
      let sleeperTotalProj = 0;
      let improvedTotalProj = 0;
      let playersCompleted = 0;
      let playersInProgress = 0;
      let playersYetToPlay = 0;
      let playersOnByeOrOut = 0;
      let totalRemLinear = 0;

      for (const m of r.starters) {
        currentPtsSum += m.currentPts;
        pregameTotalProj += m.pregameProj;
        sleeperTotalProj += m.sleeperProj;
        improvedTotalProj += m.improvedProj;
        totalRemLinear += m.factors.linearRemFrac;

        const isOut = ['Out', 'IR', 'Doubtful', 'Suspension', 'COV'].includes(m.injuryStatus);
        if (m.gameStatus === 'bye' || (isOut && m.gameStatus !== 'complete')) {
          playersOnByeOrOut++;
        }
        if (m.gameStatus === 'in_game') playersInProgress++;
        else if (m.gameStatus === 'pre_game') playersYetToPlay++;
        else playersCompleted++;
      }

      r.currentPts = Number(currentPtsSum.toFixed(2));
      r.pregameTotalProj = Number(pregameTotalProj.toFixed(2));
      r.sleeperTotalProj = Number(sleeperTotalProj.toFixed(2));
      r.improvedTotalProj = Number(improvedTotalProj.toFixed(2));
      r.projDeltaVsSleeper = Number((improvedTotalProj - sleeperTotalProj).toFixed(2));
      r.playersCompleted = playersCompleted;
      r.playersInProgress = playersInProgress;
      r.playersYetToPlay = playersYetToPlay;
      r.playersOnByeOrOut = playersOnByeOrOut;
      r.pctComplete = Number((((r.starters.length - totalRemLinear) / Math.max(1, r.starters.length)) * 100).toFixed(1));
    }

    const simOutput = runJointLeagueSimulation(rostersData, 10000);
    const simByRoster = new Map(simOutput.teamSimResults.map((s) => [s.rosterId, s]));
    for (const r of rostersData) {
      r.simulation = simByRoster.get(r.rosterId);
    }

    const matchupsById = new Map();
    for (const r of rostersData) {
      if (!matchupsById.has(r.matchupId)) matchupsById.set(r.matchupId, []);
      matchupsById.get(r.matchupId).push(r);
    }

    const matchupCards = Array.from(matchupsById.entries())
      .sort((a, b) => a[0] - b[0])
      .map(([matchupId, pair]) => ({
        matchupId,
        teamA: pair[0],
        teamB: pair[1] || null
      }));

    const medianLeaderboard = [...rostersData]
      .sort((a, b) => b.improvedTotalProj - a.improvedTotalProj || b.currentPts - a.currentPts)
      .map((r, idx) => ({
        rank: idx + 1,
        rosterId: r.rosterId,
        matchupId: r.matchupId,
        teamName: r.teamName,
        ownerName: r.ownerName,
        avatarUrl: r.avatarUrl,
        record: `${r.wins}-${r.losses}${r.ties ? `-${r.ties}` : ''}`,
        currentPts: r.currentPts,
        sleeperTotalProj: r.sleeperTotalProj,
        improvedTotalProj: r.improvedTotalProj,
        projDeltaVsSleeper: r.projDeltaVsSleeper,
        vsExpectedMedian: Number((r.improvedTotalProj - simOutput.leagueMedian.expectedMedianImproved).toFixed(2)),
        playersCompleted: r.playersCompleted,
        playersInProgress: r.playersInProgress,
        playersYetToPlay: r.playersYetToPlay,
        playersOnByeOrOut: r.playersOnByeOrOut,
        autoSubCount: r.autoSubCount || 0,
        pctComplete: r.pctComplete,
        simulation: r.simulation
      }));

    return {
      ...baseData,
      rostersData,
      leagueMedian: simOutput.leagueMedian,
      matchups: matchupCards,
      medianLeaderboard
    };
  }

  function applyAutoSubForByeOrOut(rostersData) {
    return rostersData.map((r) => {
      const clonedStarters = r.rawStarters.map((p) => ({ ...p, autoSubFor: null }));
      const availableBench = r.rawBench.map((p) => ({ ...p }));
      const usedBenchIds = new Set();
      let autoSubCount = 0;

      for (let i = 0; i < clonedStarters.length; i++) {
        const st = clonedStarters[i];
        const isOut = ['Out', 'IR', 'Doubtful', 'Suspension', 'COV'].includes(st.injuryStatus);
        const needsSub = st.gameStatus === 'bye' || (isOut && st.gameStatus === 'pre_game');
        if (!needsSub) continue;

        const slot = st.slot;
        let bestCandidate = null;
        for (const b of availableBench) {
          if (usedBenchIds.has(b.playerId)) continue;
          const bOut = ['Out', 'IR', 'Doubtful', 'Suspension', 'COV'].includes(b.injuryStatus);
          if (b.gameStatus === 'bye' || bOut || b.improvedProj <= 0.5) continue;
          if (isEligibleForSlot(b, slot)) {
            if (!bestCandidate || b.improvedProj > bestCandidate.improvedProj) {
              bestCandidate = b;
            }
          }
        }

        if (bestCandidate) {
          usedBenchIds.add(bestCandidate.playerId);
          const reasonTag = st.gameStatus === 'bye' ? 'BYE' : st.injuryStatus;
          clonedStarters[i] = {
            ...bestCandidate,
            slot,
            autoSubFor: `${st.shortName} (${reasonTag})`
          };
          autoSubCount++;
        }
      }

      const remainingBench = availableBench.filter((b) => !usedBenchIds.has(b.playerId));
      return {
        ...r,
        starters: clonedStarters,
        bench: remainingBench,
        autoSubCount
      };
    });
  }

  async function getLeagueLiveState(leagueId = DEFAULT_LEAGUE_ID, requestedWeek = null, forceRefresh = false, autoSubByeOut = false, marketWeight = 0.75, marketSource = 'consensus') {
    const startMs = Date.now();
    const ttl = forceRefresh ? 0 : 8000;

    const [nflState, league, users, rosters, udByName, owsFeed, espnBoard] = await Promise.all([
      fetchJson('https://api.sleeper.app/v1/state/nfl', {}, ttl),
      fetchJson(`https://api.sleeper.app/v1/league/${leagueId}`, {}, ttl * 4),
      fetchJson(`https://api.sleeper.app/v1/league/${leagueId}/users`, {}, ttl * 2),
      fetchJson(`https://api.sleeper.app/v1/league/${leagueId}/rosters`, {}, ttl),
      fetchUnderdogNflMap(forceRefresh ? 0 : 180000).catch(() => new Map()),
      fetchJson(OWS_SPORTSBOOK_URL, {}, 120000).catch(() => null),
      fetchJson(ESPN_SCOREBOARD_URL, {}, 60000).catch(() => null)
    ]);

    const season = league.season || nflState.season || '2026';
    const currentNflWeek = Number(nflState.display_week || nflState.week || league.settings?.leg || 8);
    const activeWeek = Number(requestedWeek || currentNflWeek);

    const canonicalPosQuery = 'position[]=DB&position[]=DL&position[]=LB&position[]=QB&position[]=RB&position[]=TE&position[]=WR&season_type=regular';
    const graphqlQuery = JSON.stringify({
      query: `query batch_scores { scores(sport: "nfl", season_type: "regular", season: "${season}", week: ${activeWeek}) { game_id status metadata date start_time } }`
    });

    const [matchups, projections, stats, scoresResp] = await Promise.all([
      fetchJson(`https://api.sleeper.app/v1/league/${leagueId}/matchups/${activeWeek}`, {}, ttl),
      fetchJson(`https://api.sleeper.com/projections/nfl/${season}/${activeWeek}?${canonicalPosQuery}`, {}, ttl * 4),
      fetchJson(`https://api.sleeper.com/stats/nfl/${season}/${activeWeek}?${canonicalPosQuery}`, {}, ttl),
      fetchJson('https://api.sleeper.com/graphql', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: graphqlQuery
      }, ttl)
    ]);

    // Build external game odds & sportsbook player prop maps (when activeWeek matches current week)
    const externalOddsMap = {};
    const sbPropsByName = new Map();
    const isCurrentWeek = activeWeek === currentNflWeek;

    if (espnBoard && Array.isArray(espnBoard.events)) {
      for (const ev of espnBoard.events) {
        const comp = ev.competitions?.[0];
        const odds = comp?.odds?.[0];
        if (!comp || !odds) continue;
        const competitors = comp.competitors || [];
        const homeComp = competitors.find((c) => c.homeAway === 'home');
        const awayComp = competitors.find((c) => c.homeAway === 'away');
        const homeAbbr = normalizeTeamCode(homeComp?.team?.abbreviation);
        const awayAbbr = normalizeTeamCode(awayComp?.team?.abbreviation);
        if (homeAbbr && awayAbbr) {
          const entry = {
            homeTeam: homeAbbr,
            awayTeam: awayAbbr,
            total: Number(odds.overUnder || 45.5),
            homeSpread: Number(odds.spread || 0)
          };
          externalOddsMap[homeAbbr] = entry;
          externalOddsMap[awayAbbr] = entry;
        }
      }
    }

    if (owsFeed && Number(owsFeed.week) === activeWeek) {
      for (const g of (owsFeed.games || [])) {
        const homeAbbr = normalizeTeamCode(g.home);
        const awayAbbr = normalizeTeamCode(g.away);
        const favAbbr = normalizeTeamCode(g.fav);
        const spreadMag = Math.abs(Number(g.spread || 0));
        const homeSpread = favAbbr === homeAbbr ? -spreadMag : (favAbbr === awayAbbr ? spreadMag : 0);
        const entry = {
          homeTeam: homeAbbr,
          awayTeam: awayAbbr,
          total: Number(g.total || 45.5),
          homeSpread
        };
        externalOddsMap[homeAbbr] = entry;
        externalOddsMap[awayAbbr] = entry;
      }
      for (const p of (owsFeed.players || [])) {
        sbPropsByName.set(normalizePlayerName(p.n), p);
      }
    }

    const scoresList = scoresResp?.data?.scores || [];
    const gameStatesByTeam = {};
    const allGames = [];

    for (const g of scoresList) {
      const gs = analyzeGameState(g, externalOddsMap);
      allGames.push(gs);
      if (gs.homeTeam) gameStatesByTeam[gs.homeTeam] = gs;
      if (gs.awayTeam) gameStatesByTeam[gs.awayTeam] = gs;
    }

    const projMap = new Map();
    if (Array.isArray(projections)) {
      for (const p of projections) projMap.set(String(p.player_id), p);
    }
    const statMap = new Map();
    if (Array.isArray(stats)) {
      for (const s of stats) statMap.set(String(s.player_id), s);
    }

    const userMap = new Map(users.map((u) => [u.user_id, u]));
    const rosterMap = new Map(rosters.map((r) => [r.roster_id, r]));
    const scoringSettings = league.scoring_settings || {};
    const starterSlots = (league.roster_positions || []).filter((p) => p !== 'BN');

    let rostersData = matchups.map((m) => {
      const r = rosterMap.get(m.roster_id) || {};
      const u = userMap.get(r.owner_id) || {};
      const teamName = u.metadata?.team_name || u.display_name || `Team ${m.roster_id}`;
      const ownerName = u.display_name || `User ${m.roster_id}`;
      const avatarUrl = u.metadata?.avatar || (u.avatar ? `https://sleepercdn.com/avatars/thumbs/${u.avatar}` : null);

      const starterIds = (m.starters && m.starters.length ? m.starters : (r.starters || [])).map(String);
      const allPlayerIds = (m.players && m.players.length ? m.players : (r.players || [])).map(String);
      const starterPts = m.starters_points || [];
      const playersPoints = m.players_points || {};

      const buildModeledPlayer = (pidStr, actualPts, slotName) => {
        const pEntry = projMap.get(pidStr);
        const sEntry = statMap.get(pidStr);
        const pInfo = pEntry?.player || sEntry?.player || {};
        const fullName = `${pInfo.first_name || ''} ${pInfo.last_name || ''}`.trim();
        const normKey = normalizePlayerName(fullName);
        const sbEntry = sbPropsByName.get(normKey) || null;
        const udEntry = (isCurrentWeek && udByName) ? (udByName.get(normKey) || null) : null;
        const team = normalizeTeamCode(pEntry?.team || sEntry?.team || pInfo.team || 'FA');
        const gs = gameStatesByTeam[team] || null;
        const modeled = modelPlayer(pidStr, actualPts, pEntry, sEntry, gs, scoringSettings, sbEntry, marketWeight, udEntry, marketSource);
        modeled.slot = slotName;
        return modeled;
      };

      const starters = starterIds.map((pidStr, idx) => {
        const actualPts = starterPts[idx] !== undefined ? starterPts[idx] : (playersPoints[pidStr] ?? 0);
        return buildModeledPlayer(pidStr, actualPts, starterSlots[idx] || 'FLEX');
      });

      const starterSet = new Set(starterIds.filter((id) => id !== '0'));
      const benchIds = allPlayerIds.filter((pid) => !starterSet.has(pid));
      const bench = benchIds
        .map((pidStr) => buildModeledPlayer(pidStr, playersPoints[pidStr] ?? 0, 'BN'))
        .sort((a, b) => b.improvedProj - a.improvedProj);

      return {
        rosterId: m.roster_id,
        matchupId: m.matchup_id,
        teamName,
        ownerName,
        avatarUrl,
        division: r.settings?.division || 1,
        wins: r.settings?.wins || 0,
        losses: r.settings?.losses || 0,
        ties: r.settings?.ties || 0,
        fptsSeason: Number(((r.settings?.fpts || 0) + (r.settings?.fpts_decimal || 0) / 100).toFixed(2)),
        rawStarters: starters,
        rawBench: bench,
        starters,
        bench,
        autoSubCount: 0
      };
    });

    if (autoSubByeOut) {
      rostersData = applyAutoSubForByeOrOut(rostersData);
    }

    const baseData = {
      timestamp: new Date().toISOString(),
      computeTimeMs: Date.now() - startMs,
      autoSubByeOut,
      marketWeight,
      marketSource,
      sportsbookWeekLoaded: owsFeed ? Number(owsFeed.week) : null,
      underdogPlayersLoaded: udByName ? udByName.size : 0,
      league: {
        leagueId: league.league_id,
        name: league.name,
        season,
        week: activeWeek,
        currentNflWeek,
        playoffWeekStart: league.settings?.playoff_week_start || 14,
        totalRosters: league.total_rosters,
        leagueAverageMatch: Boolean(league.settings?.league_average_match),
        divisions: {
          1: league.metadata?.division_1 || 'Division 1',
          2: league.metadata?.division_2 || 'Division 2'
        },
        starterSlots
      },
      nflGames: allGames
    };

    const result = recomputeLeagueFromRosters(baseData, rostersData);
    result.computeTimeMs = Date.now() - startMs;
    return result;
  }

  /**
   * Reconstruct the exact Pregame State of any historical week (e.g. Weeks 1..7)
   * with zero hindsight leakage for pregame projections/odds, while attaching
   * `realizedActualPts` to every player for ex-post backtest evaluation.
   */
  async function getPregameHistoricalWeekState(leagueId = DEFAULT_LEAGUE_ID, week = 1, marketWeight = 0.75) {
    const startMs = Date.now();
    const longTtl = 3600000; // 1 hour cache for completed historical weeks

    const [nflState, league, users, rosters] = await Promise.all([
      fetchJson('https://api.sleeper.app/v1/state/nfl', {}, 60000),
      fetchJson(`https://api.sleeper.app/v1/league/${leagueId}`, {}, longTtl),
      fetchJson(`https://api.sleeper.app/v1/league/${leagueId}/users`, {}, longTtl),
      fetchJson(`https://api.sleeper.app/v1/league/${leagueId}/rosters`, {}, longTtl)
    ]);

    const season = league.season || nflState.season || '2026';
    const currentNflWeek = Number(nflState.display_week || nflState.week || league.settings?.leg || 8);
    const activeWeek = Number(week);

    const canonicalPosQuery = 'position[]=DB&position[]=DL&position[]=LB&position[]=QB&position[]=RB&position[]=TE&position[]=WR&season_type=regular';
    const graphqlQuery = JSON.stringify({
      query: `query batch_scores { scores(sport: "nfl", season_type: "regular", season: "${season}", week: ${activeWeek}) { game_id status metadata date start_time } }`
    });

    const [matchups, projections, scoresResp] = await Promise.all([
      fetchJson(`https://api.sleeper.app/v1/league/${leagueId}/matchups/${activeWeek}`, {}, longTtl),
      fetchJson(`https://api.sleeper.com/projections/nfl/${season}/${activeWeek}?${canonicalPosQuery}`, {}, longTtl),
      fetchJson('https://api.sleeper.com/graphql', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: graphqlQuery
      }, longTtl)
    ]);

    // Derive pregame implied game totals from each team's QB1 + top skill projections + historical metadata.spread
    const teamPlayersByTeam = {};
    const projMap = new Map();
    if (Array.isArray(projections)) {
      for (const p of projections) {
        projMap.set(String(p.player_id), p);
        const t = normalizeTeamCode(p.team || p.player?.team || '');
        if (!t || !p.stats) continue;
        if (!teamPlayersByTeam[t]) teamPlayersByTeam[t] = [];
        teamPlayersByTeam[t].push(p);
      }
    }

    const teamOffProj = {};
    for (const [t, pList] of Object.entries(teamPlayersByTeam)) {
      const qbs = pList.filter((x) => x.player?.position === 'QB').sort((a, b) => (b.stats.pass_yd || 0) - (a.stats.pass_yd || 0));
      const rbs = pList.filter((x) => x.player?.position === 'RB').sort((a, b) => (b.stats.rush_yd || 0) - (a.stats.rush_yd || 0));
      const qb1 = qbs[0]?.stats || {};
      const rb1 = rbs[0]?.stats || {};
      const rb2 = rbs[1]?.stats || {};
      const passTd = Number(qb1.pass_td || 1.35);
      const passYd = Number(qb1.pass_yd || 215);
      const rushTd = Number(rb1.rush_td || 0.45) + Number(rb2.rush_td || 0.20) + Number(qb1.rush_td || 0.15);
      const rushYd = Number(rb1.rush_yd || 60) + Number(rb2.rush_yd || 28) + Number(qb1.rush_yd || 15);
      teamOffProj[t] = { passTd, rushTd, passYd, rushYd };
    }

    const externalOddsMap = {};
    const scoresList = scoresResp?.data?.scores || [];
    for (const g of scoresList) {
      const meta = g.metadata || {};
      const homeTeam = normalizeTeamCode(meta.home_team || '');
      const awayTeam = normalizeTeamCode(meta.away_team || '');
      if (!homeTeam || !awayTeam) continue;
      const hOff = teamOffProj[homeTeam] || { passTd: 1.4, rushTd: 0.8, passYd: 220, rushYd: 105 };
      const aOff = teamOffProj[awayTeam] || { passTd: 1.4, rushTd: 0.8, passYd: 220, rushYd: 105 };
      const hEstPts = 6.8 * (hOff.passTd + hOff.rushTd) + 0.012 * (hOff.passYd + hOff.rushYd) + 3.0;
      const aEstPts = 6.8 * (aOff.passTd + aOff.rushTd) + 0.012 * (aOff.passYd + aOff.rushYd) + 3.0;
      const estTotal = Math.max(38.5, Math.min(52.5, Number((hEstPts + aEstPts).toFixed(1))));
      const slpSpread = meta.spread || meta.pickem_spread || {};
      const homeSpread = slpSpread[homeTeam] !== undefined ? Number(slpSpread[homeTeam]) : Number((aEstPts - hEstPts).toFixed(1));
      const entry = {
        homeTeam,
        awayTeam,
        total: estTotal,
        homeSpread
      };
      externalOddsMap[homeTeam] = entry;
      externalOddsMap[awayTeam] = entry;
    }

    const gameStatesByTeam = {};
    const allGames = [];
    for (const g of scoresList) {
      const gs = analyzeGameState(g, externalOddsMap, true); // asOfPregame = true!
      allGames.push(gs);
      if (gs.homeTeam) gameStatesByTeam[gs.homeTeam] = gs;
      if (gs.awayTeam) gameStatesByTeam[gs.awayTeam] = gs;
    }

    const userMap = new Map(users.map((u) => [u.user_id, u]));
    const rosterMap = new Map(rosters.map((r) => [r.roster_id, r]));
    const scoringSettings = league.scoring_settings || {};
    const starterSlots = (league.roster_positions || []).filter((p) => p !== 'BN');

    const rostersData = matchups.map((m) => {
      const r = rosterMap.get(m.roster_id) || {};
      const u = userMap.get(r.owner_id) || {};
      const teamName = u.metadata?.team_name || u.display_name || `Team ${m.roster_id}`;
      const ownerName = u.display_name || `User ${m.roster_id}`;
      const avatarUrl = u.metadata?.avatar || (u.avatar ? `https://sleepercdn.com/avatars/thumbs/${u.avatar}` : null);

      const starterIds = (m.starters && m.starters.length ? m.starters : (r.starters || [])).map(String);
      const allPlayerIds = (m.players && m.players.length ? m.players : (r.players || [])).map(String);
      const starterPts = m.starters_points || [];
      const playersPoints = m.players_points || {};

      const buildHistoricalPregamePlayer = (pidStr, realizedPts, slotName) => {
        const rawPEntry = projMap.get(pidStr);
        // Prevent current-week injury status on `player` metadata from zeroing out a player who had positive pregame projections in Week `w`
        let pEntry = rawPEntry;
        if (rawPEntry && rawPEntry.player) {
          const preProj = computeCustomPoints(rawPEntry.stats || {}, scoringSettings);
          pEntry = {
            ...rawPEntry,
            player: {
              ...rawPEntry.player,
              injury_status: preProj > 0.5 ? null : rawPEntry.player.injury_status
            }
          };
        }
        const team = normalizeTeamCode(pEntry?.team || pEntry?.player?.team || 'FA');
        const gs = gameStatesByTeam[team] || null;
        // Pass actualPts = 0 and statEntry = null so modelPlayer runs 100% pregame!
        const modeled = modelPlayer(pidStr, 0, pEntry, null, gs, scoringSettings, null, marketWeight);
        modeled.slot = slotName;
        modeled.realizedActualPts = Number((realizedPts ?? 0).toFixed(2));
        return modeled;
      };

      const starters = starterIds.map((pidStr, idx) => {
        const realizedPts = starterPts[idx] !== undefined ? starterPts[idx] : (playersPoints[pidStr] ?? 0);
        return buildHistoricalPregamePlayer(pidStr, realizedPts, starterSlots[idx] || 'FLEX');
      });

      const starterSet = new Set(starterIds.filter((id) => id !== '0'));
      const benchIds = allPlayerIds.filter((pid) => !starterSet.has(pid));
      const bench = benchIds
        .map((pidStr) => buildHistoricalPregamePlayer(pidStr, playersPoints[pidStr] ?? 0, 'BN'))
        .sort((a, b) => b.improvedProj - a.improvedProj);

      const realizedTeamPts = Number((m.points !== undefined ? m.points : starters.reduce((a, b) => a + b.realizedActualPts, 0)).toFixed(2));

      return {
        rosterId: m.roster_id,
        matchupId: m.matchup_id,
        teamName,
        ownerName,
        avatarUrl,
        division: r.settings?.division || 1,
        wins: r.settings?.wins || 0,
        losses: r.settings?.losses || 0,
        ties: r.settings?.ties || 0,
        realizedTeamPts,
        rawStarters: starters,
        rawBench: bench,
        starters,
        bench,
        autoSubCount: 0
      };
    });

    const baseData = {
      timestamp: new Date().toISOString(),
      computeTimeMs: Date.now() - startMs,
      autoSubByeOut: false,
      marketWeight,
      isHistoricalPregame: true,
      league: {
        leagueId: league.league_id,
        name: league.name,
        season,
        week: activeWeek,
        currentNflWeek,
        playoffWeekStart: league.settings?.playoff_week_start || 14,
        totalRosters: league.total_rosters,
        leagueAverageMatch: Boolean(league.settings?.league_average_match),
        divisions: {
          1: league.metadata?.division_1 || 'Division 1',
          2: league.metadata?.division_2 || 'Division 2'
        },
        starterSlots
      },
      nflGames: allGames
    };

    return recomputeLeagueFromRosters(baseData, rostersData);
  }

  /**
   * Run a full multi-week Historical Backtest (Weeks 1 .. currentNflWeek-1) comparing:
   *  1) Actual Manager-Set Lineup vs. Naive Sleeper Static-Max vs. Tool Optimal Lineup vs. Hindsight Best-Ball
   *  2) Our Pregame Model Expectations (Expected Points, p10-p90 CI, H2H Win %, Median Win %, Expected Median Cutoff)
   *     against Actual Realized Outcomes & Sleeper's Baseline Expectations.
   */
  async function runHistoricalBacktest(leagueId = DEFAULT_LEAGUE_ID, targetRosterId = 3, options = {}) {
    const startMs = Date.now();
    const marketWeight = options.marketWeight !== undefined ? Number(options.marketWeight) : 0.75;
    const objective = options.objective || 'total_wins';

    const nflState = await fetchJson('https://api.sleeper.app/v1/state/nfl', {}, 60000);
    const currentNflWeek = Number(nflState.display_week || nflState.week || 8);
    const maxCompletedWeek = Math.max(1, (options.throughWeek || (currentNflWeek - 1)));

    const weekNumbers = [];
    for (let w = 1; w <= maxCompletedWeek; w++) weekNumbers.push(w);

    // Fetch all historical weeks in parallel
    const pregameStates = await Promise.all(
      weekNumbers.map((w) => getPregameHistoricalWeekState(leagueId, w, marketWeight))
    );

    const weeklyResults = [];
    const perTeamStatsMap = new Map();
    let leagueBrierH2hToolSum = 0;
    let leagueBrierH2hSlpSum = 0;
    let leagueBrierMedToolSum = 0;
    let leagueBrierMedSlpSum = 0;
    let leagueMaePtsToolSum = 0;
    let leagueMaePtsSlpSum = 0;
    let leagueCi80CoveredCount = 0;
    let totalTeamWeeks = 0;

    for (let idx = 0; idx < pregameStates.length; idx++) {
      const w = weekNumbers[idx];
      const state = pregameStates[idx];
      const slots = state.league.starterSlots || [];

      // Actual realized scores for all 10 teams in Week w
      const allRealizedScores = state.rostersData.map((r) => ({
        rosterId: r.rosterId,
        matchupId: r.matchupId,
        teamName: r.teamName,
        realizedPts: r.realizedTeamPts
      }));
      const sortedActual = [...allRealizedScores].sort((a, b) => b.realizedPts - a.realizedPts);
      const actualMedianCutoff = Number((0.5 * (sortedActual[4].realizedPts + sortedActual[5].realizedPts)).toFixed(2));

      // Accumulate league-wide calibration stats across all 10 teams in Week w
      for (const r of state.rostersData) {
        const opp = state.rostersData.find((x) => x.matchupId === r.matchupId && x.rosterId !== r.rosterId);
        const actualH2hWin = opp ? (r.realizedTeamPts > opp.realizedTeamPts ? 1 : (r.realizedTeamPts === opp.realizedTeamPts ? 0.5 : 0)) : 0.5;
        const actualMedWin = r.realizedTeamPts > actualMedianCutoff ? 1 : 0;

        const sim = r.simulation;
        leagueBrierH2hToolSum += Math.pow(sim.h2hWinProbImproved - actualH2hWin, 2);
        leagueBrierH2hSlpSum += Math.pow(sim.h2hWinProbSleeper - actualH2hWin, 2);
        leagueBrierMedToolSum += Math.pow(sim.medianWinProbImproved - actualMedWin, 2);
        leagueBrierMedSlpSum += Math.pow(sim.medianWinProbSleeper - actualMedWin, 2);

        leagueMaePtsToolSum += Math.abs(r.improvedTotalProj - r.realizedTeamPts);
        leagueMaePtsSlpSum += Math.abs(r.sleeperTotalProj - r.realizedTeamPts);

        const inCi80 = r.realizedTeamPts >= sim.p10 && r.realizedTeamPts <= sim.p90;
        if (inCi80) {
          leagueCi80CoveredCount++;
        }
        totalTeamWeeks++;

        if (!perTeamStatsMap.has(r.rosterId)) {
          perTeamStatsMap.set(r.rosterId, {
            rosterId: r.rosterId,
            teamName: r.teamName,
            ownerName: r.ownerName,
            weeks: 0,
            toolExpectedPts: 0,
            sleeperExpectedPts: 0,
            realizedPts: 0,
            toolExpectedWins: 0,
            sleeperExpectedWins: 0,
            realizedH2hWins: 0,
            realizedMedWins: 0,
            realizedTotalWins: 0,
            ci80CoveredWeeks: 0
          });
        }
        const tEntry = perTeamStatsMap.get(r.rosterId);
        tEntry.weeks += 1;
        tEntry.toolExpectedPts += r.improvedTotalProj;
        tEntry.sleeperExpectedPts += r.sleeperTotalProj;
        tEntry.realizedPts += r.realizedTeamPts;
        tEntry.toolExpectedWins += sim.expectedWeeklyWinsImproved;
        tEntry.sleeperExpectedWins += sim.expectedWeeklyWinsSleeper;
        tEntry.realizedH2hWins += actualH2hWin;
        tEntry.realizedMedWins += actualMedWin;
        tEntry.realizedTotalWins += (actualH2hWin + actualMedWin);
        if (inCi80) tEntry.ci80CoveredWeeks += 1;
      }

      // Run pregame simulation optimizer for targetRosterId in Week w
      const opt = runSimulationLineupOptimizer(state, targetRosterId, {
        numSims: 4000,
        objective,
        marketWeight
      });

      const targetRoster = state.rostersData.find((r) => r.rosterId === Number(targetRosterId)) || state.rostersData[0];
      const oppRoster = state.rostersData.find((r) => r.matchupId === targetRoster.matchupId && r.rosterId !== targetRoster.rosterId);
      const other9ActualScores = state.rostersData
        .filter((r) => r.rosterId !== targetRoster.rosterId)
        .map((r) => r.realizedTeamPts)
        .sort((a, b) => b - a);
      // To beat the Top-5 median among 10 teams, a team's score must exceed at least 5 of the other 9 teams (i.e. > other9ActualScores[4])
      const targetMedianThreshold = other9ActualScores[4] || actualMedianCutoff;

      // Helper to evaluate what any 14-player lineup actually scored in Week w
      const evaluateRealizedLineup = (lineup) => {
        const realizedPts = Number(lineup.reduce((sum, p) => sum + (p?.realizedActualPts || 0), 0).toFixed(2));
        const oppPts = oppRoster ? oppRoster.realizedTeamPts : 0;
        const h2hWin = realizedPts > oppPts ? 1 : (realizedPts === oppPts ? 0.5 : 0);
        const medWin = realizedPts > targetMedianThreshold ? 1 : (realizedPts === targetMedianThreshold ? 0.5 : 0);
        return {
          realizedPts,
          h2hWin,
          medWin,
          totalWins: h2hWin + medWin
        };
      };

      // Build Hindsight Best-Ball Lineup (greedy on realizedActualPts)
      const allPool = [...targetRoster.rawStarters, ...targetRoster.rawBench].filter((p) => p && p.playerId !== '0');
      const usedHindsight = new Set();
      const hindsightLineup = new Array(slots.length).fill(null);
      const slotOrder = slots.map((s, i) => ({ s, i, prio: s.includes('FLEX') ? 2 : 1 })).sort((a, b) => a.prio - b.prio);
      for (const { s, i } of slotOrder) {
        const cands = allPool
          .filter((p) => !usedHindsight.has(p.playerId) && isEligibleForSlot(p, s))
          .sort((a, b) => (b.realizedActualPts || 0) - (a.realizedActualPts || 0));
        if (cands.length > 0) {
          hindsightLineup[i] = { ...cands[0], slot: s };
          usedHindsight.add(cands[0].playerId);
        }
      }

      const actualSetEval = evaluateRealizedLineup(targetRoster.rawStarters);
      const sleeperMaxEval = evaluateRealizedLineup(opt.sleeperMaxLineup);
      const toolOptEval = evaluateRealizedLineup(opt.optimalCandidate.lineup);
      const hindsightEval = evaluateRealizedLineup(hindsightLineup.filter(Boolean));

      // Audit recommended swaps in Week w against realized actual points
      const swapAudits = opt.recommendedSwaps.map((sw) => {
        const outActual = sw.outPlayer ? (sw.outPlayer.realizedActualPts || 0) : 0;
        const inActual = sw.inPlayer ? (sw.inPlayer.realizedActualPts || 0) : 0;
        return {
          slot: sw.slot,
          outPlayer: {
            shortName: sw.outPlayer?.shortName || 'Empty',
            fullName: sw.outPlayer?.fullName || 'Empty Slot',
            team: sw.outPlayer?.team || 'FA',
            position: sw.outPlayer?.position || '',
            expectedPts: sw.outPlayer?.improvedProj || 0,
            sleeperPts: sw.outPlayer?.sleeperProj || 0,
            actualPts: outActual
          },
          inPlayer: {
            shortName: sw.inPlayer?.shortName || 'Empty',
            fullName: sw.inPlayer?.fullName || 'Empty Slot',
            team: sw.inPlayer?.team || 'FA',
            position: sw.inPlayer?.position || '',
            expectedPts: sw.inPlayer?.improvedProj || 0,
            sleeperPts: sw.inPlayer?.sleeperProj || 0,
            actualPts: inActual
          },
          expectedDelta: sw.projDelta,
          actualDelta: Number((inActual - outActual).toFixed(2)),
          reasons: sw.reasons,
          summaryReason: sw.summaryReason
        };
      });

      weeklyResults.push({
        week: w,
        targetTeamName: targetRoster.teamName,
        oppTeamName: oppRoster ? oppRoster.teamName : 'Opponent',
        oppExpectedPts: oppRoster ? oppRoster.improvedTotalProj : 0,
        oppSleeperPts: oppRoster ? oppRoster.sleeperTotalProj : 0,
        oppActualPts: oppRoster ? oppRoster.realizedTeamPts : 0,
        medianCutoff: {
          toolExpected: state.leagueMedian.expectedMedianImproved,
          sleeperStatic: state.leagueMedian.staticMedianSleeper,
          actualRealized: actualMedianCutoff,
          toolError: Number((state.leagueMedian.expectedMedianImproved - actualMedianCutoff).toFixed(2)),
          sleeperError: Number((state.leagueMedian.staticMedianSleeper - actualMedianCutoff).toFixed(2))
        },
        actualSet: {
          expectedPts: opt.currentSetMetrics.expectedPts,
          sleeperPts: opt.currentSetMetrics.sleeperPts,
          p10: opt.currentSetMetrics.p10,
          p50: opt.currentSetMetrics.p50,
          p90: opt.currentSetMetrics.p90,
          expectedH2hWinProb: opt.currentSetMetrics.h2hWinProb,
          expectedMedWinProb: opt.currentSetMetrics.medianWinProb,
          expectedWeeklyWins: opt.currentSetMetrics.expectedWeeklyWins,
          realizedPts: actualSetEval.realizedPts,
          realizedH2hWin: actualSetEval.h2hWin,
          realizedMedWin: actualSetEval.medWin,
          realizedTotalWins: actualSetEval.totalWins,
          insideCi80: actualSetEval.realizedPts >= opt.currentSetMetrics.p10 && actualSetEval.realizedPts <= opt.currentSetMetrics.p90
        },
        sleeperMax: {
          expectedPts: opt.sleeperMaxMetrics.expectedPts,
          sleeperPts: opt.sleeperMaxMetrics.sleeperPts,
          expectedWeeklyWins: opt.sleeperMaxMetrics.expectedWeeklyWins,
          realizedPts: sleeperMaxEval.realizedPts,
          realizedH2hWin: sleeperMaxEval.h2hWin,
          realizedMedWin: sleeperMaxEval.medWin,
          realizedTotalWins: sleeperMaxEval.totalWins,
          ptsGainVsActualSet: Number((sleeperMaxEval.realizedPts - actualSetEval.realizedPts).toFixed(2)),
          winsGainVsActualSet: sleeperMaxEval.totalWins - actualSetEval.totalWins
        },
        toolOptimal: {
          expectedPts: opt.optimalCandidate.metrics.expectedPts,
          sleeperPts: opt.optimalCandidate.metrics.sleeperPts,
          p10: opt.optimalCandidate.metrics.p10,
          p50: opt.optimalCandidate.metrics.p50,
          p90: opt.optimalCandidate.metrics.p90,
          expectedH2hWinProb: opt.optimalCandidate.metrics.h2hWinProb,
          expectedMedWinProb: opt.optimalCandidate.metrics.medianWinProb,
          expectedWeeklyWins: opt.optimalCandidate.metrics.expectedWeeklyWins,
          realizedPts: toolOptEval.realizedPts,
          realizedH2hWin: toolOptEval.h2hWin,
          realizedMedWin: toolOptEval.medWin,
          realizedTotalWins: toolOptEval.totalWins,
          ptsGainVsActualSet: Number((toolOptEval.realizedPts - actualSetEval.realizedPts).toFixed(2)),
          ptsGainVsSleeperMax: Number((toolOptEval.realizedPts - sleeperMaxEval.realizedPts).toFixed(2)),
          winsGainVsActualSet: toolOptEval.totalWins - actualSetEval.totalWins,
          winsGainVsSleeperMax: toolOptEval.totalWins - sleeperMaxEval.totalWins,
          insideCi80: toolOptEval.realizedPts >= opt.optimalCandidate.metrics.p10 && toolOptEval.realizedPts <= opt.optimalCandidate.metrics.p90
        },
        hindsightBest: {
          realizedPts: hindsightEval.realizedPts,
          realizedTotalWins: hindsightEval.totalWins
        },
        swapAudits
      });
    }

    // Aggregate Summary across all backtested weeks
    const sumField = (arr, fn) => arr.reduce((acc, x) => acc + fn(x), 0);
    const allSwaps = weeklyResults.flatMap((w) => w.swapAudits.map((s) => ({ ...s, week: w.week })));
    const winningSwaps = allSwaps.filter((s) => s.actualDelta > 0);

    const summary = {
      weeksEvaluated: weeklyResults.length,
      targetRosterId: Number(targetRosterId),
      targetTeamName: weeklyResults[0]?.targetTeamName || 'Target Team',
      // Realized Points Totals
      actualSetTotalPts: Number(sumField(weeklyResults, (w) => w.actualSet.realizedPts).toFixed(2)),
      sleeperMaxTotalPts: Number(sumField(weeklyResults, (w) => w.sleeperMax.realizedPts).toFixed(2)),
      toolOptimalTotalPts: Number(sumField(weeklyResults, (w) => w.toolOptimal.realizedPts).toFixed(2)),
      hindsightBestTotalPts: Number(sumField(weeklyResults, (w) => w.hindsightBest.realizedPts).toFixed(2)),
      toolPtsGainVsActualSet: Number(sumField(weeklyResults, (w) => w.toolOptimal.ptsGainVsActualSet).toFixed(2)),
      toolPtsGainVsSleeperMax: Number(sumField(weeklyResults, (w) => w.toolOptimal.ptsGainVsSleeperMax).toFixed(2)),
      // Expected Points Totals (for Expectations vs Actual calibration)
      actualSetExpectedTotalPts: Number(sumField(weeklyResults, (w) => w.actualSet.expectedPts).toFixed(2)),
      actualSetSleeperExpectedPts: Number(sumField(weeklyResults, (w) => w.actualSet.sleeperPts).toFixed(2)),
      toolOptimalExpectedTotalPts: Number(sumField(weeklyResults, (w) => w.toolOptimal.expectedPts).toFixed(2)),
      // Realized Wins vs Expected Wins (out of 2 * weeksEvaluated possible games)
      maxPossibleWins: weeklyResults.length * 2,
      actualSetRealizedWins: sumField(weeklyResults, (w) => w.actualSet.realizedTotalWins),
      actualSetRealizedH2hWins: sumField(weeklyResults, (w) => w.actualSet.realizedH2hWin),
      actualSetRealizedMedWins: sumField(weeklyResults, (w) => w.actualSet.realizedMedWin),
      actualSetExpectedWins: Number(sumField(weeklyResults, (w) => w.actualSet.expectedWeeklyWins).toFixed(2)),
      sleeperMaxRealizedWins: sumField(weeklyResults, (w) => w.sleeperMax.realizedTotalWins),
      toolOptimalRealizedWins: sumField(weeklyResults, (w) => w.toolOptimal.realizedTotalWins),
      toolOptimalRealizedH2hWins: sumField(weeklyResults, (w) => w.toolOptimal.realizedH2hWin),
      toolOptimalRealizedMedWins: sumField(weeklyResults, (w) => w.toolOptimal.realizedMedWin),
      toolOptimalExpectedWins: Number(sumField(weeklyResults, (w) => w.toolOptimal.expectedWeeklyWins).toFixed(2)),
      toolWinsGainVsActualSet: sumField(weeklyResults, (w) => w.toolOptimal.winsGainVsActualSet),
      toolWinsGainVsSleeperMax: sumField(weeklyResults, (w) => w.toolOptimal.winsGainVsSleeperMax),
      // Swap Audit Stats
      totalSwapsRecommended: allSwaps.length,
      positiveSwapsCount: winningSwaps.length,
      swapHitRate: allSwaps.length > 0 ? Number((winningSwaps.length / allSwaps.length).toFixed(4)) : 0,
      avgPtsGainedPerWeek: Number((sumField(weeklyResults, (w) => w.toolOptimal.ptsGainVsActualSet) / Math.max(1, weeklyResults.length)).toFixed(2)),
      // League-Wide Model Calibration (Across all 70 Team-Weeks)
      calibration: {
        totalTeamWeeks,
        toolBrierH2h: Number((leagueBrierH2hToolSum / Math.max(1, totalTeamWeeks)).toFixed(4)),
        sleeperBrierH2h: Number((leagueBrierH2hSlpSum / Math.max(1, totalTeamWeeks)).toFixed(4)),
        toolBrierMed: Number((leagueBrierMedToolSum / Math.max(1, totalTeamWeeks)).toFixed(4)),
        sleeperBrierMed: Number((leagueBrierMedSlpSum / Math.max(1, totalTeamWeeks)).toFixed(4)),
        toolMaePts: Number((leagueMaePtsToolSum / Math.max(1, totalTeamWeeks)).toFixed(2)),
        sleeperMaePts: Number((leagueMaePtsSlpSum / Math.max(1, totalTeamWeeks)).toFixed(2)),
        ci80CoverageRate: Number((leagueCi80CoveredCount / Math.max(1, totalTeamWeeks)).toFixed(4)),
        toolMedianCutoffMae: Number((sumField(weeklyResults, (w) => Math.abs(w.medianCutoff.toolError)) / Math.max(1, weeklyResults.length)).toFixed(2)),
        sleeperMedianCutoffMae: Number((sumField(weeklyResults, (w) => Math.abs(w.medianCutoff.sleeperError)) / Math.max(1, weeklyResults.length)).toFixed(2))
      }
    };

    const leagueTeamCalibration = Array.from(perTeamStatsMap.values())
      .map((t) => ({
        rosterId: t.rosterId,
        teamName: t.teamName,
        ownerName: t.ownerName,
        weeks: t.weeks,
        toolExpectedPts: Number(t.toolExpectedPts.toFixed(2)),
        sleeperExpectedPts: Number(t.sleeperExpectedPts.toFixed(2)),
        realizedPts: Number(t.realizedPts.toFixed(2)),
        ptsVsToolExp: Number((t.realizedPts - t.toolExpectedPts).toFixed(2)),
        ptsVsSleeperExp: Number((t.realizedPts - t.sleeperExpectedPts).toFixed(2)),
        toolExpectedWins: Number(t.toolExpectedWins.toFixed(2)),
        sleeperExpectedWins: Number(t.sleeperExpectedWins.toFixed(2)),
        realizedH2hWins: t.realizedH2hWins,
        realizedMedWins: t.realizedMedWins,
        realizedTotalWins: t.realizedTotalWins,
        winsVsToolExp: Number((t.realizedTotalWins - t.toolExpectedWins).toFixed(2)),
        ci80CoveredWeeks: t.ci80CoveredWeeks,
        ci80CoverageRate: Number((t.ci80CoveredWeeks / Math.max(1, t.weeks)).toFixed(4))
      }))
      .sort((a, b) => b.realizedTotalWins - a.realizedTotalWins || b.realizedPts - a.realizedPts);

    return {
      computeTimeMs: Date.now() - startMs,
      summary,
      weeklyResults,
      leagueTeamCalibration
    };
  }

  return {
    DEFAULT_LEAGUE_ID,
    getLeagueLiveState,
    getPregameHistoricalWeekState,
    runHistoricalBacktest,
    recomputeLeagueFromRosters,
    applyAutoSubForByeOrOut,
    runJointLeagueSimulation,
    runSimulationLineupOptimizer,
    computePairwisePlayerCorrelation,
    isEligibleForSlot
  };
});
