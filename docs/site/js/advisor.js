/**
 * Darts Cricket Strategy Advisor — Strategy Engine + UI
 * Ports all 22 strategies (S1-S17, E1-E4, PS) from Python to JS.
 * Client-side only, no dependencies.
 */
(function () {
  'use strict';

  /* ================================================================== */
  /*  Constants                                                          */
  /* ================================================================== */

  var TARGETS = [15, 16, 17, 18, 19, 20, 25];
  var DESCENDING = [20, 19, 18, 17, 16, 15, 25];
  var EARLY_BULL_ORDER = [20, 19, 18, 17, 25, 16, 15];
  var MARKS_TO_CLOSE = 3;

  // Frongello strategy parameters: [leadMult, extraDarts, chase]
  // leadMult=null means S1 (always cover)
  var FRONGELLO_PARAMS = {
    S1:  [null, false, false],
    S2:  [0,    false, false],
    S3:  [3,    false, false],
    S4:  [6,    false, false],
    S5:  [9,    false, false],
    S6:  [0,    true,  false],
    S7:  [3,    true,  false],
    S8:  [6,    true,  false],
    S9:  [9,    true,  false],
    S10: [0,    false, true],
    S11: [3,    false, true],
    S12: [6,    false, true],
    S13: [9,    false, true],
    S14: [0,    true,  true],
    S15: [3,    true,  true],
    S16: [6,    true,  true],
    S17: [9,    true,  true]
  };

  var STRATEGY_NAMES = [
    'S1','S2','S3','S4','S5','S6','S7','S8','S9',
    'S10','S11','S12','S13','S14','S15','S16','S17',
    'E1','E2','E3','E4','PS'
  ];

  var MPR_LEVELS = [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.6, 4.0, 4.9, 5.6];

  /* ================================================================== */
  /*  Game State Helpers                                                  */
  /* ================================================================== */

  /**
   * Game state object shape:
   * {
   *   marks: [[p0 marks x7], [p1 marks x7]],  // index matches TARGETS
   *   scores: [p0Score, p1Score],
   *   dartsRemaining: 1|2|3
   * }
   * player: 0 or 1 (whose turn it is)
   */

  function targetIdx(target) {
    return TARGETS.indexOf(target);
  }

  function targetValue(target) {
    return target === 25 ? 25 : target;
  }

  function isClosed(state, player, idx) {
    return state.marks[player][idx] >= MARKS_TO_CLOSE;
  }

  function unclosedTargets(state, player) {
    var result = [];
    for (var i = 0; i < TARGETS.length; i++) {
      if (!isClosed(state, player, i)) result.push(TARGETS[i]);
    }
    return result;
  }

  function scoreableTargets(state, player) {
    var opp = 1 - player;
    var result = [];
    for (var i = 0; i < TARGETS.length; i++) {
      if (isClosed(state, player, i) && !isClosed(state, opp, i)) {
        result.push(TARGETS[i]);
      }
    }
    return result;
  }

  function highestUnblockedValue(state, player) {
    var opp = 1 - player;
    for (var i = 0; i < DESCENDING.length; i++) {
      var idx = targetIdx(DESCENDING[i]);
      if (!isClosed(state, opp, idx)) return targetValue(DESCENDING[i]);
    }
    return 0;
  }

  function bestHitType(target) {
    return target === 25 ? 'double' : 'triple';
  }

  function highestIn(list) {
    var best = null;
    for (var i = 0; i < list.length; i++) {
      if (best === null || targetValue(list[i]) > targetValue(best)) {
        best = list[i];
      }
    }
    return best;
  }

  /** Close highest unclosed by self in descending order */
  function closeHighestUnclosed(state, player) {
    var unclosed = unclosedTargets(state, player);
    for (var i = 0; i < DESCENDING.length; i++) {
      if (unclosed.indexOf(DESCENDING[i]) !== -1) {
        return { target: DESCENDING[i], hitType: bestHitType(DESCENDING[i]) };
      }
    }
    // All closed fallback
    var sc = scoreableTargets(state, player);
    if (sc.length) {
      var best = highestIn(sc);
      return { target: best, hitType: bestHitType(best) };
    }
    return { target: 25, hitType: 'double' };
  }

  /** Cover: close highest opp-unclosed that we haven't closed */
  function coverHighest(state, player) {
    var opp = 1 - player;
    var oppUnclosed = unclosedTargets(state, opp);
    for (var i = 0; i < DESCENDING.length; i++) {
      var t = DESCENDING[i];
      if (oppUnclosed.indexOf(t) !== -1 && !isClosed(state, player, targetIdx(t))) {
        return { target: t, hitType: bestHitType(t) };
      }
    }
    return closeHighestUnclosed(state, player);
  }

  /* ================================================================== */
  /*  Frongello Strategy Engine (S1-S17)                                 */
  /* ================================================================== */

  function frongelloChoose(state, player, params, dartInTurn) {
    var leadMult = params[0];
    var extraDarts = params[1];
    var chase = params[2];

    // S1 mode: always close in order
    if (leadMult === null) {
      return s1Decision(state, player);
    }

    // Chase first (S10-S17)
    if (chase) {
      var chaseTarget = getChaseTarget(state, player);
      if (chaseTarget !== null) {
        return { target: chaseTarget, hitType: bestHitType(chaseTarget), phase: 'CHASE' };
      }
    }

    // Core score/cover decision
    var result = scoreOrCover(state, player, leadMult);

    // Extra darts redirect
    if (extraDarts) {
      var redirected = applyExtraDarts(state, player, result.target);
      if (redirected !== null) {
        redirected.phase = 'REDIRECT';
        return redirected;
      }
    }

    return result;
  }

  function s1Decision(state, player) {
    var unclosed = unclosedTargets(state, player);
    if (unclosed.length) {
      for (var i = 0; i < DESCENDING.length; i++) {
        if (unclosed.indexOf(DESCENDING[i]) !== -1) {
          return { target: DESCENDING[i], hitType: bestHitType(DESCENDING[i]), phase: 'CLOSE' };
        }
      }
    }
    // All closed: if behind, score
    var opp = 1 - player;
    if (state.scores[player] < state.scores[opp]) {
      var sc = scoreableTargets(state, player);
      if (sc.length) {
        for (var j = 0; j < DESCENDING.length; j++) {
          if (sc.indexOf(DESCENDING[j]) !== -1) {
            return { target: DESCENDING[j], hitType: bestHitType(DESCENDING[j]), phase: 'SCORE' };
          }
        }
      }
    }
    return { target: 25, hitType: 'double', phase: 'CLOSE' };
  }

  function scoreOrCover(state, player, leadMult) {
    var opp = 1 - player;
    var lead = state.scores[player] - state.scores[opp];
    var threshold = leadMult * highestUnblockedValue(state, player);

    if (lead <= threshold) {
      // SCORE phase
      var sc = scoreableTargets(state, player);
      if (sc.length) {
        var best = highestIn(sc);
        return { target: best, hitType: bestHitType(best), phase: 'SCORE' };
      }
      var r = closeHighestUnclosed(state, player);
      r.phase = 'CLOSE';
      return r;
    } else {
      // COVER phase
      var c = coverHighest(state, player);
      c.phase = 'COVER';
      return c;
    }
  }

  function getChaseTarget(state, player) {
    var opp = 1 - player;
    for (var i = 0; i < DESCENDING.length; i++) {
      var t = DESCENDING[i];
      var idx = targetIdx(t);
      if (isClosed(state, opp, idx) && !isClosed(state, player, idx)) {
        return t;
      }
    }
    return null;
  }

  function applyExtraDarts(state, player, coreTarget) {
    var idx = targetIdx(coreTarget);
    if (isClosed(state, player, idx)) return null;

    var marksHave = state.marks[player][idx];
    var marksNeeded = MARKS_TO_CLOSE - marksHave;
    if (marksNeeded <= state.dartsRemaining) return null;

    // Can't close this turn — redirect to scoring
    var sc = scoreableTargets(state, player);
    if (sc.length) {
      var best = highestIn(sc);
      return { target: best, hitType: bestHitType(best) };
    }

    // No scoreable — close highest not closed by opponent
    var opp = 1 - player;
    for (var i = 0; i < DESCENDING.length; i++) {
      var t = DESCENDING[i];
      var tIdx = targetIdx(t);
      if (!isClosed(state, opp, tIdx) && !isClosed(state, player, tIdx)) {
        return { target: t, hitType: bestHitType(t) };
      }
    }
    return null;
  }

  /* ================================================================== */
  /*  Experimental Strategies (E1-E4)                                    */
  /* ================================================================== */

  function e1Choose(state, player) {
    var opp = 1 - player;
    var lead = state.scores[player] - state.scores[opp];

    if (lead <= 0) {
      var sc = scoreableTargets(state, player);
      if (sc.length) {
        var best = highestIn(sc);
        return { target: best, hitType: bestHitType(best), phase: 'SCORE' };
      }
      return e1CloseNext(state, player);
    } else {
      // Cover in early-bull order
      var oppUnclosed = unclosedTargets(state, opp);
      for (var i = 0; i < EARLY_BULL_ORDER.length; i++) {
        var t = EARLY_BULL_ORDER[i];
        if (oppUnclosed.indexOf(t) !== -1 && !isClosed(state, player, targetIdx(t))) {
          return { target: t, hitType: bestHitType(t), phase: 'COVER' };
        }
      }
      return e1CloseNext(state, player);
    }
  }

  function e1CloseNext(state, player) {
    var unclosed = unclosedTargets(state, player);
    for (var i = 0; i < EARLY_BULL_ORDER.length; i++) {
      if (unclosed.indexOf(EARLY_BULL_ORDER[i]) !== -1) {
        return { target: EARLY_BULL_ORDER[i], hitType: bestHitType(EARLY_BULL_ORDER[i]), phase: 'CLOSE' };
      }
    }
    var sc = scoreableTargets(state, player);
    if (sc.length) {
      var best = highestIn(sc);
      return { target: best, hitType: bestHitType(best), phase: 'SCORE' };
    }
    return { target: 25, hitType: 'double', phase: 'CLOSE' };
  }

  function e2Choose(state, player) {
    var opp = 1 - player;
    var lead = state.scores[player] - state.scores[opp];

    if (lead <= 0) {
      var sc = scoreableTargets(state, player);
      if (sc.length) {
        var best = highestIn(sc);
        return { target: best, hitType: bestHitType(best), phase: 'SCORE' };
      }
      var r = closeHighestUnclosed(state, player);
      r.phase = 'CLOSE';
      return r;
    } else {
      return honeypotCover(state, player);
    }
  }

  function honeypotCover(state, player) {
    var opp = 1 - player;
    var sc = scoreableTargets(state, player);
    var honeypot = null;
    if (sc.length) {
      for (var i = 0; i < DESCENDING.length; i++) {
        if (sc.indexOf(DESCENDING[i]) !== -1) {
          honeypot = DESCENDING[i];
          break;
        }
      }
    }

    var oppUnclosed = unclosedTargets(state, opp);
    for (var j = 0; j < DESCENDING.length; j++) {
      var t = DESCENDING[j];
      if (t === honeypot) continue;
      if (oppUnclosed.indexOf(t) !== -1 && !isClosed(state, player, targetIdx(t))) {
        return { target: t, hitType: bestHitType(t), phase: 'COVER' };
      }
    }
    var r = closeHighestUnclosed(state, player);
    r.phase = 'CLOSE';
    return r;
  }

  function e3Choose(state, player, dartInTurn) {
    // Simplified: no intra-turn memory in advisor (snapshot state).
    // Use S2 logic — the "greedy" part only matters mid-turn when we
    // can track what just closed.
    var opp = 1 - player;
    var lead = state.scores[player] - state.scores[opp];
    if (lead <= 0) {
      var sc = scoreableTargets(state, player);
      if (sc.length) {
        var best = highestIn(sc);
        return { target: best, hitType: bestHitType(best), phase: 'SCORE' };
      }
      var r = closeHighestUnclosed(state, player);
      r.phase = 'CLOSE';
      return r;
    } else {
      var c = coverHighest(state, player);
      c.phase = 'COVER';
      return c;
    }
  }

  function e4Choose(state, player) {
    var opp = 1 - player;
    var unclosedCount = unclosedTargets(state, player).length;
    var thresholdMult = 3.0 * (unclosedCount / TARGETS.length);
    var lead = state.scores[player] - state.scores[opp];
    var huv = highestUnblockedValue(state, player);
    var threshold = thresholdMult * huv;

    if (lead <= threshold) {
      var sc = scoreableTargets(state, player);
      if (sc.length) {
        var best = highestIn(sc);
        return { target: best, hitType: bestHitType(best), phase: 'SCORE' };
      }
      var r = closeHighestUnclosed(state, player);
      r.phase = 'CLOSE';
      return r;
    } else {
      var c = coverHighest(state, player);
      c.phase = 'COVER';
      return c;
    }
  }

  /* ================================================================== */
  /*  Phase Switch (PS)                                                  */
  /* ================================================================== */

  function psChoose(state, player) {
    var unclosed = unclosedTargets(state, player);
    var unclosedCount = unclosed.length;

    // Compute marks remaining
    var marksRemaining = 0;
    for (var i = 0; i < TARGETS.length; i++) {
      if (state.marks[player][i] < MARKS_TO_CLOSE) {
        marksRemaining += (MARKS_TO_CLOSE - state.marks[player][i]);
      }
    }

    var leadMult;
    if (unclosedCount <= 3 && marksRemaining <= 9) {
      leadMult = 0; // S2 behavior
    } else {
      leadMult = 13; // Aggressive scoring
    }

    return scoreOrCover(state, player, leadMult);
  }

  /* ================================================================== */
  /*  Coordinator                                                        */
  /* ================================================================== */

  function chooseThrow(strategyName, state, player, dartInTurn) {
    if (FRONGELLO_PARAMS[strategyName]) {
      return frongelloChoose(state, player, FRONGELLO_PARAMS[strategyName], dartInTurn);
    }
    switch (strategyName) {
      case 'E1': return e1Choose(state, player);
      case 'E2': return e2Choose(state, player);
      case 'E3': return e3Choose(state, player, dartInTurn);
      case 'E4': return e4Choose(state, player);
      case 'PS': return psChoose(state, player);
      default:   return { target: 20, hitType: 'triple', phase: 'CLOSE' };
    }
  }

  function getAllRecommendations(state, player, dartInTurn) {
    var results = [];
    for (var i = 0; i < STRATEGY_NAMES.length; i++) {
      var name = STRATEGY_NAMES[i];
      var rec = chooseThrow(name, state, player, dartInTurn);
      results.push({
        strategy: name,
        target: rec.target,
        hitType: rec.hitType,
        phase: rec.phase || 'CLOSE',
        description: describeThrow(rec.target, rec.hitType, rec.phase, state, player)
      });
    }
    return results;
  }

  function getConsensus(recommendations) {
    var groups = {};
    for (var i = 0; i < recommendations.length; i++) {
      var r = recommendations[i];
      var key = r.target + ':' + r.hitType;
      if (!groups[key]) {
        groups[key] = { target: r.target, hitType: r.hitType, strategies: [], count: 0 };
      }
      groups[key].strategies.push(r.strategy);
      groups[key].count++;
    }

    // Sort by count descending
    var sorted = [];
    for (var k in groups) {
      if (groups.hasOwnProperty(k)) sorted.push(groups[k]);
    }
    sorted.sort(function (a, b) { return b.count - a.count; });
    return sorted;
  }

  function formatTarget(target) {
    return target === 25 ? 'Bull' : String(target);
  }

  function formatThrow(target, hitType) {
    if (target === 25) {
      return hitType === 'double' ? 'Double Bull' : 'Single Bull';
    }
    return hitType.charAt(0).toUpperCase() + hitType.slice(1) + ' ' + target;
  }

  function describeThrow(target, hitType, phase, state, player) {
    var opp = 1 - player;
    var idx = targetIdx(target);
    var label = formatTarget(target);

    switch (phase) {
      case 'SCORE':
        return 'Scoring on ' + label + ' (closed by you, open for opponent)';
      case 'COVER':
        return 'Covering ' + label + ' (blocking opponent\'s scoring)';
      case 'CHASE':
        return 'Chasing ' + label + ' (opponent has it closed)';
      case 'CLOSE':
        return 'Closing ' + label + ' (working toward 3 marks)';
      case 'REDIRECT':
        return 'Redirecting to score on ' + label + ' (can\'t close current target this turn)';
      default:
        return 'Aiming at ' + label;
    }
  }

  /* ================================================================== */
  /*  Win Probability Data                                               */
  /* ================================================================== */

  var winDataCache = {};

  var bullLookupData = null;
  var bullLookupLoading = false;
  var bullLookupCallbacks = [];

  function loadBullLookup(callback) {
    if (bullLookupData) {
      callback(bullLookupData);
      return;
    }
    bullLookupCallbacks.push(callback);
    if (bullLookupLoading) return;
    bullLookupLoading = true;
    fetch('data/bull_lookup.json')
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (data) {
        bullLookupData = data;
        bullLookupLoading = false;
        var cbs = bullLookupCallbacks.slice();
        bullLookupCallbacks = [];
        for (var i = 0; i < cbs.length; i++) cbs[i](data);
      })
      .catch(function () {
        bullLookupLoading = false;
        var cbs = bullLookupCallbacks.slice();
        bullLookupCallbacks = [];
        for (var i = 0; i < cbs.length; i++) cbs[i](null);
      });
  }

  var EXPECTED_SINGLE_RATE = {
    '0.8': 0.178, '1.0': 0.222, '1.2': 0.267, '1.5': 0.333,
    '2.0': 0.444, '2.5': 0.556, '3.0': 0.667, '3.6': 0.800,
    '4.0': 0.889, '4.9': 0.900, '5.6': 0.960
  };

  var BULL_LEVELS = [0.25, 0.5, 0.75, 1.0];

  function snapToNearest(value) {
    var best = BULL_LEVELS[0];
    var bestDist = Math.abs(value - best);
    for (var i = 1; i < BULL_LEVELS.length; i++) {
      var dist = Math.abs(value - BULL_LEVELS[i]);
      if (dist < bestDist) { best = BULL_LEVELS[i]; bestDist = dist; }
    }
    return best;
  }

  function getBullSelection(playerIdx) {
    var pctInput = document.getElementById('bull-pct-' + playerIdx);
    var displayEl = document.getElementById('bull-mult-display-' + playerIdx);
    var mprEl = document.querySelector('.advisor-mpr-toggle[data-player="' + playerIdx + '"] .toggle-btn.active');
    var mpr = mprEl ? mprEl.getAttribute('data-mpr') : '2.0';

    if (!pctInput || pctInput.value === '') {
      if (displayEl) displayEl.textContent = 'Using default: 0.75x multiplier.';
      return '0.75';
    }

    var pct = parseFloat(pctInput.value);
    if (isNaN(pct) || pct < 0) {
      if (displayEl) displayEl.textContent = 'Using default: 0.75x multiplier.';
      return '0.75';
    }

    var expectedRate = EXPECTED_SINGLE_RATE[mpr] || 0.667;
    var rawMult = (pct / 100) / expectedRate;
    var capped = Math.min(rawMult, 1.0);
    var snapped = snapToNearest(capped);

    if (displayEl) {
      displayEl.textContent = 'Using ' + snapped.toFixed(2) + 'x bull multiplier';
    }

    var s = String(snapped);
    if (s.indexOf('.') === -1) s += '.0';
    return s;
  }

  function lookupBullData(p1Mpr, p2Mpr, p1Bull, p2Bull) {
    if (!bullLookupData || !bullLookupData.lookup) return null;
    var key = p1Mpr + '|' + p2Mpr + '|' + p1Bull + '|' + p2Bull;
    return bullLookupData.lookup[key] || null;
  }

  function loadWinData(mpr, callback) {
    var key = String(mpr);
    if (winDataCache[key]) {
      callback(winDataCache[key]);
      return;
    }
    var path = 'data/tournament_mpr_' + key + '.json';
    fetch(path)
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (data) {
        if (data) {
          winDataCache[key] = data;
          callback(data);
        } else {
          callback(null);
        }
      })
      .catch(function () { callback(null); });
  }

  /* ================================================================== */
  /*  UI Rendering                                                       */
  /* ================================================================== */

  var currentState = null;

  function getStateFromUI() {
    var marks = [[], []];
    for (var p = 0; p < 2; p++) {
      for (var i = 0; i < TARGETS.length; i++) {
        var cell = document.getElementById('mark-' + p + '-' + i);
        marks[p].push(cell ? parseInt(cell.getAttribute('data-marks'), 10) || 0 : 0);
      }
    }

    var s0 = parseInt(document.getElementById('score-0').value, 10) || 0;
    var s1 = parseInt(document.getElementById('score-1').value, 10) || 0;

    var dartsEl = document.querySelector('.advisor-darts-toggle .toggle-btn.active');
    var darts = dartsEl ? parseInt(dartsEl.getAttribute('data-darts'), 10) : 3;

    var turnEl = document.querySelector('input[name="turn"]:checked');
    var turn = turnEl ? parseInt(turnEl.value, 10) : 0;

    return {
      marks: marks,
      scores: [s0, s1],
      dartsRemaining: darts,
      currentPlayer: turn
    };
  }

  function updateAdvisor() {
    var state = getStateFromUI();
    currentState = state;
    var player = state.currentPlayer;
    var dartInTurn = 4 - state.dartsRemaining; // 1st, 2nd, or 3rd dart

    var recs = getAllRecommendations(state, player, dartInTurn);
    var consensus = getConsensus(recs);

    renderConsensus(consensus, recs.length);
    renderRecommendations(recs, state, player);
    renderWinProbability(state, player);
    highlightScoreable(state);
  }

  function renderConsensus(consensus, total) {
    var el = document.getElementById('advisor-consensus');
    if (!el) return;

    if (!consensus.length) {
      el.innerHTML = '<p>Enter a game state to see recommendations.</p>';
      return;
    }

    var top = consensus[0];
    var throwStr = formatThrow(top.target, top.hitType);
    var pct = Math.round((top.count / total) * 100);
    var topStrats = top.strategies.slice(0, 5).join(', ');

    var html = '<p class="advisor-consensus-throw">' + throwStr + '</p>';
    html += '<p class="advisor-consensus-detail">';
    html += '<strong>' + top.count + ' of ' + total + '</strong> strategies agree (' + pct + '%)';
    html += '<br>Including: ' + topStrats;
    if (top.strategies.length > 5) html += ' + ' + (top.strategies.length - 5) + ' more';
    html += '</p>';

    if (consensus.length > 1) {
      var second = consensus[1];
      var s2Throw = formatThrow(second.target, second.hitType);
      html += '<p class="advisor-consensus-alt">Runner-up: <strong>' + s2Throw + '</strong> (' + second.count + ' strategies)</p>';
    }

    el.innerHTML = html;
  }

  function renderRecommendations(recs, state, player) {
    var container = document.getElementById('advisor-recs');
    if (!container) return;

    // Get current MPR for ranking
    var mprEl = document.querySelector('.advisor-mpr-toggle[data-player="' + player + '"] .toggle-btn.active');
    var mpr = mprEl ? mprEl.getAttribute('data-mpr') : '2.0';
    var data = winDataCache[mpr];

    // Rank strategies by tournament performance if data available
    var ranked = recs.slice();
    var rankMap = {};
    if (data && data.rankings) {
      for (var i = 0; i < data.rankings.length; i++) {
        rankMap[data.rankings[i].name] = i;
      }
      ranked.sort(function (a, b) {
        var ra = rankMap[a.strategy] !== undefined ? rankMap[a.strategy] : 999;
        var rb = rankMap[b.strategy] !== undefined ? rankMap[b.strategy] : 999;
        return ra - rb;
      });
    }

    // Bull-aware ranking override
    var bull0 = getBullSelection(0);
    var bull1 = getBullSelection(1);
    var p0MprVal = document.querySelector('.advisor-mpr-toggle[data-player="0"] .toggle-btn.active');
    var p1MprVal = document.querySelector('.advisor-mpr-toggle[data-player="1"] .toggle-btn.active');
    var mpr0Val = p0MprVal ? p0MprVal.getAttribute('data-mpr') : '2.0';
    var mpr1Val = p1MprVal ? p1MprVal.getAttribute('data-mpr') : '2.0';

    if (bullLookupData) {
      var bullEntry = lookupBullData(mpr0Val, mpr1Val, bull0, bull1);
      if (bullEntry) {
        var bullBest = player === 0 ? bullEntry.p1_best : bullEntry.p2_best;
        if (bullBest) {
          var bullRankMap = {};
          for (var bi = 0; bi < bullBest.length; bi++) {
            bullRankMap[bullBest[bi].name] = bi;
          }
          ranked.sort(function (a, b) {
            var ra = bullRankMap[a.strategy] !== undefined ? bullRankMap[a.strategy] : 100 + (rankMap[a.strategy] || 999);
            var rb = bullRankMap[b.strategy] !== undefined ? bullRankMap[b.strategy] : 100 + (rankMap[b.strategy] || 999);
            return ra - rb;
          });
        }
      }
    }

    var showAll = container.getAttribute('data-show-all') === 'true';
    var limit = showAll ? ranked.length : 5;

    var html = '<table class="advisor-rec-table">';
    html += '<thead><tr><th>Strategy</th><th>Recommendation</th><th>Phase</th><th>Reason</th></tr></thead>';
    html += '<tbody>';

    for (var j = 0; j < Math.min(limit, ranked.length); j++) {
      var r = ranked[j];
      var phaseClass = 'advisor-phase-' + r.phase.toLowerCase();
      html += '<tr>';
      html += '<td class="text-accent">' + r.strategy + '</td>';
      html += '<td><strong>' + formatThrow(r.target, r.hitType) + '</strong></td>';
      html += '<td><span class="advisor-phase-badge ' + phaseClass + '">' + r.phase + '</span></td>';
      html += '<td class="text-small">' + r.description + '</td>';
      html += '</tr>';
    }

    html += '</tbody></table>';

    if (ranked.length > 5) {
      html += '<button class="advisor-expand-link" onclick="window._advisorToggleAll()">';
      html += showAll ? 'Show top 5' : 'Show all ' + ranked.length;
      html += '</button>';
    }

    container.innerHTML = html;
  }

  window._advisorToggleAll = function () {
    var container = document.getElementById('advisor-recs');
    if (!container) return;
    var isAll = container.getAttribute('data-show-all') === 'true';
    container.setAttribute('data-show-all', isAll ? 'false' : 'true');
    updateAdvisor();
  };

  function renderWinProbability(state, player) {
    var panel = document.getElementById('advisor-win-panel');
    if (!panel) return;

    var p0Mpr = document.querySelector('.advisor-mpr-toggle[data-player="0"] .toggle-btn.active');
    var p1Mpr = document.querySelector('.advisor-mpr-toggle[data-player="1"] .toggle-btn.active');
    var mpr0 = p0Mpr ? p0Mpr.getAttribute('data-mpr') : '2.0';
    var mpr1 = p1Mpr ? p1Mpr.getAttribute('data-mpr') : '2.0';
    var bull0 = getBullSelection(0);
    var bull1 = getBullSelection(1);

    loadBullLookup(function (data) {
      if (!data) {
        // Fall back to legacy equal-skill display
        if (mpr0 === mpr1) {
          var wd = winDataCache[mpr0];
          if (wd) { renderWinPanel(wd, panel); return; }
          loadWinData(mpr0, function (d) {
            if (d) renderWinPanel(d, panel);
            else panel.innerHTML = '<p class="text-muted">Loading...</p>';
          });
        } else {
          panel.innerHTML = '<p class="text-muted">Loading bull data...</p>';
        }
        return;
      }

      var entry = lookupBullData(mpr0, mpr1, bull0, bull1);
      if (!entry) {
        panel.innerHTML = '<p class="text-muted">No data for this exact combination. Try adjusting MPR or bull levels.</p>';
        return;
      }

      renderBullWinPanel(entry, panel, player, mpr0, mpr1, bull0, bull1);
    });
  }

  function renderWinPanel(data, panel) {
    if (!data.rankings || !data.rankings.length) {
      panel.innerHTML = '<p class="text-muted">No ranking data available.</p>';
      return;
    }

    var top = data.rankings[0];
    var second = data.rankings.length > 1 ? data.rankings[1] : null;
    var third = data.rankings.length > 2 ? data.rankings[2] : null;

    var html = '<div class="advisor-win-grid">';
    html += '<div class="advisor-win-item">';
    html += '<span class="advisor-win-label">Best Strategy</span>';
    html += '<span class="advisor-win-value text-accent">' + top.name + '</span>';
    html += '<span class="advisor-win-sub">Avg win rate: ' + top.avg.toFixed(1) + '%</span>';
    html += '</div>';

    if (second) {
      html += '<div class="advisor-win-item">';
      html += '<span class="advisor-win-label">2nd Best</span>';
      html += '<span class="advisor-win-value">' + second.name + '</span>';
      html += '<span class="advisor-win-sub">Avg win rate: ' + second.avg.toFixed(1) + '%</span>';
      html += '</div>';
    }

    if (third) {
      html += '<div class="advisor-win-item">';
      html += '<span class="advisor-win-label">3rd Best</span>';
      html += '<span class="advisor-win-value">' + third.name + '</span>';
      html += '<span class="advisor-win-sub">Avg win rate: ' + third.avg.toFixed(1) + '%</span>';
      html += '</div>';
    }

    // Head-to-head between top two
    if (second && data.strategies && data.matrix) {
      var topIdx = data.strategies.indexOf(top.name);
      var secIdx = data.strategies.indexOf(second.name);
      if (topIdx !== -1 && secIdx !== -1) {
        var h2h = data.matrix[topIdx][secIdx];
        html += '<div class="advisor-win-item advisor-win-h2h">';
        html += '<span class="advisor-win-label">' + top.name + ' vs ' + second.name + '</span>';
        html += '<span class="advisor-win-value">' + h2h.toFixed(1) + '%</span>';
        html += '<span class="advisor-win-sub">Head-to-head win rate</span>';
        html += '</div>';
      }
    }

    html += '</div>';
    html += '<p class="text-small text-muted" style="margin-top:var(--space-sm)">Based on ' + (data.games_per_matchup || 20000).toLocaleString() + ' games per matchup at MPR ' + (data.mpr_empirical || data.mpr_target) + '</p>';

    panel.innerHTML = html;
  }

  function renderBullWinPanel(entry, panel, player, mpr0, mpr1, bull0, bull1) {
    var myBest = player === 0 ? entry.p1_best : entry.p2_best;
    var oppBest = player === 0 ? entry.p2_best : entry.p1_best;

    var html = '<div class="advisor-win-grid">';

    if (myBest && myBest.length) {
      html += '<div class="advisor-win-item">';
      html += '<span class="advisor-win-label">Your Best Strategy</span>';
      html += '<span class="advisor-win-value text-accent">' + myBest[0].name + '</span>';
      html += '<span class="advisor-win-sub">Avg win rate: ' + myBest[0].avg.toFixed(1) + '%</span>';
      html += '</div>';

      if (myBest.length > 1) {
        html += '<div class="advisor-win-item">';
        html += '<span class="advisor-win-label">2nd Best</span>';
        html += '<span class="advisor-win-value">' + myBest[1].name + '</span>';
        html += '<span class="advisor-win-sub">' + myBest[1].avg.toFixed(1) + '%</span>';
        html += '</div>';
      }

      if (myBest.length > 2) {
        html += '<div class="advisor-win-item">';
        html += '<span class="advisor-win-label">3rd Best</span>';
        html += '<span class="advisor-win-value">' + myBest[2].name + '</span>';
        html += '<span class="advisor-win-sub">' + myBest[2].avg.toFixed(1) + '%</span>';
        html += '</div>';
      }
    }

    if (oppBest && oppBest.length) {
      html += '<div class="advisor-win-item advisor-win-h2h">';
      html += '<span class="advisor-win-label">Opponent\'s Best</span>';
      html += '<span class="advisor-win-value">' + oppBest[0].name + '</span>';
      html += '<span class="advisor-win-sub">' + oppBest[0].avg.toFixed(1) + '% (their perspective)</span>';
      html += '</div>';
    }

    html += '</div>';

    var bullLabel0 = parseFloat(bull0).toFixed(2) + 'x';
    var bullLabel1 = parseFloat(bull1).toFixed(2) + 'x';
    html += '<p class="text-small text-muted" style="margin-top:var(--space-sm)">';
    html += 'Based on 20,000 games per matchup. ';
    html += 'P1: MPR ' + mpr0 + ', bull ' + bullLabel0 + ' | ';
    html += 'P2: MPR ' + mpr1 + ', bull ' + bullLabel1;
    html += '</p>';

    panel.innerHTML = html;
  }

  function highlightScoreable(state) {
    // Remove existing highlights
    var all = document.querySelectorAll('.advisor-mark');
    for (var i = 0; i < all.length; i++) {
      all[i].classList.remove('advisor-scoreable-p0', 'advisor-scoreable-p1');
    }

    // Highlight scoreable targets for each player
    for (var p = 0; p < 2; p++) {
      var sc = scoreableTargets(state, p);
      for (var j = 0; j < sc.length; j++) {
        var idx = targetIdx(sc[j]);
        var cell = document.getElementById('mark-' + p + '-' + idx);
        if (cell) cell.classList.add('advisor-scoreable-p' + p);
      }
    }
  }

  /* ================================================================== */
  /*  Board Interaction                                                  */
  /* ================================================================== */

  var MARK_SYMBOLS = ['', '/', 'X', '\u2297'];  // empty, /, X, circled

  function cycleMark(cell) {
    var current = parseInt(cell.getAttribute('data-marks'), 10) || 0;
    var next = (current + 1) % 4;
    cell.setAttribute('data-marks', next);
    cell.textContent = MARK_SYMBOLS[next];
    cell.className = 'advisor-mark advisor-mark-' + next;
    updateAdvisor();
  }

  function initBoard() {
    var marks = document.querySelectorAll('.advisor-mark');
    for (var i = 0; i < marks.length; i++) {
      marks[i].addEventListener('click', function () { cycleMark(this); });
    }

    // Score inputs
    var scoreInputs = document.querySelectorAll('.advisor-score-input');
    for (var j = 0; j < scoreInputs.length; j++) {
      scoreInputs[j].addEventListener('input', updateAdvisor);
    }

    // Turn radio buttons
    var turnRadios = document.querySelectorAll('input[name="turn"]');
    for (var k = 0; k < turnRadios.length; k++) {
      turnRadios[k].addEventListener('change', updateAdvisor);
    }

    // Darts remaining toggle
    var dartsBtns = document.querySelectorAll('.advisor-darts-toggle .toggle-btn');
    for (var d = 0; d < dartsBtns.length; d++) {
      dartsBtns[d].addEventListener('click', function () {
        var siblings = this.parentElement.querySelectorAll('.toggle-btn');
        for (var s = 0; s < siblings.length; s++) siblings[s].classList.remove('active');
        this.classList.add('active');
        updateAdvisor();
      });
    }

    // MPR toggles
    var mprBtns = document.querySelectorAll('.advisor-mpr-toggle .toggle-btn');
    for (var m = 0; m < mprBtns.length; m++) {
      mprBtns[m].addEventListener('click', function () {
        var siblings = this.parentElement.querySelectorAll('.toggle-btn');
        for (var s2 = 0; s2 < siblings.length; s2++) siblings[s2].classList.remove('active');
        this.classList.add('active');
        // Preload data for selected MPR
        var mpr = this.getAttribute('data-mpr');
        loadWinData(mpr, function () { updateAdvisor(); });
      });
    }

    // Reset button
    var resetBtn = document.getElementById('advisor-reset');
    if (resetBtn) {
      resetBtn.addEventListener('click', resetBoard);
    }

    // Bull hit rate inputs
    var bullPctInputs = document.querySelectorAll('.advisor-bull-pct-input');
    for (var bt = 0; bt < bullPctInputs.length; bt++) {
      bullPctInputs[bt].addEventListener('input', updateAdvisor);
    }

    // Preload bull lookup data
    loadBullLookup(function () { updateAdvisor(); });

    // Preload default MPR data
    loadWinData('2.0', function () { updateAdvisor(); });
  }

  function resetBoard() {
    var marks = document.querySelectorAll('.advisor-mark');
    for (var i = 0; i < marks.length; i++) {
      marks[i].setAttribute('data-marks', '0');
      marks[i].textContent = '';
      marks[i].className = 'advisor-mark advisor-mark-0';
    }

    document.getElementById('score-0').value = '0';
    document.getElementById('score-1').value = '0';

    var turnRadio = document.getElementById('turn-0');
    if (turnRadio) turnRadio.checked = true;

    var dartsBtns = document.querySelectorAll('.advisor-darts-toggle .toggle-btn');
    for (var d = 0; d < dartsBtns.length; d++) {
      dartsBtns[d].classList.remove('active');
      if (dartsBtns[d].getAttribute('data-darts') === '3') dartsBtns[d].classList.add('active');
    }

    updateAdvisor();
  }

  /* ================================================================== */
  /*  Init                                                               */
  /* ================================================================== */

  function init() {
    var board = document.querySelector('.advisor-board');
    if (!board) return; // Not on advisor page

    initBoard();
    updateAdvisor();
  }

  document.addEventListener('DOMContentLoaded', init);

})();
