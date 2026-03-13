/**
 * Bull-Aware Strategy Advisor — Interactive lookup tool
 * Loads bull_lookup.json and renders personalized recommendations.
 */
(function () {
  'use strict';

  var lookupData = null;

  var STRATEGY_DESCRIPTIONS = {
    S1:  'Pure Cover: close numbers sequentially, never score early',
    S2:  'Score Then Cover: score until ahead, then close',
    S3:  'Threshold 3x: score until lead exceeds 3x highest open value',
    S4:  'Threshold 6x: score until lead exceeds 6x highest open value',
    S5:  'Threshold 9x: score until lead exceeds 9x highest open value',
    S6:  'Efficient Scoring: like S2 but redirects unused darts to score mid-turn',
    S7:  'Threshold 3x + Extra Darts: moderate threshold with mid-turn redirection',
    S8:  'Threshold 6x + Extra Darts: high threshold with mid-turn redirection',
    S9:  'Threshold 9x + Extra Darts: maximum scoring aggression with redirection',
    S10: 'Chase: reactively close whatever opponent has closed, then score',
    S11: 'Chase + Threshold 3x: chase first, then moderate score/cover decision',
    S12: 'Chase + Threshold 6x: chase first, then high score/cover threshold',
    S13: 'Chase + Threshold 9x: chase first, then maximum scoring threshold',
    S14: 'Chase + Extra Darts: chase first, then score with mid-turn redirection',
    S15: 'Chase + 3x + Extra Darts: all three mechanics combined (moderate)',
    S16: 'Chase + 6x + Extra Darts: all three mechanics combined (high)',
    S17: 'Chase + 9x + Extra Darts: all three mechanics combined (maximum)',
    E1:  'Early Bull: close bull 5th (after 20-19-18-17), before 16-15',
    E2:  'Honeypot Cover: standard closing but leaves one target as scoring trap',
    E3:  'Greedy Close-and-Score: score on freshly closed targets with remaining darts',
    E4:  'Adaptive Threshold: scoring aggression scales with game progress',
    PS:  'Phase Switch: extreme scoring early, locks into closing mode late'
  };

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

  function getBullForPlayer(playerIdx, mpr) {
    var pctInput = document.getElementById('bull-pct-' + playerIdx);
    var displayEl = document.getElementById('bull-mult-display-' + playerIdx);

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

  function getInputs() {
    var myMprEl = document.querySelector('.bull-mpr-toggle[data-player="0"] .toggle-btn.active');
    var oppMprEl = document.querySelector('.bull-mpr-toggle[data-player="1"] .toggle-btn.active');
    var myMpr = myMprEl ? myMprEl.getAttribute('data-mpr') : '3.0';
    var oppMpr = oppMprEl ? oppMprEl.getAttribute('data-mpr') : '3.0';

    return {
      myMpr: myMpr,
      myBull: getBullForPlayer(0, myMpr),
      oppMpr: oppMpr,
      oppBull: getBullForPlayer(1, oppMpr)
    };
  }

  function update() {
    var panel = document.getElementById('bull-advisor-result');
    if (!panel) return;

    if (!lookupData) {
      panel.innerHTML = '<p class="text-muted">Loading strategy data...</p>';
      return;
    }

    var inputs = getInputs();
    var key = inputs.myMpr + '|' + inputs.oppMpr + '|' + inputs.myBull + '|' + inputs.oppBull;
    var entry = lookupData.lookup[key];

    if (!entry) {
      panel.innerHTML = '<p class="text-muted">No data for this combination. Try adjusting inputs.</p>';
      return;
    }

    var myH2h = entry.h2h ? entry.h2h.p1 : null;
    var oppH2h = entry.h2h ? entry.h2h.p2 : null;
    var myBest = myH2h || entry.p1_best;
    var oppBest = oppH2h || entry.p2_best;

    var html = '';

    if (myBest && myBest.length) {
      var primary = myBest[0];
      var wrClass = primary.avg >= 50 ? 'bull-wr-favorable' : 'bull-wr-unfavorable';

      html += '<div class="bull-recommendation-primary">';
      html += '<div class="bull-rec-header">';
      html += '<span class="bull-rec-strategy">' + primary.name + '</span>';
      html += '<span class="bull-rec-wr ' + wrClass + '">' + primary.avg.toFixed(1) + '% avg win rate</span>';
      html += '</div>';

      var desc = STRATEGY_DESCRIPTIONS[primary.name] || '';
      if (desc) {
        html += '<p class="bull-rec-description">' + desc + '</p>';
      }

      // Show best-response info if available
      if (primary.counter) {
        var counterDesc = STRATEGY_DESCRIPTIONS[primary.counter] || '';
        html += '<p class="bull-rec-counter">If opponent plays <strong>' + primary.counter + '</strong>';
        if (counterDesc) html += ' (' + counterDesc.split(':')[0].toLowerCase() + ')';
        html += ', your win rate drops to <strong>' + primary.vs_counter.toFixed(1) + '%</strong>.</p>';
      }
      html += '</div>';

      // Strategy comparison table with head-to-head data
      html += '<div class="bull-alternatives">';
      html += '<h4>Strategy Comparison</h4>';
      html += '<table class="methodology-table">';
      if (myBest[0].counter) {
        html += '<thead><tr><th>Your Strategy</th><th>Avg WR</th><th>vs Their Counter</th><th>Counter</th></tr></thead>';
      } else {
        html += '<thead><tr><th>Your Strategy</th><th>Avg WR</th><th>Description</th></tr></thead>';
      }
      html += '<tbody>';
      for (var i = 0; i < myBest.length; i++) {
        var s = myBest[i];
        html += '<tr>';
        html += '<td class="text-accent"><strong>' + s.name + '</strong></td>';
        html += '<td>' + s.avg.toFixed(1) + '%</td>';
        if (s.counter) {
          var vsClass = s.vs_counter >= 50 ? 'bull-wr-favorable' : 'bull-wr-unfavorable';
          html += '<td><span class="' + vsClass + '" style="padding:1px 6px;border-radius:100px;font-size:0.85em">' + s.vs_counter.toFixed(1) + '%</span></td>';
          html += '<td class="text-small">' + s.counter + '</td>';
        } else {
          var altDesc = STRATEGY_DESCRIPTIONS[s.name] || '';
          html += '<td class="text-small">' + altDesc + '</td>';
        }
        html += '</tr>';
      }
      html += '</tbody></table>';
      html += '</div>';

      // Opponent's strategies
      if (oppBest && oppBest.length) {
        html += '<div class="bull-opponent-info">';
        html += '<h4>Opponent\'s Strategies</h4>';
        html += '<table class="methodology-table">';
        if (oppBest[0].counter) {
          html += '<thead><tr><th>Their Strategy</th><th>Their Avg WR</th><th>vs Your Counter</th><th>Your Counter</th></tr></thead>';
        } else {
          html += '<thead><tr><th>Their Strategy</th><th>Their Avg WR</th></tr></thead>';
        }
        html += '<tbody>';
        for (var j = 0; j < oppBest.length; j++) {
          var opp = oppBest[j];
          html += '<tr>';
          html += '<td><strong>' + opp.name + '</strong></td>';
          html += '<td>' + opp.avg.toFixed(1) + '%</td>';
          if (opp.counter) {
            html += '<td>' + opp.vs_counter.toFixed(1) + '%</td>';
            html += '<td class="text-accent text-small">' + opp.counter + '</td>';
          }
          html += '</tr>';
        }
        html += '</tbody></table>';
        html += '</div>';
      }

      var bullLabel = function (v) { return parseFloat(v).toFixed(2) + 'x'; };
      html += '<p class="text-small text-muted mt-sm">';
      html += 'You: MPR ' + inputs.myMpr + ', bull ' + bullLabel(inputs.myBull);
      html += ' vs Opponent: MPR ' + inputs.oppMpr + ', bull ' + bullLabel(inputs.oppBull);
      html += '. Based on 20,000 games per matchup.';
      html += '</p>';
    }

    panel.innerHTML = html;
  }

  function initToggles() {
    var mprBtns = document.querySelectorAll('.bull-mpr-toggle .toggle-btn');
    for (var i = 0; i < mprBtns.length; i++) {
      mprBtns[i].addEventListener('click', function () {
        var siblings = this.parentElement.querySelectorAll('.toggle-btn');
        for (var s = 0; s < siblings.length; s++) siblings[s].classList.remove('active');
        this.classList.add('active');
        update();
      });
    }

    var bullInputs = document.querySelectorAll('.advisor-bull-pct-input');
    for (var j = 0; j < bullInputs.length; j++) {
      bullInputs[j].addEventListener('input', update);
    }
  }

  function init() {
    var guide = document.getElementById('guide-section');
    if (!guide) return;

    initToggles();

    fetch('data/bull_lookup.json')
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (data) {
        lookupData = data;
        update();
      })
      .catch(function (err) {
        console.warn('Failed to load bull_lookup.json:', err);
      });
  }

  document.addEventListener('DOMContentLoaded', init);
})();
