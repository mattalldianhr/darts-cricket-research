/**
 * Darts Cricket Strategy Research — Interactive JS
 * Vanilla JS, no dependencies.
 */

(function () {
  'use strict';

  /* ------------------------------------------------------------------ */
  /*  Constants                                                          */
  /* ------------------------------------------------------------------ */

  var COLOR_RED    = { r: 231, g:  76, b:  60 };   // #e74c3c
  var COLOR_YELLOW = { r: 241, g: 196, b:  15 };   // #f1c40f
  var COLOR_GREEN  = { r:  46, g: 204, b: 113 };   // #2ecc71
  var COLOR_NEUTRAL = '#f0f0eb';

  var MPR_LEVELS = [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.6, 4.0, 4.9, 5.6];

  var dataCache = {};   // keyed by MPR string, stores parsed JSON

  /* ------------------------------------------------------------------ */
  /*  Color helpers                                                      */
  /* ------------------------------------------------------------------ */

  /** Linearly interpolate between two {r,g,b} objects at ratio t in [0,1]. */
  function lerpColor(a, b, t) {
    t = Math.max(0, Math.min(1, t));
    return {
      r: Math.round(a.r + (b.r - a.r) * t),
      g: Math.round(a.g + (b.g - a.g) * t),
      b: Math.round(a.b + (b.b - a.b) * t)
    };
  }

  /** Convert {r,g,b} to "rgb(r,g,b)" string. */
  function rgbStr(c) {
    return 'rgb(' + c.r + ',' + c.g + ',' + c.b + ')';
  }

  /**
   * Perceived luminance (0-255). Values > ~160 get dark text,
   * values <= ~160 get light text.
   */
  function luminance(c) {
    return 0.299 * c.r + 0.587 * c.g + 0.114 * c.b;
  }

  /**
   * Given a win-rate percentage (0-100), return {bg, fg} color strings.
   * Smooth continuous gradient: red (0%) -> yellow (50%) -> green (100%).
   * Exactly 50.0% gets a neutral gray.
   */
  function winRateColors(pct) {
    if (pct === 50.0) {
      return { bg: COLOR_NEUTRAL, fg: '#333' };
    }

    var bg;
    if (pct < 50) {
      // 0 -> red, 50 -> yellow.  Map [0,50] to [0,1].
      var t = pct / 50;
      // Use an eased curve for smoother middle transition
      t = t * t * (3 - 2 * t);   // smoothstep
      bg = lerpColor(COLOR_RED, COLOR_YELLOW, t);
    } else {
      // 50 -> yellow, 100 -> green.  Map [50,100] to [0,1].
      var t2 = (pct - 50) / 50;
      t2 = t2 * t2 * (3 - 2 * t2);
      bg = lerpColor(COLOR_YELLOW, COLOR_GREEN, t2);
    }

    var bgStr = rgbStr(bg);
    var fgStr = luminance(bg) > 160 ? '#222' : '#fff';
    return { bg: bgStr, fg: fgStr };
  }

  /**
   * Rescale a win-rate value so the data's actual spread fills the full
   * color range. Maps symmetrically around 50: the furthest value from 50
   * (whether above or below) maps to 0 or 100, and everything else scales
   * proportionally. This gives vivid red/green contrast even when all
   * values cluster in a narrow band like 42-58%.
   */
  function rescaleWinRate(pct, spread) {
    if (pct === 50.0 || spread <= 0) return pct;
    return 50 + ((pct - 50) / spread) * 50;
  }

  /* ------------------------------------------------------------------ */
  /*  1. Win-rate cell coloring                                          */
  /* ------------------------------------------------------------------ */

  /**
   * Find all .matrix-cell elements, parse their text as a number,
   * and apply background/text color based on win-rate.
   */
  function colorizeMatrixCells() {
    var cells = document.querySelectorAll('.matrix-cell');
    if (!cells.length) return;

    requestAnimationFrame(function () {
      for (var i = 0; i < cells.length; i++) {
        var cell = cells[i];
        var text = cell.textContent.trim();
        var val = parseFloat(text);
        if (isNaN(val)) continue;

        var colors = winRateColors(val);
        cell.style.backgroundColor = colors.bg;
        cell.style.color = colors.fg;
      }
    });
  }

  /* ------------------------------------------------------------------ */
  /*  2. MPR Level Toggle                                                */
  /* ------------------------------------------------------------------ */

  /**
   * Build the tournament matrix <table> from loaded data and inject
   * it into #tournament-matrix.
   */
  function buildMatrix(data) {
    var container = document.getElementById('tournament-matrix');
    if (!container) return;

    var strats = data.strategies;
    var matrix = data.matrix;

    // Compute row averages (exclude diagonal / self-play)
    var rowAvgs = [];
    for (var r = 0; r < strats.length; r++) {
      var sum = 0;
      var count = 0;
      for (var c = 0; c < strats.length; c++) {
        if (r !== c) {
          sum += matrix[r][c];
          count++;
        }
      }
      rowAvgs.push(count > 0 ? sum / count : 50);
    }

    // Find max deviation from 50 across all non-diagonal cells for rescaled coloring
    var matrixSpread = 0;
    for (var r3 = 0; r3 < strats.length; r3++) {
      for (var c3 = 0; c3 < strats.length; c3++) {
        if (r3 !== c3) {
          var dev = Math.abs(matrix[r3][c3] - 50);
          if (dev > matrixSpread) matrixSpread = dev;
        }
      }
    }

    // Find min/max of averages for rescaled coloring
    var avgMin = rowAvgs[0], avgMax = rowAvgs[0];
    for (var a = 1; a < rowAvgs.length; a++) {
      if (rowAvgs[a] < avgMin) avgMin = rowAvgs[a];
      if (rowAvgs[a] > avgMax) avgMax = rowAvgs[a];
    }
    var avgRange = avgMax - avgMin;

    // Build HTML string for performance
    var html = '<table class="matrix-table">';

    // Header row
    html += '<thead><tr><th class="matrix-corner"></th>';
    for (var c2 = 0; c2 < strats.length; c2++) {
      html += '<th class="matrix-header-cell">' + escapeHtml(strats[c2]) + '</th>';
    }
    html += '<th class="matrix-header-cell matrix-avg-header">Avg</th>';
    html += '</tr></thead>';

    // Body rows
    html += '<tbody>';
    for (var r2 = 0; r2 < strats.length; r2++) {
      html += '<tr>';
      html += '<th class="matrix-row-header">' + escapeHtml(strats[r2]) + '</th>';
      for (var j = 0; j < strats.length; j++) {
        var val = matrix[r2][j];
        var cls = 'matrix-cell';
        if (r2 === j) cls += ' matrix-diagonal';
        var cellColors = winRateColors(rescaleWinRate(val, matrixSpread));
        html += '<td class="' + cls + '" data-row="' + r2 + '" data-col="' + j + '"';
        if (r2 !== j) {
          html += ' style="background-color:' + cellColors.bg + ';color:' + cellColors.fg + '"';
        }
        html += '>';
        html += val.toFixed(1);
        html += '</td>';
      }
      // Avg column — rescale to min-max range for vivid color contrast
      var avgScaled = avgRange > 0 ? ((rowAvgs[r2] - avgMin) / avgRange) * 100 : 50;
      var avgColors = winRateColors(avgScaled);
      html += '<td class="matrix-cell matrix-avg-cell" data-row="' + r2 + '"';
      html += ' style="background-color:' + avgColors.bg + ';color:' + avgColors.fg + '">';
      html += rowAvgs[r2].toFixed(1);
      html += '</td>';
      html += '</tr>';
    }
    html += '</tbody></table>';

    container.innerHTML = html;

    // Attach hover delegation on the newly created table
    var table = container.querySelector('.matrix-table');
    if (table) {
      attachMatrixHover(table);
    }

    // Colors are applied inline during build (rescaled), no need for colorizeMatrixCells()
  }

  /**
   * Build the rankings table from loaded data and inject into
   * #rankings-table.
   */
  function buildRankings(data) {
    var container = document.getElementById('rankings-table');
    if (!container) return;

    var rankings = data.rankings;
    if (!rankings || !rankings.length) {
      container.innerHTML = '<p>No ranking data available.</p>';
      return;
    }

    // Find max avg for bar width scaling
    var maxAvg = 0;
    for (var i = 0; i < rankings.length; i++) {
      if (rankings[i].avg > maxAvg) maxAvg = rankings[i].avg;
    }

    var html = '<table class="rankings-table">';
    html += '<thead><tr>';
    html += '<th>Rank</th><th>Strategy</th><th>Avg Win %</th><th></th>';
    html += '</tr></thead><tbody>';

    for (var k = 0; k < rankings.length; k++) {
      var entry = rankings[k];
      var colors = winRateColors(entry.avg);
      var barWidth = maxAvg > 0 ? (entry.avg / maxAvg) * 100 : 0;

      html += '<tr>';
      html += '<td class="rank-num">' + (k + 1) + '</td>';
      html += '<td class="rank-name">' + escapeHtml(entry.name) + '</td>';
      html += '<td class="rank-avg">' + entry.avg.toFixed(1) + '%</td>';
      html += '<td class="rank-bar-cell">';
      html += '<div class="rank-bar" style="width:' + barWidth.toFixed(1) + '%;';
      html += 'background-color:' + colors.bg + ';"></div>';
      html += '</td>';
      html += '</tr>';
    }

    html += '</tbody></table>';
    container.innerHTML = html;
  }

  /**
   * Update stat badges when MPR data loads.
   */
  function updateStats(data) {
    var mappings = [
      ['stat-mpr-target',    data.mpr_target],
      ['stat-mpr-empirical', data.mpr_empirical],
      ['stat-games',         data.games_per_matchup],
      ['stat-avg-turns',     data.avg_turns],
      ['stat-avg-darts',     data.avg_darts]
    ];

    for (var i = 0; i < mappings.length; i++) {
      var el = document.getElementById(mappings[i][0]);
      if (el && mappings[i][1] != null) {
        var val = mappings[i][1];
        // Format numbers nicely
        if (typeof val === 'number') {
          el.textContent = Number.isInteger(val) ? val.toLocaleString() : val.toFixed(2);
        } else {
          el.textContent = val;
        }
      }
    }
  }

  /**
   * Load tournament data for a given MPR level, update matrix + rankings,
   * cache the result.
   */
  function loadMprData(mpr) {
    var key = String(mpr);

    // Update active button state immediately
    var buttons = document.querySelectorAll('.mpr-toggle [data-mpr]');
    for (var i = 0; i < buttons.length; i++) {
      if (buttons[i].getAttribute('data-mpr') === key) {
        buttons[i].classList.add('active');
      } else {
        buttons[i].classList.remove('active');
      }
    }

    // If cached, render from cache
    if (dataCache[key]) {
      renderMprData(dataCache[key]);
      return Promise.resolve(dataCache[key]);
    }

    // Construct path: data/ is symlinked into site/
    var jsonPath = 'data/tournament_mpr_' + key + '.json';

    return fetch(jsonPath)
      .then(function (response) {
        if (!response.ok) {
          throw new Error('HTTP ' + response.status);
        }
        return response.json();
      })
      .then(function (data) {
        dataCache[key] = data;
        renderMprData(data);
        return data;
      })
      .catch(function (err) {
        console.warn('Failed to load MPR ' + key + ' data:', err);
        showMatrixError('Data for MPR ' + key + ' is not available yet.');
      });
  }

  /** Render all elements from loaded MPR data. */
  function renderMprData(data) {
    buildMatrix(data);
    buildRankings(data);
    updateStats(data);
    updateToggleLabel(data);
  }

  /** Update the toggle button label to show the empirical MPR from JSON data. */
  function updateToggleLabel(data) {
    var key = String(data.mpr_target);
    var btn = document.querySelector('.mpr-toggle [data-mpr="' + key + '"]');
    if (btn && data.mpr_empirical != null) {
      btn.textContent = data.mpr_empirical.toFixed(2);
    }
  }

  /** Pre-load all MPR data to populate toggle labels. */
  function preloadToggleLabels() {
    for (var i = 0; i < MPR_LEVELS.length; i++) {
      var mpr = MPR_LEVELS[i];
      var key = String(mpr);
      if (dataCache[key]) {
        updateToggleLabel(dataCache[key]);
        continue;
      }
      var jsonPath = 'data/tournament_mpr_' + key + '.json';
      (function(k) {
        fetch(jsonPath)
          .then(function(r) { return r.ok ? r.json() : null; })
          .then(function(d) {
            if (d) {
              dataCache[k] = d;
              updateToggleLabel(d);
            }
          })
          .catch(function() {});
      })(key);
    }
  }

  /** Show a user-friendly message when data cannot be loaded. */
  function showMatrixError(msg) {
    var container = document.getElementById('tournament-matrix');
    if (container) {
      container.innerHTML = '<p class="data-error">' + escapeHtml(msg) + '</p>';
    }
    var rankings = document.getElementById('rankings-table');
    if (rankings) {
      rankings.innerHTML = '';
    }
  }

  /* ------------------------------------------------------------------ */
  /*  3. Row / column hover highlighting                                 */
  /* ------------------------------------------------------------------ */

  /**
   * Attach mouseover/mouseout event delegation to a matrix table
   * for row+column highlighting.
   */
  function attachMatrixHover(table) {
    table.addEventListener('mouseover', function (e) {
      var td = e.target.closest('.matrix-cell');
      if (!td) return;

      var row = td.getAttribute('data-row');
      var col = td.getAttribute('data-col');
      if (row == null || col == null) return;

      var cells = table.querySelectorAll('.matrix-cell');
      for (var i = 0; i < cells.length; i++) {
        var c = cells[i];
        if (c.getAttribute('data-row') === row) {
          c.classList.add('matrix-hover-row');
        }
        if (c.getAttribute('data-col') === col) {
          c.classList.add('matrix-hover-col');
        }
      }
    });

    table.addEventListener('mouseout', function (e) {
      var td = e.target.closest('.matrix-cell');
      if (!td) return;

      var cells = table.querySelectorAll('.matrix-cell');
      for (var i = 0; i < cells.length; i++) {
        cells[i].classList.remove('matrix-hover-row');
        cells[i].classList.remove('matrix-hover-col');
      }
    });
  }

  /* ------------------------------------------------------------------ */
  /*  4. Mobile hamburger nav                                            */
  /* ------------------------------------------------------------------ */

  function initHamburger() {
    var btn = document.querySelector('.hamburger-btn');
    var navLinks = document.querySelector('.nav-links');
    if (!btn || !navLinks) return;

    btn.addEventListener('click', function () {
      btn.classList.toggle('open');
      navLinks.classList.toggle('open');
    });

    // Close nav when clicking a link (mobile UX)
    var links = navLinks.querySelectorAll('a');
    for (var i = 0; i < links.length; i++) {
      links[i].addEventListener('click', function () {
        btn.classList.remove('open');
        navLinks.classList.remove('open');
      });
    }
  }

  /* ------------------------------------------------------------------ */
  /*  5. Details / summary smooth animation                              */
  /* ------------------------------------------------------------------ */

  function initDetailsAnimation() {
    var detailsAll = document.querySelectorAll('details');
    if (!detailsAll.length) return;

    for (var i = 0; i < detailsAll.length; i++) {
      (function (details) {
        var summary = details.querySelector('summary');
        if (!summary) return;

        // Find the content wrapper (everything after <summary>).
        // We wrap inner content in a div for animation if not already wrapped.
        var content = details.querySelector('.details-content');
        if (!content) {
          content = document.createElement('div');
          content.className = 'details-content';
          // Move all children except <summary> into wrapper
          while (summary.nextSibling) {
            content.appendChild(summary.nextSibling);
          }
          details.appendChild(content);
        }

        content.style.overflow = 'hidden';

        summary.addEventListener('click', function (e) {
          e.preventDefault();

          if (details.open) {
            // Closing: animate height to 0, then remove open
            var startHeight = content.scrollHeight;
            content.style.height = startHeight + 'px';

            requestAnimationFrame(function () {
              content.style.transition = 'height 0.3s ease';
              content.style.height = '0px';
            });

            content.addEventListener('transitionend', function handler() {
              content.removeEventListener('transitionend', handler);
              details.removeAttribute('open');
              content.style.height = '';
              content.style.transition = '';
            });
          } else {
            // Opening: set open, measure, animate from 0
            details.setAttribute('open', '');
            var targetHeight = content.scrollHeight;
            content.style.height = '0px';

            requestAnimationFrame(function () {
              content.style.transition = 'height 0.3s ease';
              content.style.height = targetHeight + 'px';
            });

            content.addEventListener('transitionend', function handler() {
              content.removeEventListener('transitionend', handler);
              content.style.height = '';
              content.style.transition = '';
            });
          }
        });
      })(detailsAll[i]);
    }
  }

  /* ------------------------------------------------------------------ */
  /*  6. Scroll shadow on matrix container                               */
  /* ------------------------------------------------------------------ */

  function initScrollShadow() {
    var containers = document.querySelectorAll('.matrix-scroll-container');
    for (var i = 0; i < containers.length; i++) {
      (function (el) {
        function updateShadow() {
          var scrollLeft = el.scrollLeft;
          var maxScroll = el.scrollWidth - el.clientWidth;

          var hasLeft = scrollLeft > 2;
          var hasRight = scrollLeft < maxScroll - 2;

          el.classList.toggle('shadow-left', hasLeft);
          el.classList.toggle('shadow-right', hasRight);
        }

        el.addEventListener('scroll', updateShadow, { passive: true });
        // Run once on load
        updateShadow();
        // Re-check after images/fonts load
        window.addEventListener('load', updateShadow);
      })(containers[i]);
    }
  }

  /* ------------------------------------------------------------------ */
  /*  Utility                                                            */
  /* ------------------------------------------------------------------ */

  function escapeHtml(str) {
    var div = document.createElement('div');
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
  }

  /* ------------------------------------------------------------------ */
  /*  Init on DOMContentLoaded                                           */
  /* ------------------------------------------------------------------ */

  function init() {
    // Color any statically-rendered matrix cells
    colorizeMatrixCells();

    // Hamburger nav
    initHamburger();

    // Smooth details/summary
    initDetailsAnimation();

    // Scroll shadow
    initScrollShadow();

    // MPR toggle buttons (results page)
    var toggleBar = document.querySelector('.mpr-toggle');
    if (toggleBar) {
      toggleBar.addEventListener('click', function (e) {
        var btn = e.target.closest('[data-mpr]');
        if (!btn) return;
        var mpr = btn.getAttribute('data-mpr');
        loadMprData(mpr);
      });

      // Auto-load the first active button or the highest MPR level
      var activeBtn = toggleBar.querySelector('[data-mpr].active');
      if (activeBtn) {
        loadMprData(activeBtn.getAttribute('data-mpr'));
      } else {
        // Default: load the highest skill level
        var firstBtn = toggleBar.querySelector('[data-mpr]');
        if (firstBtn) {
          loadMprData(firstBtn.getAttribute('data-mpr'));
        }
      }

      // Pre-load all JSON files to populate toggle labels with empirical MPR values
      preloadToggleLabels();
    }

    // Attach hover to any statically-rendered matrix tables
    var existingTables = document.querySelectorAll('.matrix-table');
    for (var i = 0; i < existingTables.length; i++) {
      attachMatrixHover(existingTables[i]);
    }
  }

  document.addEventListener('DOMContentLoaded', init);

  /* ------------------------------------------------------------------ */
  /*  Export public API                                                   */
  /* ------------------------------------------------------------------ */

  window.colorizeMatrixCells = colorizeMatrixCells;
  window.loadMprData         = loadMprData;
  window.buildMatrix         = buildMatrix;
  window.buildRankings       = buildRankings;

})();
