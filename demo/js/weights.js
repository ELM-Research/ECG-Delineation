/* Boundary-aware MT weight formulas. Port of demo/weights_numpy.py. */
(function (root) {
  "use strict";

  function maxPool1d(edge, band) {
    if (band <= 0) return edge.slice();
    var k = 2 * band + 1;
    var n = edge.length;
    var padded = new Array(n + 2 * band);
    var i;
    for (i = 0; i < padded.length; i++) padded[i] = 0;
    for (i = 0; i < n; i++) padded[i + band] = edge[i];
    var out = new Array(n);
    var t, j, m;
    for (t = 0; t < n; t++) {
      m = -Infinity;
      for (j = 0; j < k; j++) {
        if (padded[t + j] > m) m = padded[t + j];
      }
      out[t] = m;
    }
    return out;
  }

  function softBoundaryWeight(prob, band) {
    band = band === undefined ? 4 : band;
    var C = prob.length;
    var T = prob[0].length;
    var edge = new Array(T);
    edge[0] = 0;
    var t, c, s;
    for (t = 0; t < T - 1; t++) {
      s = 0;
      for (c = 0; c < C; c++) s += Math.abs(prob[c][t + 1] - prob[c][t]);
      edge[t + 1] = 0.5 * s;
    }
    var pooled = maxPool1d(edge, band);
    var denom = 1e-6;
    for (t = 0; t < T; t++) if (pooled[t] > denom) denom = pooled[t];
    var out = new Array(T);
    for (t = 0; t < T; t++) {
      var v = pooled[t] / denom;
      out[t] = v < 0 ? 0 : v > 1 ? 1 : v;
    }
    return out;
  }

  function mean(arr) {
    var s = 0;
    for (var i = 0; i < arr.length; i++) s += arr[i];
    return s / arr.length;
  }

  function regionBoundaryWeights(wB, conf, opts) {
    opts = opts || {};
    var confThresh = opts.confThresh === undefined ? 0.8 : opts.confThresh;
    var sampleConfThresh =
      opts.sampleConfThresh === undefined ? 0.5 : opts.sampleConfThresh;
    var keep =
      opts.sampleKeep === undefined
        ? mean(conf) >= sampleConfThresh
        : Boolean(opts.sampleKeep);
    var keepF = keep ? 1 : 0;
    var n = wB.length;
    var wRegion = new Array(n);
    var wBoundary = new Array(n);
    for (var t = 0; t < n; t++) {
      wRegion[t] = (1 - wB[t]) * (conf[t] >= confThresh ? 1 : 0) * keepF;
      wBoundary[t] = wB[t] * (0.5 + 0.5 * conf[t]) * keepF;
    }
    return { wRegion: wRegion, wBoundary: wBoundary, sampleKeep: keep };
  }

  function ceTeacher(pStudent, pTeacher) {
    var C = pTeacher.length;
    var T = pTeacher[0].length;
    var out = new Array(T);
    for (var t = 0; t < T; t++) {
      var s = 0;
      for (var c = 0; c < C; c++) {
        var ps = pStudent[c][t];
        if (ps < 1e-8) ps = 1e-8;
        s -= pTeacher[c][t] * Math.log(ps);
      }
      out[t] = s;
    }
    return out;
  }

  function vanillaMtUniform(length) {
    var n = length;
    var out = new Array(n);
    var v = n ? 1 / n : 0;
    for (var i = 0; i < n; i++) out[i] = v;
    return out;
  }

  function weightedMean(values, weights) {
    var num = 0;
    var den = 0;
    for (var i = 0; i < values.length; i++) {
      num += values[i] * weights[i];
      den += weights[i];
    }
    if (den < 1) den = 1;
    return num / den;
  }

  function argmax(prob) {
    var C = prob.length;
    var T = prob[0].length;
    var out = new Array(T);
    for (var t = 0; t < T; t++) {
      var best = 0;
      var v = prob[0][t];
      for (var c = 1; c < C; c++) {
        if (prob[c][t] > v) {
          v = prob[c][t];
          best = c;
        }
      }
      out[t] = best;
    }
    return out;
  }

  function mul(a, b) {
    var out = new Array(a.length);
    for (var i = 0; i < a.length; i++) out[i] = a[i] * b[i];
    return out;
  }

  root.MTWeights = {
    softBoundaryWeight: softBoundaryWeight,
    regionBoundaryWeights: regionBoundaryWeights,
    ceTeacher: ceTeacher,
    vanillaMtUniform: vanillaMtUniform,
    weightedMean: weightedMean,
    argmax: argmax,
    mul: mul,
    mean: mean,
  };
})(window);
