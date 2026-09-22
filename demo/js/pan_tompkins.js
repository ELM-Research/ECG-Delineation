(function (root) {
  "use strict";

  function argmax(arr, a, b, fn) {
    a = Math.max(0, a | 0);
    b = Math.min(arr.length - 1, b | 0);
    var bi = a;
    var bv = -Infinity;
    var i;
    var v;
    var f = fn || Math.abs;
    for (i = a; i <= b; i++) {
      v = f(arr[i]);
      if (v > bv) {
        bv = v;
        bi = i;
      }
    }
    return bi;
  }

  function arrayMax(arr) {
    var m = -Infinity;
    var i;
    for (i = 0; i < arr.length; i++) {
      if (arr[i] > m) m = arr[i];
    }
    return m;
  }

  // Pan–Tompkins R peaks, then fixed P/QRS/T windows. Ported from
  // ECG-Annotation-main/ecg-annotator.html. Works on a 1-D lead.
  function detect(ecg, fs) {
    if (!ecg || !ecg.length) return [];
    var lead = ecg;
    var N = lead.length;
    var ms = function (t) {
      return Math.round(t * fs);
    };
    var d = new Float64Array(N);
    var i;
    for (i = 2; i < N - 2; i++) {
      d[i] = (-lead[i - 2] - 2 * lead[i - 1] + 2 * lead[i + 1] + lead[i + 2]) / 8;
    }
    var w = Math.max(1, ms(0.15));
    var mwi = new Float64Array(N);
    var acc = 0;
    for (i = 0; i < N; i++) {
      acc += d[i] * d[i];
      if (i >= w) acc -= d[i - w] * d[i - w];
      mwi[i] = acc / w;
    }
    var thr = 0.3 * arrayMax(mwi);
    var refr = ms(0.2);
    var R = [];
    for (i = 1; i < N - 1; i++) {
      if (
        mwi[i] > thr &&
        mwi[i] >= mwi[i - 1] &&
        mwi[i] > mwi[i + 1] &&
        (!R.length || i - R[R.length - 1] > refr)
      ) {
        R.push(argmax(lead, i - ms(0.13), i + ms(0.02)));
      }
    }
    var out = [];
    var clip = function (x) {
      return Math.max(0, Math.min(N - 1, x));
    };
    var add = function (wave, c, half) {
      out.push({ wave: wave, start: clip(c - half), end: clip(c + half) });
    };
    var r;
    for (i = 0; i < R.length; i++) {
      r = R[i];
      add("QRS", r, ms(0.045));
      if (r - ms(0.08) > 0) {
        add("P", argmax(lead, r - ms(0.25), r - ms(0.08)), ms(0.04));
      }
      if (r + ms(0.36) < N) {
        add("T", argmax(lead, r + ms(0.10), r + ms(0.36)), ms(0.06));
      }
    }
    return out;
  }

  root.PanTompkins = { detect: detect };
})(window);
