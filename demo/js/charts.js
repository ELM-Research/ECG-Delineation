(function (root) {
  "use strict";

  var COLORS = {
    0: "#E8E8E8",
    1: "#2563EB",
    2: "#EF4444",
    3: "#22C55E",
  };
  var NAMES = ["Background", "P", "QRS", "T"];
  var FONT = { family: "IBM Plex Sans, Helvetica Neue, sans-serif", size: 13 };
  var PLOT_OPTS = {
    responsive: true,
    displaylogo: false,
    displayModeBar: false,
  };

  function timeAxis(n) {
    var x = new Array(n);
    var i;
    for (i = 0; i < n; i++) x[i] = i;
    return x;
  }

  var WAVE_COLORS = {
    P: COLORS[1],
    QRS: COLORS[2],
    T: COLORS[3],
  };

  function classShapes(labels, yref) {
    var shapes = [];
    if (!labels || !labels.length) return shapes;
    var start = 0;
    var cur = labels[0];
    var i;
    var next;
    for (i = 1; i <= labels.length; i++) {
      next = i < labels.length ? labels[i] : -1;
      if (next !== cur) {
        if (cur > 0) {
          shapes.push({
            type: "rect",
            xref: "x",
            yref: yref + " domain",
            x0: start,
            x1: i - 1,
            y0: 0,
            y1: 1,
            fillcolor: COLORS[cur],
            opacity: 0.38,
            line: { width: 0 },
            layer: "below",
          });
        }
        start = i;
        cur = next;
      }
    }
    return shapes;
  }

  function yaxis(domain, title, range) {
    var ax = {
      domain: domain,
      anchor: "x",
      showgrid: true,
      gridcolor: "#f2f2f2",
      zeroline: false,
      title: { text: title, font: { size: 12 } },
      tickfont: { size: 11 },
      fixedrange: true,
    };
    if (range) {
      ax.range = range;
      ax.autorange = false;
    }
    return ax;
  }

  function yRangeCentered(y, pad) {
    var peak = 0;
    var i;
    for (i = 0; i < y.length; i++) {
      var a = Math.abs(y[i]);
      if (a > peak) peak = a;
    }
    if (peak < 1e-6) peak = 1;
    var p = pad == null ? 0.12 : pad;
    var hi = peak * (1 + p);
    return [-hi, hi];
  }

  function keepCount(x, t) {
    if (t == null) return x.length;
    var n = 0;
    while (n < x.length && x[n] <= t) n++;
    return Math.max(1, n);
  }

  function storePlotState(el, traces, shapes, zoomRange) {
    el._mtFullTraces = traces;
    el._mtBaseShapes = shapes.slice();
    el._mtXRange = zoomRange;
  }

  function visibleTraces(el, t) {
    var full = el._mtFullTraces || [];
    if (t == null) return full;
    return full.map(function (tr) {
      var n = keepCount(tr.x, t);
      var copy = {};
      var key;
      for (key in tr) {
        if (Object.prototype.hasOwnProperty.call(tr, key)) copy[key] = tr[key];
      }
      copy.x = tr.x.slice(0, n);
      copy.y = tr.y.slice(0, n);
      return copy;
    });
  }

  function line(x, y, name, yaxisName, color, extra) {
    var trace = {
      x: x,
      y: y,
      name: name,
      type: "scatter",
      mode: "lines",
      xaxis: "x",
      yaxis: yaxisName,
      line: { color: color, width: extra && extra.width ? extra.width : 1.27 },
      hovertemplate: "%{x} · %{y:.3f}<extra>" + name + "</extra>",
      showlegend: false,
    };
    if (extra && extra.dash) trace.line.dash = extra.dash;
    if (extra && extra.opacity !== undefined) trace.opacity = extra.opacity;
    if (extra && extra.fill) {
      trace.fill = extra.fill;
      trace.fillcolor = extra.fillcolor;
    }
    return trace;
  }

  function xaxis(anchor, zoomRange) {
    return {
      title: { text: "", font: { size: 12 } },
      domain: [0, 1],
      anchor: anchor,
      showgrid: true,
      gridcolor: "#f4f4f4",
      range: zoomRange,
      autorange: false,
      fixedrange: true,
    };
  }

  function baseLayout(height, shapes, zoomRange, anchor) {
    return {
      font: FONT,
      margin: { l: 62, r: 18, t: 9, b: 31 },
      height: height,
      paper_bgcolor: "#fff",
      plot_bgcolor: "#fff",
      hovermode: "x unified",
      showlegend: false,
      uirevision: "mt",
      shapes: shapes,
      xaxis: xaxis(anchor, zoomRange),
    };
  }

  function playMask(t, xmax) {
    if (t == null) return null;
    return {
      type: "rect",
      xref: "x",
      yref: "paper",
      x0: Math.max(0, t),
      x1: xmax,
      y0: 0,
      y1: 1,
      fillcolor: "#ffffff",
      line: { width: 0 },
      layer: "above",
    };
  }

  function withPlayMask(shapes, t, xmax) {
    var out = (shapes || []).slice();
    var mask = playMask(t, xmax);
    if (mask) out.push(mask);
    return out;
  }

  function clipShapes(base, t) {
    var src = base || [];
    if (t == null) return src.slice();
    var out = [];
    var i;
    var s;
    var copy;
    var key;
    for (i = 0; i < src.length; i++) {
      s = src[i];
      if (s.xref === "paper") {
        out.push(s);
        continue;
      }
      if (s.x0 > t) continue;
      copy = {};
      for (key in s) {
        if (Object.prototype.hasOwnProperty.call(s, key)) copy[key] = s[key];
      }
      if (copy.x1 > t) copy.x1 = t;
      out.push(copy);
    }
    return out;
  }

  function buildTrain(step, derived, view, zoomRange) {
    var u = step.unlabeled;
    var n = u.ecg_weak.length;
    var x = timeAxis(n);
    var teacherArg = MTWeights.argmax(u.p_teacher);
    var studentArg = MTWeights.argmax(u.p_student);
    var traces;
    var shapes;
    var layout;

    if (view === "pair") {
      traces = [
        line(x, u.ecg_weak, "Weak", "y", "#111"),
        line(x, u.ecg_strong, "Strong", "y2", "#111"),
      ];
      shapes = classShapes(teacherArg, "y").concat(classShapes(studentArg, "y2"));
      layout = baseLayout(420, shapes, zoomRange, "y2");
      layout.yaxis = yaxis([0.56, 1.0], "Teacher · weak", yRangeCentered(u.ecg_weak));
      layout.yaxis2 = yaxis([0.0, 0.44], "Student · strong", yRangeCentered(u.ecg_strong));
      return { traces: traces, shapes: shapes, layout: layout };
    }

    if (view === "weights") {
      traces = [
        line(x, u.ecg_weak, "ECG", "y", "#111"),
        line(x, derived.wBoundary, "w_boundary", "y2", "#CC6677", {
          fill: "tozeroy",
          fillcolor: "rgba(204,102,119,0.22)",
        }),
        line(x, derived.uniform, "vanilla 1/T", "y2", "#888", {
          dash: "dot",
          width: 1,
        }),
      ];
      shapes = classShapes(teacherArg, "y");
      layout = baseLayout(420, shapes, zoomRange, "y2");
      layout.yaxis = yaxis([0.52, 1.0], "Teacher", yRangeCentered(u.ecg_weak));
      layout.yaxis2 = yaxis([0.0, 0.42], "Weight", [0, 1.05]);
      return { traces: traces, shapes: shapes, layout: layout };
    }

    traces = [line(x, u.ecg_weak, "ECG", "y", "#111")];
    shapes = classShapes(teacherArg, "y");
    layout = baseLayout(320, shapes, zoomRange, "y");
    layout.yaxis = yaxis([0.0, 1.0], "", yRangeCentered(u.ecg_weak));
    return { traces: traces, shapes: shapes, layout: layout };
  }

  function drawTrain(el, step, derived, view, zoomRange, revealT) {
    var built = buildTrain(step, derived, view, zoomRange);
    storePlotState(el, built.traces, built.shapes, zoomRange);
    built.layout.shapes = clipShapes(built.shapes, revealT);
    return Plotly.react(el, visibleTraces(el, revealT), built.layout, PLOT_OPTS);
  }

  function intervalShapes(intervals) {
    var shapes = [];
    var i;
    var iv;
    var c;
    for (i = 0; i < intervals.length; i++) {
      iv = intervals[i];
      c = WAVE_COLORS[iv.wave] || "#888";
      if (iv.end == null) {
        shapes.push({
          type: "line",
          xref: "x",
          yref: "paper",
          x0: iv.start,
          x1: iv.start,
          y0: 0,
          y1: 1,
          line: { color: c, width: 2, dash: "dot" },
        });
        continue;
      }
      shapes.push({
        type: "rect",
        xref: "x",
        yref: "y domain",
        x0: iv.start,
        x1: iv.end,
        y0: 0,
        y1: 1,
        fillcolor: c,
        opacity: 0.38,
        line: { width: 0 },
        layer: "below",
      });
    }
    return shapes;
  }

  function cursorShape(t, color) {
    return {
      type: "line",
      xref: "x",
      yref: "paper",
      x0: t,
      x1: t,
      y0: 0,
      y1: 1,
      line: { color: color || "#111", width: 1.25 },
      layer: "above",
    };
  }

  function setCursor(el, t, color, label) {
    if (!el || !el._mtBaseShapes) return;
    el._mtCursor = t;
    var shapes = el._mtBaseShapes.slice();
    var annotations = [];
    if (t != null) {
      shapes.push(cursorShape(t, color));
      var xmax = el._mtXRange ? el._mtXRange[1] : t;
      var right = xmax > 0 && t > 0.72 * xmax;
      annotations.push({
        x: t,
        y: 0.98,
        xref: "x",
        yref: "paper",
        text: label || String(t),
        showarrow: false,
        xanchor: right ? "right" : "left",
        yanchor: "top",
        font: {
          size: 11,
          family: FONT.family,
          color: color || "#111",
        },
        bgcolor: "rgba(255,255,255,0.92)",
        borderpad: 3,
      });
    }
    return Plotly.relayout(el, {
      shapes: shapes,
      annotations: annotations,
      "xaxis.range": el._mtXRange,
      "xaxis.autorange": false,
    });
  }

  function drawAnnotate(el, ecg, intervals, revealT) {
    var n = ecg.length;
    var x = timeAxis(n);
    var traces = [line(x, ecg, "ECG", "y", "#111")];
    var shapes = intervalShapes(intervals || []);
    var zoomRange = [0, n - 1];
    storePlotState(el, traces, shapes, zoomRange);
    var layout = baseLayout(
      352,
      withPlayMask(shapes, revealT, zoomRange[1]),
      zoomRange,
      "y"
    );
    layout.yaxis = yaxis([0.0, 1.0], "", yRangeCentered(ecg));
    layout.dragmode = false;
    layout.hovermode = false;
    layout.xaxis.showspikes = false;
    return Plotly.react(el, traces, layout, PLOT_OPTS);
  }

  function bandAnchor(el, start, end) {
    if (!el || !el._fullLayout) return null;
    var xa = el._fullLayout.xaxis;
    var ya = el._fullLayout.yaxis;
    if (!xa || !ya || typeof xa.l2p !== "function") return null;
    var mid = (Math.min(start, end) + Math.max(start, end)) / 2;
    var left = xa._offset + xa.l2p(mid);
    var top = ya._offset + 8;
    if (!isFinite(left) || !isFinite(top)) return null;
    return { left: left, top: top };
  }

  function sampleIndexFromClick(el, ev, n) {
    if (!el || !el._fullLayout) return null;
    if (ev.target && ev.target.closest && ev.target.closest(".modebar")) return null;
    var xa = el._fullLayout.xaxis;
    if (!xa) return null;
    var rect = el.getBoundingClientRect();
    var px = ev.clientX - rect.left - xa._offset;
    if (px < 0 || px > xa._length) return null;
    var idx = Math.round(xa.p2d(px));
    if (!isFinite(idx)) return null;
    return Math.max(0, Math.min(n - 1, idx));
  }

  function drawTest(el, testCase, modelKey, revealT) {
    var n = testCase.ecg.length;
    var x = timeAxis(n);
    var pred = testCase[modelKey];
    var traces = [
      line(x, testCase.ecg, "ECG", "y", "#111"),
      line(x, testCase.ecg, "ECG", "y2", "#111"),
    ];
    var shapes = classShapes(testCase.gt, "y").concat(classShapes(pred, "y2"));
    var zoomRange = [0, n - 1];
    storePlotState(el, traces, shapes, zoomRange);
    var layout = baseLayout(360, clipShapes(shapes, revealT), zoomRange, "y2");
    layout.yaxis = yaxis([0.54, 1.0], "Ground truth", yRangeCentered(testCase.ecg));
    layout.yaxis2 = yaxis([0.0, 0.46], "Prediction", yRangeCentered(testCase.ecg));
    return Plotly.react(el, visibleTraces(el, revealT), layout, PLOT_OPTS);
  }

  function setIntervals(el, intervals) {
    if (!el) return;
    el._mtBaseShapes = intervalShapes(intervals || []);
  }

  function setBandReveal(el, t) {
    if (!el || !el._mtXRange) return;
    return Plotly.relayout(el, {
      shapes: clipShapes(el._mtBaseShapes, t),
      "xaxis.range": el._mtXRange,
      "xaxis.autorange": false,
    });
  }

  function setPlayhead(el, t) {
    if (!el || !el._mtXRange) return;
    return Plotly.relayout(el, {
      shapes: withPlayMask(el._mtBaseShapes, t, el._mtXRange[1]),
      "xaxis.range": el._mtXRange,
      "xaxis.autorange": false,
    });
  }

  function clearPlayhead(el) {
    if (!el || !el._mtXRange) return;
    return Plotly.relayout(el, {
      shapes: el._mtBaseShapes || [],
      "xaxis.range": el._mtXRange,
      "xaxis.autorange": false,
    });
  }

  root.MTCharts = {
    COLORS: COLORS,
    NAMES: NAMES,
    WAVE_COLORS: WAVE_COLORS,
    drawTrain: drawTrain,
    drawTest: drawTest,
    drawAnnotate: drawAnnotate,
    sampleIndexFromClick: sampleIndexFromClick,
    bandAnchor: bandAnchor,
    setCursor: setCursor,
    setPlayhead: setPlayhead,
    clearPlayhead: clearPlayhead,
    setIntervals: setIntervals,
    setBandReveal: setBandReveal,
  };
})(window);
