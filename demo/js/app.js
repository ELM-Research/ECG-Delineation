(function () {
  "use strict";

  var FS = 250;
  var PLAY_MS = 900;
  var CAPTION_MS = 700;
  var CASE_LABELS = {
    best: "Best",
    median: "Typical",
    difficult: "Hard",
  };

  var state = {
    caseIndex: 0,
    wave: "P",
    draft: "runet",
    origin: null,
    intervals: [],
    playing: false,
    shouldPlay: true,
    animFrame: null,
    playGen: 0,
    playMode: null,
    hoverHit: -1,
    captionGen: 0,
    submittedMiou: null,
  };

  function $(id) {
    return document.getElementById(id);
  }

  function cases() {
    return (window.TEST_CASES && window.TEST_CASES.cases) || [];
  }

  function currentCase() {
    var list = cases();
    return list[state.caseIndex] || list[0];
  }

  function setActive(ids, activeId) {
    ids.forEach(function (id) {
      var el = $(id);
      if (el) el.classList.toggle("active", id === activeId);
    });
  }

  function stopPlay() {
    state.playGen += 1;
    if (state.animFrame) {
      cancelAnimationFrame(state.animFrame);
      state.animFrame = null;
    }
    state.playing = false;
    state.playMode = null;
  }

  function finishOverlay() {
    var plot = $("anno-plot");
    var mode = state.playMode;
    stopPlay();
    if (!plot) return;
    if (mode === "bands-reverse") clearBandsNow();
    else if (mode === "bands") MTCharts.setBandReveal(plot, null);
    else MTCharts.clearPlayhead(plot);
  }

  function playSignal() {
    var plot = $("anno-plot");
    var c = currentCase();
    if (!plot || !c) return;
    stopPlay();
    var x0 = 0;
    var x1 = c.ecg.length - 1;
    var started = performance.now();
    var gen = state.playGen;
    state.playing = true;
    state.playMode = "signal";
    if (ctaIsShown()) setCta("on");
    else setCta("emerge");
    MTCharts.setPlayhead(plot, x0);

    function tick(now) {
      if (gen !== state.playGen || !state.playing) return;
      var u = Math.min(1, (now - started) / PLAY_MS);
      var xt = x0 + (x1 - x0) * u;
      if (u < 1) {
        MTCharts.setPlayhead(plot, xt);
        state.animFrame = requestAnimationFrame(tick);
      } else {
        MTCharts.clearPlayhead(plot);
        stopPlay();
      }
    }
    state.animFrame = requestAnimationFrame(tick);
  }

  function playBands() {
    var plot = $("anno-plot");
    var c = currentCase();
    if (!plot || !c || !plot._mtXRange) {
      state.shouldPlay = true;
      render();
      return;
    }
    stopPlay();
    MTCharts.clearPlayhead(plot);
    MTCharts.setIntervals(plot, state.intervals);
    var x0 = 0;
    var x1 = c.ecg.length - 1;
    var started = performance.now();
    var gen = state.playGen;
    state.playing = true;
    state.playMode = "bands";
    setCta("on");
    MTCharts.setBandReveal(plot, x0);

    function tick(now) {
      if (gen !== state.playGen || !state.playing) return;
      var u = Math.min(1, (now - started) / PLAY_MS);
      var xt = x0 + (x1 - x0) * u;
      if (u < 1) {
        MTCharts.setBandReveal(plot, xt);
        state.animFrame = requestAnimationFrame(tick);
      } else {
        MTCharts.setBandReveal(plot, null);
        stopPlay();
      }
    }
    state.animFrame = requestAnimationFrame(tick);
  }

  function clearBandsNow() {
    var plot = $("anno-plot");
    state.intervals = [];
    state.origin = null;
    state.submittedMiou = null;
    state.shouldPlay = false;
    hideBandDelete();
    if (plot) {
      MTCharts.setIntervals(plot, []);
      MTCharts.setBandReveal(plot, null);
    }
    renderChips();
    setActive(["wave-P", "wave-QRS", "wave-T"], "wave-" + state.wave);
    setActive(["draft-rules", "draft-runet", "draft-gt"], "");
    updateCaption();
    setCta("on");
  }

  function playReverseBands() {
    var plot = $("anno-plot");
    var c = currentCase();
    if (!plot || !c || !plot._mtXRange || !state.intervals.length) {
      clearBandsNow();
      return;
    }
    stopPlay();
    MTCharts.clearPlayhead(plot);
    MTCharts.setIntervals(plot, state.intervals);
    var x0 = 0;
    var x1 = c.ecg.length - 1;
    var started = performance.now();
    var gen = state.playGen;
    state.playing = true;
    state.playMode = "bands-reverse";
    hideBandDelete();
    setCta("on");
    MTCharts.setBandReveal(plot, x1);

    function tick(now) {
      if (gen !== state.playGen || !state.playing) return;
      var u = Math.min(1, (now - started) / PLAY_MS);
      var xt = x1 + (x0 - x1) * u;
      if (u < 1) {
        MTCharts.setBandReveal(plot, xt);
        state.animFrame = requestAnimationFrame(tick);
      } else {
        stopPlay();
        clearBandsNow();
      }
    }
    state.animFrame = requestAnimationFrame(tick);
  }

  function isBlankInvite() {
    return !state.intervals.length && !state.origin;
  }

  function shouldShowCta() {
    return true;
  }

  function ctaIsShown() {
    var el = $("anno-cta");
    return !!(el && !el.hidden && el.classList.contains("show"));
  }

  function setCta(mode) {
    var el = $("anno-cta");
    if (!el) return;
    if (mode === "off" || !shouldShowCta()) {
      el.classList.remove("show");
      el.hidden = true;
      return;
    }
    el.hidden = false;
    if (mode === "on") {
      el.classList.add("show");
      return;
    }
    el.classList.remove("show");
    requestAnimationFrame(function () {
      requestAnimationFrame(function () {
        if (!shouldShowCta()) return;
        var again = $("anno-cta");
        if (again) again.classList.add("show");
      });
    });
  }

  function formatMiou(value) {
    if (value == null || !isFinite(value)) return null;
    return Number(value).toFixed(2);
  }

  function draftMiou(origin) {
    var c = currentCase();
    if (!c || !c.sample_mean_iou) return null;
    if (origin === "runet") return formatMiou(c.sample_mean_iou.boundary_mt_unet);
    if (origin === "rules") return formatMiou(c.sample_mean_iou.rules);
    return null;
  }

  function draftButtonId(origin) {
    if (origin === "rules") return "draft-rules";
    if (origin === "runet") return "draft-runet";
    if (origin === "gt") return "draft-gt";
    return "";
  }

  function isDraftOrigin(origin) {
    return origin === "runet" || origin === "rules" || origin === "gt";
  }

  function leadLabel() {
    var c = currentCase();
    if (!c) return null;
    var name = c.waveform || c.id || "";
    var m = String(name).match(/_lead_([^.]+)/i);
    return m ? m[1] : null;
  }

  function withLead(text) {
    var lead = leadLabel();
    return lead ? "Lead " + lead + " · " + text : text;
  }

  function captionText() {
    var open = MTIntervals.pending(state.intervals);
    var miou;
    if (open) {
      return withLead("Click the end of this " + open.wave + " interval.");
    }
    if (state.origin === "runet") {
      miou = draftMiou("runet");
      return withLead(
        "R-U-Net draft" +
          (miou ? " · Mean IoU " + miou : "") +
          ". Hover a band and click × to delete, or click the trace to add."
      );
    }
    if (state.origin === "rules") {
      miou = draftMiou("rules");
      return withLead(
        "Rules draft" +
          (miou ? " · Mean IoU " + miou : "") +
          ". Hover a band and click × to delete, or click the trace to add."
      );
    }
    if (state.origin === "gt") {
      return withLead("Ground truth.");
    }
    if (state.submittedMiou != null) {
      return withLead("Your labels · Mean IoU " + state.submittedMiou + ".");
    }
    return withLead("Click to annotate");
  }

  function updateCaption() {
    var el = document.querySelector(".anno-cta-text");
    if (!el) return;
    var next = captionText();
    if (el.textContent === next) {
      el.classList.remove("is-out");
      return;
    }
    state.captionGen += 1;
    var gen = state.captionGen;
    el.classList.add("is-out");
    window.setTimeout(function () {
      if (gen !== state.captionGen) return;
      el.textContent = next;
      requestAnimationFrame(function () {
        if (gen !== state.captionGen) return;
        el.classList.remove("is-out");
      });
    }, CAPTION_MS);
  }

  function renderChips() {
    var host = $("case-chips");
    if (!host) return;
    host.innerHTML = "";
    cases().forEach(function (c, i) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = i === state.caseIndex ? "active" : "";
      btn.textContent = CASE_LABELS[c.id] || CASE_LABELS[c.selection] || c.id;
      btn.addEventListener("click", function () {
        state.caseIndex = i;
        state.shouldPlay = true;
        if (isDraftOrigin(state.origin)) {
          applyDraft();
        } else {
          state.intervals = [];
          state.origin = null;
          render();
        }
      });
      host.appendChild(btn);
    });
  }

  function hideBandDelete() {
    state.hoverHit = -1;
    var btn = $("band-delete");
    if (!btn) return;
    btn.hidden = true;
  }

  function showBandDelete(hit) {
    var btn = $("band-delete");
    var plot = $("anno-plot");
    var iv = state.intervals[hit];
    if (!btn || !plot || !iv || iv.end == null) {
      hideBandDelete();
      return;
    }
    var pos = MTCharts.bandAnchor(plot, iv.start, iv.end);
    if (!pos) {
      hideBandDelete();
      return;
    }
    state.hoverHit = hit;
    var color = MTCharts.WAVE_COLORS[iv.wave] || "#111";
    btn.hidden = false;
    btn.style.left = pos.left + "px";
    btn.style.top = pos.top + "px";
    btn.style.setProperty("--band-color", color);
    btn.setAttribute("aria-label", "Delete this " + iv.wave);
  }

  function renderAnno() {
    var list = cases();
    if (!list.length) {
      $("anno-missing").hidden = false;
      return;
    }
    $("anno-missing").hidden = true;
    hideBandDelete();
    stopPlay();
    renderChips();
    setActive(["wave-P", "wave-QRS", "wave-T"], "wave-" + state.wave);
    setActive(
      ["draft-rules", "draft-runet", "draft-gt"],
      draftButtonId(state.origin)
    );
    updateCaption();
    if (ctaIsShown()) setCta("on");
    else if (state.shouldPlay && isBlankInvite()) setCta("off");
    else setCta("on");
    var c = currentCase();
    var revealT = state.shouldPlay ? 0 : null;
    var drawn = MTCharts.drawAnnotate(
      $("anno-plot"),
      c.ecg,
      state.intervals,
      revealT
    );
    if (state.shouldPlay) {
      state.shouldPlay = false;
      Promise.resolve(drawn).then(function () {
        playSignal();
      });
    }
  }

  function nextExample() {
    var list = cases();
    if (!list.length) return;
    state.caseIndex = (state.caseIndex + 1) % list.length;
    state.shouldPlay = true;
    state.submittedMiou = null;
    hideBandDelete();
    if (isDraftOrigin(state.origin)) {
      applyDraft();
    } else {
      state.intervals = [];
      state.origin = null;
      render();
    }
  }

  function render() {
    renderAnno();
  }

  function submitLabels() {
    var c = currentCase();
    var complete;
    var labels;
    var miou;
    if (!c || !c.gt) return;
    if (MTIntervals.pending(state.intervals)) {
      state.submittedMiou = null;
      updateCaption();
      return;
    }
    complete = state.intervals.filter(function (iv) {
      return iv && iv.end != null;
    });
    if (!complete.length) {
      state.submittedMiou = null;
      updateCaption();
      return;
    }
    labels = MTIntervals.intervalsToLabels(complete, c.gt.length);
    miou = MTIntervals.meanIoU(labels, c.gt);
    state.submittedMiou = formatMiou(miou);
    state.origin = "manual";
    setActive(["draft-rules", "draft-runet", "draft-gt"], "");
    updateCaption();
    setCta("on");
  }

  function applyDraft(bandsOnly) {
    var c = currentCase();
    if (!c) return;
    state.submittedMiou = null;
    if (state.draft === "runet") {
      state.intervals = MTIntervals.labelsToIntervals(c.boundary_mt_unet);
      state.origin = "runet";
    } else if (state.draft === "gt") {
      state.intervals = MTIntervals.labelsToIntervals(c.gt);
      state.origin = "gt";
    } else {
      state.intervals = PanTompkins.detect(c.ecg, FS);
      state.origin = "rules";
    }
    if (!bandsOnly) {
      render();
      return;
    }
    hideBandDelete();
    renderChips();
    setActive(["wave-P", "wave-QRS", "wave-T"], "wave-" + state.wave);
    setActive(
      ["draft-rules", "draft-runet", "draft-gt"],
      draftButtonId(state.origin)
    );
    updateCaption();
    setCta("on");
    playBands();
  }

  function bind() {
    ["P", "QRS", "T"].forEach(function (wave) {
      $("wave-" + wave).addEventListener("click", function () {
        stopPlay();
        state.shouldPlay = false;
        state.wave = wave;
        render();
      });
    });
    $("draft-rules").addEventListener("click", function () {
      state.draft = "rules";
      applyDraft(true);
    });
    $("draft-runet").addEventListener("click", function () {
      state.draft = "runet";
      applyDraft(true);
    });
    $("draft-gt").addEventListener("click", function () {
      state.draft = "gt";
      applyDraft(true);
    });
    $("reset-anno").addEventListener("click", function () {
      playReverseBands();
    });
    $("submit-anno").addEventListener("click", function () {
      submitLabels();
    });
    $("new-example").addEventListener("click", function () {
      nextExample();
    });
    $("anno-plot-wrap").addEventListener("mousemove", function (ev) {
      if (state.playing) {
        hideBandDelete();
        return;
      }
      if (ev.target && ev.target.closest && ev.target.closest("#band-delete")) {
        return;
      }
      var c = currentCase();
      var plot = $("anno-plot");
      if (!c || !plot) return;
      var idx = MTCharts.sampleIndexFromClick(plot, ev, c.ecg.length);
      var label = null;
      if (idx != null) {
        label =
          idx +
          "  ·  " +
          (idx / FS).toFixed(2) +
          " s  ·  " +
          c.ecg[idx].toFixed(2);
      }
      MTCharts.setCursor(
        plot,
        idx,
        MTCharts.WAVE_COLORS[state.wave],
        label
      );
      var hit =
        idx != null
          ? MTIntervals.hitIndex(state.intervals, idx, state.wave)
          : -1;
      if (hit >= 0) showBandDelete(hit);
      else hideBandDelete();
    });
    $("anno-plot-wrap").addEventListener("mouseleave", function () {
      if (state.playing) return;
      hideBandDelete();
      MTCharts.setCursor($("anno-plot"), null);
    });
    $("band-delete").addEventListener("click", function (ev) {
      ev.preventDefault();
      ev.stopPropagation();
      var hit = state.hoverHit;
      if (hit < 0 || hit >= state.intervals.length) return;
      state.intervals.splice(hit, 1);
      state.origin = "manual";
      state.submittedMiou = null;
      state.shouldPlay = false;
      hideBandDelete();
      render();
    });
    $("anno-plot").addEventListener("click", function (ev) {
      if (state.playing) {
        finishOverlay();
        setCta("on");
        return;
      }
      var c = currentCase();
      if (!c) return;
      var idx = MTCharts.sampleIndexFromClick($("anno-plot"), ev, c.ecg.length);
      if (idx == null) return;
      if (
        !MTIntervals.pending(state.intervals) &&
        MTIntervals.hitIndex(state.intervals, idx, state.wave) >= 0
      ) {
        return;
      }
      MTIntervals.clickAt(state.intervals, state.wave, idx);
      state.origin = "manual";
      state.submittedMiou = null;
      state.shouldPlay = false;
      render();
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    bind();
    render();
  });
})();
