/* GraspScope dashboard — zero-dependency, SVG charts. */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const fmt = (x, d) => (x == null ? "—" : Number(x).toFixed(d == null ? 2 : d));
  const pct = (x, d) => (x == null ? "—" : (Number(x) * 100).toFixed(d == null ? 0 : d) + "%");

  const FAIL_COLORS = {
    empty_grasp: "#e8a33d",
    wrong_object: "#e85d5d",
    drop: "#4f9fd1",
    success: "#4fd1a5",
  };

  /* ---------- charts ---------- */

  function svg(attrs) {
    const el = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, v);
    return el;
  }

  function makeLineChart(id, opts) {
    const host = $(id);
    host.innerHTML = "";
    const W = 560, H = 260, padL = 46, padR = 14, padT = 16, padB = 30;
    const iw = W - padL - padR, ih = H - padT - padB;
    const xMax = Math.max(1, opts.xMax || 1);
    const yMax = Math.max(0.1, opts.yMax || 1);

    const sx = (x) => padL + (x / xMax) * iw;
    const sy = (y) => padT + ih - (y / yMax) * ih;

    const s = svg({ viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img" });
    // gridlines
    for (let i = 0; i <= 4; i++) {
      const y = sy((i / 4) * yMax);
      const g = svg({ x1: padL, x2: W - padR, y1: y, y2: y, stroke: "#162226", "stroke-width": 1 });
      s.appendChild(g);
      const t = svg({ x: 4, y: y + 3 });
      t.textContent = pct(i / 4, 0);
      s.appendChild(t);
    }
    for (let i = 0; i <= 4; i++) {
      const x = sx((i / 4) * xMax);
      const t = svg({ x: x - 10, y: H - 10 });
      t.textContent = fmt(i / 4, 1);
      s.appendChild(t);
    }
    // y axis label
    const yl = svg({ x: padL - 40, y: padT + 6 });
    yl.textContent = opts.yLabel || "rate";
    s.appendChild(yl);

    const addSeries = (data, color, opts2) => {
      if (!data || data.length < 2) return;
      let d = "";
      data.forEach((p, i) => {
        d += (i ? "L" : "M") + sx(p.x).toFixed(1) + " " + sy(p.y).toFixed(1);
      });
      const path = svg({ d, fill: "none", stroke: color, "stroke-width": opts2.width || 2, "stroke-linecap": "round" });
      s.appendChild(path);
      if (opts2.dots) {
        data.forEach((p) => {
          const c = svg({ cx: sx(p.x), cy: sy(p.y), r: 3.4, fill: color });
          s.appendChild(c);
        });
      }
    };

    // CI band first (behind lines)
    if (opts.ci) {
      const pts = opts.ci;
      let d = "";
      pts.forEach((p, i) => d += (i ? "L" : "M") + sx(p.x).toFixed(1) + " " + sy(p.lo).toFixed(1));
      pts.slice().reverse().forEach((p) => d += "L" + sx(p.x).toFixed(1) + " " + sy(p.hi).toFixed(1));
      d += "Z";
      const band = svg({ d, fill: opts.ciColor || "rgba(79,159,209,.16)", stroke: "none" });
      s.appendChild(band);
    }

    addSeries(opts.main, opts.color, { dots: true, width: 2.5 });
    if (opts.sec) addSeries(opts.sec, opts.secColor, { width: 1.6 });

    // gate threshold marker
    if (opts.gateX != null) {
      const x = sx(opts.gateX);
      const g = svg({ x1: x, x2: x, y1: padT, y2: H - padB, stroke: "rgba(79,209,165,.5)", "stroke-dasharray": "5 4", "stroke-width": 1.4 });
      s.appendChild(g);
      const t = svg({ x: x + 4, y: padT + 6, fill: "#4fd1a5" });
      t.textContent = "gate α=" + fmt(opts.gateX, 2);
      s.appendChild(t);
    }
    // cliff marker
    if (opts.cliffX != null) {
      const x = sx(opts.cliffX);
      const g = svg({ x1: x, x2: x, y1: padT, y2: H - padB, stroke: "rgba(232,163,61,.6)", "stroke-dasharray": "2 4", "stroke-width": 1.4 });
      s.appendChild(g);
      const t = svg({ x: x - 90, y: H - padB + 4, fill: "#e8a33d" });
      t.textContent = "cliff α=" + fmt(opts.cliffX, 2);
      s.appendChild(t);
    }
    // real-world scatter anchor
    if (opts.scatter) {
      opts.scatter.forEach((p) => {
        const cx = sx(p.x), cy = sy(p.y);
        const c = svg({ cx, cy, r: 6, fill: "none", stroke: "#fff", "stroke-width": 2 });
        s.appendChild(c);
        const inner = svg({ cx, cy, r: 3.4, fill: "#fff" });
        s.appendChild(inner);
        const t = svg({ x: cx - 6, y: cy - 10, fill: "#fff" });
        t.textContent = p.label || "real";
        s.appendChild(t);
      });
    }
    host.appendChild(s);
  }

  function makeDecomp(id, tiers) {
    const host = $(id);
    host.innerHTML = "";
    const W = 360, H = 260, padL = 40, padR = 10, padT = 14, padB = 26;
    const iw = W - padL - padR, ih = H - padT - padB;
    const s = svg({ viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img" });
    const n = tiers.length;
    const bw = iw / n;
    const colors = ["empty_grasp", "wrong_object", "drop", "success"];
    let prev = tiers.map(() => 0);
    for (const key of colors) {
      tiers.forEach((t, i) => {
        const v = t[key] || 0;
        if (v <= 0) return;
        const x = padL + i * bw;
        const y0 = padT + ih - prev[i] * ih;
        const h = v * ih;
        const r = svg({ x, y: y0 - h, width: bw - 4, height: h, fill: FAIL_COLORS[key] });
        s.appendChild(r);
        prev[i] += v;
      });
    }
    tiers.forEach((t, i) => {
      const x = padL + i * bw;
      const l = svg({ x: x + bw / 2 - 10, y: H - 10 });
      l.textContent = "α=" + t.alpha;
      s.appendChild(l);
    });
    host.appendChild(s);
  }

  /* ---------- rendering ---------- */

  function render(data) {
    $("statusBadge").textContent = "live results";
    const curve = (data.frontier && data.frontier.curve) || [];
    const dec = data.failure_decomposition || {};

    // gate card
    const gate = data.gate || {};
    const gateOk = gate.found;
    $("coverageMin").textContent = gate.coverage_min != null ? fmt(gate.coverage_min, 2) : "—";
    $("gateCard").classList.add(gateOk ? "pass" : "fail");
    $("gateBadge").textContent = gateOk ? "gate α ≥ " + fmt(gate.coverage_min, 2) : "no gate constraint";
    const gKv = $("gateKv");
    gKv.innerHTML = "";
    const gRows = [
      ["max failure", gate.max_fail_rate != null ? pct(gate.max_fail_rate, 0) : "—"],
      ["interp from", gate.interp_from != null ? "α=" + fmt(gate.coverage_min - (gate.coverage_min - 0.6), 2) + " (" + pct(gate.interp_from, 1) + ")" : "—"],
      ["cliff tier", gate.cliff_tier || "—"],
      ["method", "linear interp on monotone frontier"],
    ];
    if (gate.tier_below && gate.tier_above) {
      gRows[1] = ["segment", gate.tier_below + " → " + gate.tier_above];
    }
    for (const [k, v] of gRows) {
      const dt = document.createElement("dt"); dt.textContent = k;
      const dd = document.createElement("dd"); dd.textContent = v;
      gKv.appendChild(dt); gKv.appendChild(dd);
    }

    // cliff card
    const cl = data.frontier || {};
    $("cliffCoverage").textContent = fmt(cl.cliff_coverage, 2);
    const cKv = $("cliffKv");
    cKv.innerHTML = "";
    const cRows = [
      ["separation", fmt(cl.cliff_separation, 1) + "×"],
      ["separation CI", cl.cliff_separation_ci ? fmt(cl.cliff_separation_ci[0], 2) + "–" + fmt(cl.cliff_separation_ci[1], 2) + "×" : "—"],
      ["monotonic", cl.monotonic ? "yes" : "no"],
      ["method", cl.method || "max separation"],
    ];
    for (const [k, v] of cRows) {
      const dt = document.createElement("dt"); dt.textContent = k;
      const dd = document.createElement("dd"); dd.textContent = v;
      cKv.appendChild(dt); cKv.appendChild(dd);
    }

    // real anchor
    const real = data.real_anchor || {};
    $("realSuccess").textContent = pct(real.success_rate, 0);
    const rKv = $("realKv");
    rKv.innerHTML = "";
    const rRows = [
      ["failure", pct(real.failure_rate, 0)],
      ["success 95% CI", real.success_ci ? pct(real.success_ci[0], 0) + "–" + pct(real.success_ci[1], 0) : "—"],
      ["n", String(real.n || "—")],
      ["v α=1.0 synth", curve.length ? pct(curve[0].failure_rate, 1) : "—"],
    ];
    for (const [k, v] of rRows) {
      const dt = document.createElement("dt"); dt.textContent = k;
      const dd = document.createElement("dd"); dd.textContent = v;
      rKv.appendChild(dt); rKv.appendChild(dd);
    }

    // chart
    const asc = curve.slice().sort((a, b) => a.coverage - b.coverage);
    const main = asc.map((p) => ({ x: p.coverage, y: p.failure_rate }));
    const sec = asc.map((p) => ({ x: p.coverage, y: 1 - p.failure_rate }));
    const ci = asc.map((p) => ({ x: p.coverage, lo: p.failure_rate_ci_lo, hi: p.failure_rate_ci_hi }));
    makeLineChart("cliffChart", {
      main, sec, ci,
      color: "#e85d5d", secColor: "rgba(79,209,165,.7)", ciColor: "rgba(79,159,209,.14)",
      xMax: 1, yMax: 0.7, yLabel: "rate",
      gateX: gate.coverage_min,
      cliffX: cl.cliff_coverage,
      scatter: real && real.failure_rate != null
        ? [{ x: 1.0, y: real.failure_rate, label: "real" }]
        : null,
    });

    // decomposition
    const tiers = Object.entries(dec).map(([k, v]) => ({ alpha: k.replace("alpha_", ""), ...v }));
    makeDecomp("decompChart", tiers);

    // table
    const tb = document.querySelector("#frontierTable tbody");
    tb.innerHTML = "";
    const rows = curve.slice().sort((a, b) => b.coverage - a.coverage);
    for (const p of rows) {
      const tr = document.createElement("tr");
      if (p.coverage === cl.cliff_coverage) tr.className = "cliff-row";
      const cells = [
        p.tier || "—",
        fmt(p.coverage, 1),
        "150",
        pct(p.failure_rate, 1),
        pct(1 - p.failure_rate, 1),
        pct(p.failure_rate_ci_lo, 1),
        pct(p.failure_rate_ci_hi, 1),
      ];
      cells.forEach((c, i) => {
        const td = document.createElement("td");
        if (i > 0) td.className = "num";
        td.textContent = c;
        tr.appendChild(td);
      });
      tb.appendChild(tr);
    }

    // scenes
    const grid = $("sceneGrid");
    grid.innerHTML = "";
    const scenes = data.scenes || [];
    if (scenes.length) {
      scenes.forEach((s) => {
        const card = document.createElement("div");
        card.className = "scene-card";
        const img = document.createElement("img");
        img.src = s.url;
        img.loading = "lazy";
        img.alt = "scene " + s.scene_id;
        const cap = document.createElement("div");
        cap.className = "cap";
        cap.innerHTML = "<span><b>" + s.scene_id + "</b> · α=" + fmt(s.coverage, 1) + "</span><span class='" + (s.success ? "pos" : "neg") + "'>" + (s.success ? "success" : "fail") + "</span>";
        card.appendChild(img); card.appendChild(cap);
        grid.appendChild(card);
      });
    } else {
      const p = document.createElement("p");
      p.className = "hint";
      p.textContent = "No scene thumbnails available — run the closed-loop sweep with scene export enabled.";
      grid.appendChild(p);
    }

    // representative cases
    const cg = $("caseGrid");
    cg.innerHTML = "";
    const cases = data.cases || [];
    if (cases.length) {
      cases.forEach((c) => {
        const card = document.createElement("div");
        card.className = "scene-card " + (c.success ? "ok" : "fail");
        const img = document.createElement("img");
        img.src = c.image_url;
        img.loading = "lazy";
        img.alt = "case " + c.scene_id;
        const cap = document.createElement("div");
        cap.className = "cap";
        const outcome = c.success
          ? '<span class="pos">grasp ok</span>'
          : '<span class="neg">' + (c.failure_type || "fail") + '</span>';
        cap.innerHTML = "<span><b>" + c.scene_id + "</b> · α=" + fmt(c.coverage, 1) + "</span>" + outcome;
        if (c.target_cls) {
          const extra = document.createElement("div");
          extra.className = "cap";
          extra.innerHTML = "<span>target <b>" + c.target_cls + "</b> · attempts " + (c.n_attempts || 1) + "</span>";
          card.appendChild(img); card.appendChild(cap); card.appendChild(extra);
        } else {
          card.appendChild(img); card.appendChild(cap);
        }
        cg.appendChild(card);
      });
    }

    const ts = (data.exported_at || "").replace("T", " ").slice(0, 19);
    $("footTs").textContent = "exported " + (ts || "—");
  }

  async function load() {
    try {
      const r = await fetch("/api/graspscope/frontier.json");
      if (!r.ok) throw new Error("HTTP " + r.status);
      const data = await r.json();
      render(data);
    } catch (err) {
      $("statusBadge").textContent = "offline";
      $("statusBadge").classList.remove("ok");
      $("statusBadge").classList.add("cliff");
      const s = document.createElement("p");
      s.id = "status";
      s.className = "err";
      s.textContent = "Could not load grasp gate results: " + err.message;
      document.querySelector(".wrap").appendChild(s);
    }
  }

  load();
})();
