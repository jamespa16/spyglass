(function () {
  "use strict";

  const state = {
    index: null,
    mode: "layers",
    family: null,
    layerPos: 0, // position into index.layer_indices
    globalName: null,
    chainView: "heatmap", // "heatmap" | "scatter"
    stackBuiltFor: null, // cache key ("family:<name>" or "blocks") the #stack-list DOM currently reflects
    rotationEnabled: true, // whether the stack views transpose channel-axis-vertical tensors
  };

  const el = (id) => document.getElementById(id);
  const image = el("viewer-image");
  const stackList = el("stack-list");
  const emptyState = el("empty-state");

  function fmtNum(n) {
    if (n === null || n === undefined) return "—";
    if (Math.abs(n) >= 1000 || (Math.abs(n) < 0.001 && n !== 0)) return n.toExponential(3);
    return n.toFixed(6).replace(/0+$/, "").replace(/\.$/, "");
  }

  function shapeStr(shape) {
    return shape ? `(${shape.join(" × ")})` : "unknown";
  }

  async function loadIndex() {
    const res = await fetch("/api/index");
    state.index = await res.json();
    const idx = state.index;

    el("model-name").textContent = idx.input_path || idx.title || "";
    document.title = `spyglass · ${idx.title || "viewer"}`;

    if (idx.families.length) state.family = idx.families[0];
    if (!idx.layer_indices.length && idx.global_tensors.length) state.mode = "global";
    else if (!idx.layer_indices.length && idx.chain.available) state.mode = "chain";

    renderModeButtons();
    renderSidebar();
    setupSlider();
    render();
  }

  function renderModeButtons() {
    const idx = state.index;
    document.querySelectorAll(".mode-btn").forEach((btn) => {
      const mode = btn.dataset.mode;
      let disabled = false;
      if (mode === "layers") disabled = idx.layer_indices.length === 0;
      if (mode === "stack") disabled = idx.layer_indices.length === 0;
      if (mode === "blocks") disabled = idx.layer_indices.length === 0;
      if (mode === "global") disabled = idx.global_tensors.length === 0;
      if (mode === "chain") disabled = !idx.chain.available;
      btn.disabled = disabled;
      btn.classList.toggle("active", state.mode === mode);
      btn.onclick = () => {
        if (btn.disabled) return;
        state.mode = mode;
        renderModeButtons();
        renderSidebar();
        render();
      };
    });
  }

  function renderSidebar() {
    const idx = state.index;

    const showFamilyList = state.mode === "layers" || state.mode === "stack";
    el("family-list").classList.toggle("hidden", !showFamilyList);
    el("block-list").classList.toggle("hidden", state.mode !== "blocks");
    el("global-list").classList.toggle("hidden", state.mode !== "global");
    el("chain-list").classList.toggle("hidden", state.mode !== "chain");
    el("slice-controls").style.visibility = state.mode === "layers" ? "visible" : "hidden";

    if (showFamilyList) {
      const list = el("family-list");
      list.innerHTML = "";
      const heading = document.createElement("div");
      heading.className = "panel-heading";
      heading.textContent = `${idx.families.length} tensor families`;
      list.appendChild(heading);
      idx.families.forEach((fam) => {
        const btn = document.createElement("button");
        btn.className = "panel-item" + (fam === state.family ? " active" : "");
        btn.textContent = fam;
        btn.onclick = () => {
          state.family = fam;
          renderSidebar();
          render();
        };
        list.appendChild(btn);
      });
    }

    if (state.mode === "blocks") {
      const list = el("block-list");
      list.innerHTML = "";
      const heading = document.createElement("div");
      heading.className = "panel-heading";
      heading.textContent = `${idx.layer_indices.length} blocks — jump to`;
      list.appendChild(heading);
      idx.layer_indices.forEach((layerIndex) => {
        const btn = document.createElement("button");
        btn.className = "panel-item";
        btn.textContent = `blk.${layerIndex}`;
        btn.onclick = () => {
          const target = document.getElementById(`block-${layerIndex}`);
          if (target) target.scrollIntoView({ block: "start" });
        };
        list.appendChild(btn);
      });
    }

    if (state.mode === "global") {
      const list = el("global-list");
      list.innerHTML = "";
      const heading = document.createElement("div");
      heading.className = "panel-heading";
      heading.textContent = `${idx.global_tensors.length} global tensors`;
      list.appendChild(heading);
      if (!state.globalName && idx.global_tensors.length) {
        state.globalName = idx.global_tensors[0].name;
      }
      idx.global_tensors.forEach((entry) => {
        const btn = document.createElement("button");
        btn.className = "panel-item" + (entry.name === state.globalName ? " active" : "");
        btn.textContent = entry.name;
        btn.onclick = () => {
          state.globalName = entry.name;
          renderSidebar();
          render();
        };
        list.appendChild(btn);
      });
    }

    if (state.mode === "chain") {
      const list = el("chain-list");
      list.innerHTML = "";

      const toggleRow = document.createElement("div");
      toggleRow.className = "panel-heading";
      toggleRow.textContent = "view";
      list.appendChild(toggleRow);

      [
        ["heatmap", "Depth × channel heatmap"],
        ["scatter", "Gamma scatter"],
      ].forEach(([key, label]) => {
        if (key === "scatter" && !idx.chain.scatter) return;
        const btn = document.createElement("button");
        btn.className = "panel-item" + (state.chainView === key ? " active" : "");
        btn.textContent = label;
        btn.onclick = () => {
          state.chainView = key;
          renderSidebar();
          render();
        };
        list.appendChild(btn);
      });

      const heading = document.createElement("div");
      heading.className = "panel-heading";
      heading.textContent = "top channels";
      list.appendChild(heading);

      const table = document.createElement("table");
      table.className = "channel-table";
      table.innerHTML = "<thead><tr><th>ch</th><th>hits</th><th>rate</th></tr></thead>";
      const tbody = document.createElement("tbody");
      (idx.chain.top_channels || []).slice(0, 25).forEach((c, i) => {
        const tr = document.createElement("tr");
        if (i < Math.ceil((idx.chain.top_channels || []).length * 0.01)) {
          tr.className = "top-channel-row";
        }
        tr.innerHTML = `<td>${c.channel}</td><td>${c.outlier_hits}</td><td>${(c.hit_rate * 100).toFixed(1)}%</td>`;
        tbody.appendChild(tr);
      });
      table.appendChild(tbody);
      list.appendChild(table);
    }
  }

  function setupSlider() {
    const idx = state.index;
    const slider = el("layer-slider");
    slider.min = 0;
    slider.max = Math.max(0, idx.layer_indices.length - 1);
    slider.value = state.layerPos;
    slider.oninput = () => {
      state.layerPos = Number(slider.value);
      render();
    };
    el("layer-prev").onclick = () => stepLayer(-1);
    el("layer-next").onclick = () => stepLayer(1);
  }

  function stepLayer(delta) {
    const idx = state.index;
    if (!idx || state.mode !== "layers" || !idx.layer_indices.length) return;
    const max = idx.layer_indices.length - 1;
    state.layerPos = Math.min(max, Math.max(0, state.layerPos + delta));
    el("layer-slider").value = state.layerPos;
    render();
  }

  function currentLayerEntry() {
    const idx = state.index;
    if (!idx.layer_indices.length) return null;
    const layerIndex = idx.layer_indices[state.layerPos];
    const layerObj = idx.layers[String(layerIndex)] || {};
    return { layerIndex, entry: layerObj[state.family] || null };
  }

  function setImage(src) {
    stackList.classList.add("hidden");
    if (!src) {
      image.classList.add("hidden");
      emptyState.style.display = "flex";
      return;
    }
    image.classList.remove("hidden");
    emptyState.style.display = "none";
    image.src = src;
  }

  function makeStackRow(label, entry, missingText) {
    const rotated = Boolean(entry && entry.rotate && state.rotationEnabled);
    const row = document.createElement("div");
    row.className = "stack-row";
    const labelEl = document.createElement("span");
    labelEl.className = "stack-row-label";
    labelEl.textContent = rotated ? `${label} ⟲` : label;
    labelEl.title = rotated ? "transposed so the residual-stream axis reads horizontally" : "";
    row.appendChild(labelEl);

    if (entry) {
      const img = document.createElement("img");
      img.loading = "lazy";
      img.alt = entry.name;
      img.src = rotated ? `/data/${entry.image}?transpose=1` : `/data/${entry.image}`;
      row.appendChild(img);
    } else {
      labelEl.classList.add("stack-row-missing-label");
      const missing = document.createElement("div");
      missing.className = "stack-row-missing";
      missing.textContent = missingText;
      row.appendChild(missing);
    }
    return row;
  }

  function buildFamilyStack(family) {
    const idx = state.index;
    stackList.innerHTML = "";
    idx.layer_indices.forEach((layerIndex) => {
      const entry = (idx.layers[String(layerIndex)] || {})[family];
      stackList.appendChild(
        makeStackRow(`blk.${layerIndex}`, entry, `(no ${family} tensor at this layer)`)
      );
    });
    state.stackBuiltFor = `family:${family}`;
  }

  function buildBlockStack() {
    const idx = state.index;
    stackList.innerHTML = "";
    idx.layer_indices.forEach((layerIndex) => {
      const layerObj = idx.layers[String(layerIndex)] || {};
      const header = document.createElement("div");
      header.className = "block-header";
      header.id = `block-${layerIndex}`;
      header.textContent = `blk.${layerIndex}`;
      stackList.appendChild(header);

      idx.families.forEach((fam) => {
        const entry = layerObj[fam];
        if (!entry) return; // this family just isn't part of this block's architecture
        stackList.appendChild(makeStackRow(fam, entry, ""));
      });
    });
    state.stackBuiltFor = "blocks";
  }

  function showStack(key, buildFn) {
    image.classList.add("hidden");
    emptyState.style.display = "none";
    stackList.classList.remove("hidden");
    if (state.stackBuiltFor !== key) buildFn();
  }

  function setCorners({ tl, tr, bl, br }) {
    el("corner-tl").textContent = tl || "";
    el("corner-tr").textContent = tr || "";
    el("corner-bl").textContent = bl || "";
    el("corner-br").textContent = br || "";
  }

  function setStats(entry) {
    const dl = el("stats-list");
    dl.innerHTML = "";
    const rows = [];
    if (entry) {
      rows.push(["name", entry.name]);
      if (entry.ggml_type) rows.push(["ggml type", entry.ggml_type]);
      if (entry.shape) rows.push(["shape", shapeStr(entry.shape)]);
      if (entry.stats) {
        rows.push(["min", fmtNum(entry.stats.min)]);
        rows.push(["max", fmtNum(entry.stats.max)]);
        rows.push(["mean", fmtNum(entry.stats.mean)]);
        rows.push(["std", fmtNum(entry.stats.std)]);
        if (entry.stats.tiles) {
          rows.push(["tiles", `${entry.stats.tiles.count} × ${shapeStr(entry.stats.tiles.tile_shape)}`]);
        }
      }
    }
    rows.forEach(([k, v]) => {
      const dt = document.createElement("dt");
      dt.textContent = k;
      const dd = document.createElement("dd");
      dd.textContent = v;
      dl.appendChild(dt);
      dl.appendChild(dd);
    });
  }

  function render() {
    const idx = state.index;
    if (!idx) return;

    if (state.mode === "layers") {
      const current = currentLayerEntry();
      const readout = el("layer-readout");
      if (current) {
        readout.textContent = `layer ${current.layerIndex} (${state.layerPos + 1}/${idx.layer_indices.length})`;
      }
      if (current && current.entry) {
        setImage(`/data/${current.entry.image}`);
        setCorners({
          tl: `${state.family}\nblk.${current.layerIndex}`,
          tr: current.entry.ggml_type ? `${current.entry.ggml_type}\n${shapeStr(current.entry.shape)}` : "",
          bl: `layer ${current.layerIndex}`,
          br: current.entry.stats
            ? `μ ${fmtNum(current.entry.stats.mean)}  σ ${fmtNum(current.entry.stats.std)}`
            : "",
        });
        setStats(current.entry);
      } else {
        setImage(null);
        setCorners({});
        setStats(null);
      }
      return;
    }

    if (state.mode === "stack") {
      if (!state.family) {
        setImage(null);
        setCorners({});
        setStats(null);
        return;
      }
      showStack(`family:${state.family}:${state.rotationEnabled}`, () => buildFamilyStack(state.family));
      const layerObj = idx.layers[String(idx.layer_indices[0])] || {};
      const sample = layerObj[state.family];
      setCorners({
        tl: `${state.family}\nstacked over depth`,
        tr: sample ? `${sample.ggml_type || ""}\n${shapeStr(sample.shape)}`.trim() : "",
        bl: `${idx.layer_indices.length} layers, blk.${idx.layer_indices[0]} → blk.${idx.layer_indices[idx.layer_indices.length - 1]}`,
        br: idx.hidden_size ? `hidden axis (${idx.hidden_size}) horizontal` : "",
      });
      setStats(sample ? { name: `blk.*.${state.family}.weight`, ggml_type: sample.ggml_type, shape: sample.shape } : null);
      return;
    }

    if (state.mode === "blocks") {
      showStack(`blocks:${state.rotationEnabled}`, buildBlockStack);
      setCorners({
        tl: "every tensor, grouped by block",
        tr: `${idx.families.length} families`,
        bl: `${idx.layer_indices.length} blocks, blk.${idx.layer_indices[0]} → blk.${idx.layer_indices[idx.layer_indices.length - 1]}`,
        br: idx.hidden_size ? `hidden axis (${idx.hidden_size}) horizontal` : "",
      });
      setStats(null);
      return;
    }

    if (state.mode === "global") {
      const entry = idx.global_tensors.find((e) => e.name === state.globalName);
      if (entry) {
        setImage(`/data/${entry.image}`);
        setCorners({
          tl: entry.name,
          tr: entry.ggml_type ? `${entry.ggml_type}\n${shapeStr(entry.shape)}` : "",
          bl: "global tensor",
          br: entry.stats ? `μ ${fmtNum(entry.stats.mean)}  σ ${fmtNum(entry.stats.std)}` : "",
        });
        setStats(entry);
      } else {
        setImage(null);
        setCorners({});
        setStats(null);
      }
      return;
    }

    if (state.mode === "chain") {
      const chain = idx.chain;
      const src = state.chainView === "scatter" ? chain.scatter : chain.heatmap;
      setImage(src ? `/data/${src}` : null);
      setCorners({
        tl: state.chainView === "scatter" ? "gamma scatter" : "depth × channel heatmap",
        tr: chain.hidden_size ? `hidden size ${chain.hidden_size}\n${chain.n_layers} layers` : "",
        bl: chain.z_outlier_threshold ? `z > ${chain.z_outlier_threshold}` : "",
        br: "",
      });
      setStats(null);
    }
  }

  function setupWindowLevel() {
    const brightness = el("brightness");
    const contrast = el("contrast");
    const widthScale = el("width-scale");
    const apply = () => {
      const filter = `brightness(${brightness.value / 100}) contrast(${contrast.value / 100})`;
      image.style.filter = filter;
      stackList.style.filter = filter;
      document.documentElement.style.setProperty("--img-width", `${widthScale.value}%`);
    };
    brightness.oninput = apply;
    contrast.oninput = apply;
    widthScale.oninput = apply;
    el("reset-wl").onclick = () => {
      brightness.value = 100;
      contrast.value = 100;
      widthScale.value = 100;
      apply();
    };
  }

  function setupRotateToggle() {
    const btn = el("toggle-rotate");
    const sync = () => {
      btn.textContent = `Channel-axis alignment: ${state.rotationEnabled ? "On" : "Off"}`;
      btn.classList.toggle("active", state.rotationEnabled);
    };
    sync();
    btn.onclick = () => {
      state.rotationEnabled = !state.rotationEnabled;
      sync();
      render();
    };
  }

  function setupInputHandlers() {
    window.addEventListener("keydown", (e) => {
      if (state.mode !== "layers") return;
      if (e.key === "ArrowUp" || e.key === "ArrowLeft") {
        e.preventDefault();
        stepLayer(-1);
      } else if (e.key === "ArrowDown" || e.key === "ArrowRight") {
        e.preventDefault();
        stepLayer(1);
      }
    });

    el("viewport").addEventListener(
      "wheel",
      (e) => {
        if (state.mode !== "layers") return;
        e.preventDefault();
        stepLayer(e.deltaY > 0 ? 1 : -1);
      },
      { passive: false }
    );
  }

  setupWindowLevel();
  setupRotateToggle();
  setupInputHandlers();
  loadIndex().catch((err) => {
    emptyState.textContent = `Failed to load results: ${err}`;
  });
})();
