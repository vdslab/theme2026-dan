"use strict";

const get = (id) => document.getElementById(id);
const colors = { A: "#58bfea", B: "#ee8de1", C: "#7cdda2", D: "#ff9a65" };
const buttons = [...get("choices").querySelectorAll("button")];
const byItem = new Map();
const records = [];
let session = null;
let active = null;
const limitMilliseconds = 60000;

for (const button of buttons) {
    const color = colors[button.dataset.choice];
    if (color) button.style.setProperty("--edge-color", color);
}

function validate(study, slot) {
    if (study.schema !== 1 || study.experiment !== "visnotes_reading_study_01" ||
        !/^[a-f0-9]{64}$/.test(study.study_id) || study.width !== 900 || study.height !== 520 ||
        !Number.isInteger(slot) || slot < 0 || slot >= 18 ||
        study.allocation_slots !== 18 || study.assignments?.length !== 18) {
        throw new Error("刺激ファイルの形式または割付番号が違います。");
    }
    const conditions = ["planar", "first_fit", "proposed"];
    const trials = study.assignments[slot];
    const point = (value) => Array.isArray(value) && value.length === 2 && value.every(Number.isFinite);
    if (trials.length !== study.items.length ||
        trials.length !== conditions.length * study.trials_per_condition ||
        new Set(trials.map(({ item }) => item)).size !== trials.length ||
        conditions.some((key) => trials.filter(({ condition }) => condition === key).length !== study.trials_per_condition)) {
        throw new Error("課題と条件の割付が一致しません。");
    }
    for (const item of study.items) {
        for (const key of conditions) {
            const panel = item.panels[key];
            if (panel.guide.length !== (key === "planar" ? 1 : 2) || !panel.guide.every(point) ||
                panel.faces.length !== 4 || panel.faces.map(({ code }) => code).sort().join("") !== "ABCD" ||
                panel.faces.some((face) => !point(face.anchor) || face.ring.length < 3 || !face.ring.every(point))) {
                throw new Error("刺激画像の座標が不正です。");
            }
        }
        if (byItem.has(item.id)) throw new Error("同じ項目が複数回あります。");
        byItem.set(item.id, item);
    }
    if (trials.some(({ item }) => !byItem.has(item))) throw new Error("未定義の項目があります。");
    return trials;
}

function svgElement(tag, attrs) {
    const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
    return element;
}

function points(coords) {
    return coords.map(([x, y]) => `${x},${y}`).join(" ");
}

function draw(panel) {
    const image = get("image");
    image.replaceChildren();
    for (const { code, ring } of panel.faces) {
        image.append(svgElement("polygon", {
            points: points(ring), fill: colors[code], "fill-opacity": .23,
            stroke: colors[code], "stroke-width": 2.5, "stroke-linejoin": "round",
        }));
        for (const [x, y] of ring) {
            image.append(svgElement("circle", { cx: x, cy: y, r: 2.4,
                fill: colors[code], "fill-opacity": .82 }));
        }
    }
    if (panel.guide.length > 1) {
        image.append(svgElement("polyline", {
            points: points(panel.guide), stroke: "#fff393", "stroke-width": 3,
            "stroke-dasharray": "7 5", fill: "none",
        }));
    }
    for (const [x, y] of panel.guide) {
        image.append(svgElement("circle", { cx: x, cy: y, r: 6, fill: "#fff393", stroke: "#151f2b" }));
    }
    const used = [...panel.guide];
    const candidates = [[0, 0], [0, -30], [0, 30], [-30, 0], [30, 0],
                        [-28, -28], [28, -28], [-28, 28], [28, 28]];
    const ordered = [...panel.faces].sort((first, second) => first.code.localeCompare(second.code));
    for (const { code, anchor: [x, y] } of ordered) {
        const place = candidates.map(([dx, dy]) => [x + dx, y + dy]).find(([px, py]) =>
            used.every(([ux, uy]) => Math.hypot(px - ux, py - uy) >= 24)
        ) || [x, y];
        used.push(place);
        if (place[0] !== x || place[1] !== y) {
            image.append(svgElement("line", { x1: x, y1: y, x2: place[0], y2: place[1],
                stroke: colors[code], "stroke-width": 1.5 }));
        }
        const label = svgElement("text", {
            x: place[0], y: place[1], fill: colors[code], stroke: "#172332", "stroke-width": 5,
            "paint-order": "stroke", "text-anchor": "middle", "dominant-baseline": "central",
            "font-size": 26, "font-weight": 700,
        });
        label.textContent = code;
        image.append(label);
    }
}

function ready() {
    get("image").setAttribute("hidden", "");
    get("choices").hidden = true;
    get("viewport-warning").hidden = true;
    get("export").disabled = records.length === 0;
    const finished = records.length === session.trials.length;
    get("next").hidden = finished;
    get("instruct").hidden = finished;
    get("progress").textContent = finished
        ? "全課題が終了しました。記録を保存してください。"
        : `課題 ${records.length + 1} / ${session.trials.length}`;
    (finished ? get("export") : get("next")).focus();
}

function viewport() {
    return { width: innerWidth, height: innerHeight, device_pixel_ratio: devicePixelRatio };
}

get("setup").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (session) return;
    const submit = event.currentTarget.querySelector("button");
    submit.disabled = true;
    byItem.clear();
    get("error").textContent = "";
    try {
        const study = JSON.parse(await get("stimulus").files[0].text());
        const slot = get("slot").valueAsNumber;
        const trials = validate(study, slot);
        session = {
            study_id: study.study_id, slot, trials, participant: get("participant").value,
            purpose: get("purpose").value, viewport: viewport(),
        };
        get("image").setAttribute("viewBox", `0 0 ${study.width} ${study.height}`);
        get("setup").hidden = true;
        get("session").hidden = false;
        ready();
    } catch (error) {
        get("error").textContent = error.message;
        submit.disabled = false;
    }
});

get("next").addEventListener("click", () => {
    if (active || records.length === session.trials.length) return;
    const trial = session.trials[records.length];
    const resized = JSON.stringify(session.viewport) !== JSON.stringify(viewport());
    get("viewport-warning").hidden = !resized;
    active = { ...trial, started: null,
               interrupted: resized || document.hidden || !document.hasFocus() };
    draw(byItem.get(trial.item).panels[trial.condition]);
    get("next").hidden = true;
    get("export").disabled = true;
    // Reveal the already-built image and start timing in the same rendering frame.
    requestAnimationFrame(() => {
        get("image").removeAttribute("hidden");
        get("choices").hidden = false;
        active.started = performance.now();
        active.timeout = setTimeout(() => record("timeout"), limitMilliseconds);
        get("image").focus();
    });
});

function interrupt() {
    if (active) active.interrupted = true;
}
window.addEventListener("blur", interrupt);
window.addEventListener("resize", interrupt);
document.addEventListener("visibilitychange", () => { if (document.hidden) interrupt(); });

function record(response) {
    const { item, condition, interrupted, started, timeout } = active;
    clearTimeout(timeout);
    const elapsed = performance.now() - started;
    const timedOut = response === "timeout" || elapsed >= limitMilliseconds;
    records.push({ item, condition, interrupted,
                   response: timedOut ? "timeout" : response,
                   milliseconds: timedOut ? limitMilliseconds : Math.min(Math.round(elapsed), limitMilliseconds - 1) });
    active = null;
    ready();
}

get("choices").addEventListener("click", (event) => {
    const response = event.target.dataset.choice;
    if (response && active && active.started !== null && !event.target.disabled) record(response);
});

get("export").addEventListener("click", () => {
    if (active || !records.length) return;
    const { trials, ...metadata } = session;
    const data = JSON.stringify({
        schema: 1, ...metadata, complete: records.length === trials.length, records,
    }, null, 2);
    const url = URL.createObjectURL(new Blob([data], { type: "application/json" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `reading-${session.participant}-${session.purpose}.json`;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
});
