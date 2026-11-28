// node --experimental-websocket experiments/visnotes_dense_validation_01/capture_viewer.mjs <new-output-directory> [dblp|original]
import assert from "node:assert/strict";
import { execFileSync, spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { once } from "node:events";
import { mkdir, mkdtemp, readFile, realpath, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../../", import.meta.url));
const [destination, only] = process.argv.slice(2);
assert.ok(destination, "Pass a new output directory");
assert.ok(!only || ["dblp", "original"].includes(only));
const out = resolve(destination);
const runs = "results/experiments/";
const cases = {
    dblp: { folder: runs + "visnotes_public_validation_01/public-main/dblp",
        source: runs + "visnotes_public_validation_01/inputs/dblp.json", focus: 48 },
    original: { folder: runs + "visnotes_remaining_01/exp04/original", source: "original.json", focus: 198 },
};
const viewport = { width: 1600, height: 1020, deviceScaleFactor: 1, mobile: false };
const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");
const resources = new Map();
const hashes = {};
const paths = {};
async function record(path) {
    const bytes = await readFile(join(root, path));
    hashes[path] = hash(bytes);
    return bytes;
}
for (const [path, type] of [["viewer_visnotes.html", "text/html; charset=utf-8"], ["viewer_visnotes.js", "text/javascript"]]) {
    resources.set("/" + path, [type, await record(path)]);
}
const scriptPath = "experiments/visnotes_dense_validation_01/capture_viewer.mjs";
await record(scriptPath);
for (const [name, entry] of Object.entries(cases)) {
    if (only && only !== name) continue;
    const source = JSON.parse(await record(entry.source));
    paths[name] = [];
    for (const [method, suffix] of [["first-fit", "greedy-input/deterministic"], ["proposed-seed0", "full/seed-0"]]) {
        const layoutPath = entry.folder + "/" + suffix + "/layout.json";
        const layout = JSON.parse(await record(layoutPath));
        const nodes = layout.nodes.map(({ id, x, y }) => ({ id, x, y }));
        assert.deepEqual(nodes, source.nodes.map(({ id, x, y }) => ({ id, x, y })));
        assert.deepEqual(layout.hyperedges.map(({ id, nodes }) => ({ id, nodes })),
            source.hyperedges.map(({ id, nodes }) => ({ id, nodes })));
        const safe = { nodes: nodes.map((node) => ({ ...node, label: "Vertex " + node.id })),
            hyperedges: layout.hyperedges.map(({ id, nodes, layer }) => ({ id, nodes, layer, label: "Edge " + id })) };
        const url = "/" + name + "-" + method + ".json";
        resources.set(url, ["application/json", JSON.stringify(safe)]);
        paths[name].push({ layoutPath, safe, url: url.slice(1) });
    }
}
await mkdir(dirname(out), { recursive: true });
await mkdir(out);
const server = createServer((request, response) => {
    const reply = resources.get(new URL(request.url, "http://127.0.0.1").pathname);
    if (reply) response.writeHead(200, { "Content-Type": reply[0] }).end(reply[1]);
    else response.writeHead(404).end();
});
const git = (...args) => execFileSync("git", args, { cwd: root, encoding: "utf8" }).trim();
const result = { git_commit: git("rev-parse", "HEAD"), capture_script_status: git("status", "--short", "--", scriptPath),
    command: ["node", "--experimental-websocket", scriptPath, destination, ...(only ? [only] : [])],
    budget: "Two sequential archived layout pairs; each wait up to 40 s, three interactive actions plus zoom and reset; no solver rerun.",
    sources_sha256: hashes, viewport, cases: {},
    timing: "Navigation to both displayed counts plus two animation frames (includes CDN, polling, cold/warm cache); actions to second animation frame, one observation each, software WebGL. No human timing or GPU benchmark." };

async function wait(action, label) {
    const deadline = Date.now() + 40_000;
    while (Date.now() < deadline) {
        const value = await action();
        if (value) return value;
        await sleep(100);
    }
    throw new Error("Timeout: " + label);
}

// Observe the unmodified viewer after Three.js renders; each pane supplies its own scene and camera.
async function installProbe() {
    const three = await import("three");
    const frames = new Map();
    const previous = three.Scene.prototype.onAfterRender;
    three.Scene.prototype.onAfterRender = function(renderer, scene, camera, ...rest) {
        previous.call(this, renderer, scene, camera, ...rest);
        frames.set(renderer.domElement.closest(".pane").id, { renderer, scene, camera });
    };
    window.captureState = () => ["pane-a", "pane-b"].map((id, idx) => {
        const { renderer, scene, camera } = frames.get(id);
        const focus = scene.children[1];
        const line = focus.children.find((obj) => obj.isLine);
        const marks = focus.children.filter((obj) => obj.isMesh).map((obj) => obj.position.toArray());
        const counts = document.querySelector("#" + id + " .counts").textContent;
        const members = document.querySelectorAll("#node-details .memberships")[idx];
        const gl = renderer.getContext();
        const ext = gl.getExtension("WEBGL_debug_renderer_info");
        return { counts, camera_position: camera.position.toArray(), quaternion: camera.quaternion.toArray(),
            zoom: camera.zoom, units_per_pixel: (camera.right - camera.left) / renderer.domElement.clientWidth / camera.zoom,
            contacts: marks, pillar: line ? Array.from(line.geometry.getAttribute("position").array) : null,
            contacts_in_view: marks.filter((mark) => {
                const p = new three.Vector3(...mark).project(camera);
                return Math.abs(p.x)<=1 && Math.abs(p.y)<=1 && Math.abs(p.z)<=1;
            }).length,
            membership_labels: [...(members?.children ?? [])].map((node) => [node.textContent, !node.classList.contains("muted")]),
            webgl: ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
            draw_calls: renderer.info.render.calls };
    });
}

const temp = await realpath(tmpdir());
const profile = await mkdtemp(join(temp, "visnotes-dense-"));
let browser, socket;
try {
    server.listen(0, "127.0.0.1");
    await once(server, "listening");
    const chrome = process.env.CHROME_PATH || join(process.env.ProgramFiles || "", "Google", "Chrome", "Application", "chrome.exe");
    browser = spawn(chrome, ["--headless=new", "--remote-debugging-port=0", "--remote-debugging-address=127.0.0.1",
        "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--disable-background-timer-throttling",
        "--disable-background-networking", "--user-data-dir=" + profile, "about:blank"], { windowsHide: true, stdio: "ignore" });
    const portFile = await wait(() => readFile(join(profile, "DevToolsActivePort"), "utf8").catch(() => null), "Chrome DevTools");
    const tabs = await fetch("http://127.0.0.1:" + portFile.split("\n")[0] + "/json/list").then((r) => r.json());
    socket = new WebSocket(tabs.find((tab) => tab.type === "page").webSocketDebuggerUrl);
    await once(socket, "open");
    let seq = 0;
    const pending = new Map();
    socket.addEventListener("message", ({ data }) => {
        const msg = JSON.parse(data);
        if (pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); }
    });
    function call(method, params = {}) {
        const id = ++seq;
        return new Promise((done, fail) => {
            const timeout = setTimeout(() => { pending.delete(id); fail(new Error(method + " timed out")); }, 40_000);
            pending.set(id, (m) => { clearTimeout(timeout); if (m.error) fail(new Error(m.error.message)); else done(m.result); });
            socket.send(JSON.stringify({ id, method, params }));
        });
    }
    async function evaluate(expression) {
        const r = await call("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
        if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
        return r.result.value;
    }
    const frames = "new Promise(done=>requestAnimationFrame(()=>requestAnimationFrame(()=>done())))";
    result.browser = (await call("Browser.getVersion")).product;
    await call("Page.enable");
    await call("Emulation.setDeviceMetricsOverride", viewport);
    for (const [name, pair] of Object.entries(paths)) {
        const { focus } = cases[name];
        const query = new URLSearchParams({ a: pair[0].url, b: pair[1].url });
        const route = "http://127.0.0.1:" + server.address().port + "/viewer_visnotes.html?" + query;
        await call("Page.navigate", { url: route });
        await wait(() => evaluate(`(() => {
            const status=document.querySelector("#status");
            if(status?.classList.contains("error")) throw Error(status.textContent);
            return document.querySelector("#pane-b .dataset")?.textContent === ${JSON.stringify(pair[1].url)}
                && document.querySelector("#pane-a .counts")?.textContent && document.querySelector("#pane-b .counts")?.textContent;
        })()`), "both complete datasets");
        const loaded = await evaluate(`(async()=>{await ${frames};return performance.now()})()`);
        await evaluate("(" + installProbe.toString() + ")()");
        const dataset = result.cases[name] = { focus, n_vertices: pair[0].safe.nodes.length, n_edges: pair[0].safe.hyperedges.length,
            layouts: pair.map((entry) => entry.layoutPath), navigation_to_observed_display_ms: loaded, stages: {} };
        const maxLayer = Math.max(...pair.flatMap(({ safe }) => safe.hyperedges.map((edge) => edge.layer)));
        const highSlice = Math.max(1, Math.floor(maxLayer / 4));
        const lowSlice = 0;
        async function stage(label, action, selected, low, high, capture = true) {
            const start = performance.now();
            await action();
            await evaluate(frames);
            const actionMs = performance.now() - start;
            const panes = await evaluate("window.captureState()");
            for (const key of ["camera_position", "quaternion"]) {
                assert.ok(panes[0][key].every((v, i) => Math.abs(v - panes[1][key][i]) < 1e-10), key);
            }
            assert.equal(panes[0].zoom, panes[1].zoom);
            assert.ok(Math.abs(panes[0].units_per_pixel - panes[1].units_per_pixel) < 1e-9);
            for (const [idx, p] of panes.entries()) {
                const { nodes, hyperedges } = pair[idx].safe;
                const min = Math.min(...hyperedges.map((edge) => edge.layer));
                const height = (e) => e.layer - min;
                const visible = (e) => low <= height(e) && height(e) <= high;
                const active = hyperedges.filter(visible);
                const counts = Array.from(p.counts.replaceAll(",", "").matchAll(/\d+/g), (m) => Number(m[0]));
                const layers = (edges) => new Set(edges.map(height)).size;
                assert.deepEqual(counts.slice(0, 4), [layers(active), layers(hyperedges), active.length, hyperedges.length]);
                const incident = selected ? hyperedges.filter((e) => e.nodes.includes(focus)) : [];
                const xy = nodes.find((n) => n.id === focus);
                const xs = nodes.map((n) => n.x), ys = nodes.map((n) => n.y);
                const span = Math.max(Math.max(...xs)-Math.min(...xs), Math.max(...ys)-Math.min(...ys));
                const x = (xy.x - (Math.min(...xs)+Math.max(...xs))/2)/span;
                const y = (xy.y - (Math.min(...ys)+Math.max(...ys))/2)/span;
                const expected = [...new Set(incident.filter(visible).map(height))].sort((a,b)=>a-b);
                assert.deepEqual(p.contacts.map((m) => Math.round(m[2]*100)), expected);
                assert.ok(p.contacts.every((m) => Math.abs(m[0]-x)<1e-6 && Math.abs(m[1]-y)<1e-6));
                const bottom = Math.max(low, Math.min(...incident.map(height)));
                const top = Math.min(high, Math.max(...incident.map(height)));
                assert.equal(Boolean(p.pillar), top > bottom);
                if (p.pillar) assert.ok(p.pillar.every((v, i) => Math.abs(v-[x,y,bottom*.01,x,y,top*.01][i])<1e-6));
                const labels = incident.map((e) => [height(e)+" 層 · Edge "+e.id, visible(e)]);
                assert.deepEqual(p.membership_labels.sort(), labels.sort());
                p.incident_edges = incident.length;
                p.visible_incident_edges = incident.filter(visible).length;
                p.contact_layers = expected;
                p.pillar_layer_range = p.pillar ? [bottom,top] : null;
                delete p.contacts; delete p.pillar; delete p.membership_labels;
            }
            const item = dataset.stages[label] = { low, high, action_to_two_frames_ms: actionMs, panes };
            if (capture) {
                item.png = name + "-" + label + ".png";
                const shot = await call("Page.captureScreenshot", { format: "png" });
                const bytes = Buffer.from(shot.data, "base64");
                await writeFile(join(out, item.png), bytes, { flag: "wx" });
                item.sha256 = hash(bytes);
            }
            console.log(name, label, panes.map((p) => p.counts).join(" | "));
        }
        await stage("overview", () => evaluate("document.querySelector('#reset-view').click()"), false, 0, maxLayer);
        await stage("hub", () => evaluate(`(() => {
            const select=document.querySelector("#node-results");
            select.value=[...select.options].find(o=>JSON.parse(o.value)[0]===${focus}).value;
            select.dispatchEvent(new Event("change",{bubbles:true}));
        })()`), true, 0, maxLayer);
        await stage("zoom", async () => {
            const pos=await evaluate("(()=>{const r=document.querySelector('#pane-a .stage').getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()");
            await call("Input.dispatchMouseEvent", { type: "mouseWheel", ...pos, deltaX: 0, deltaY: -1200 });
        }, true, 0, maxLayer);
        assert.ok(dataset.stages.zoom.panes[0].zoom > dataset.stages.hub.panes[0].zoom);
        await stage("slice", () => evaluate(`(() => {
            const ctrl=document.querySelector("#layer-high");ctrl.value=${highSlice};ctrl.dispatchEvent(new Event("change",{bubbles:true}));
        })()`), true, lowSlice, highSlice);
        await stage("restored", () => evaluate("document.querySelector('#all-layers').click()"), true, 0, maxLayer, false);
        await stage("cleared", () => evaluate("document.querySelector('#clear-node').click()"), false, 0, maxLayer, false);
        await writeFile(join(out, name+".json"), JSON.stringify(dataset,null,2)+"\n", { flag: "wx" });
    }
    result.end = "Both files, paired cameras, counts, memberships and rendered pillar/contacts verified for all requested cases.";
    await writeFile(join(out, "manifest.json"), JSON.stringify(result, null, 2)+"\n", { flag: "wx" });
} finally {
    socket?.close();
    browser?.kill();
    server.close();
    const path = await realpath(profile).catch(() => null);
    if (path && dirname(path) === temp) await rm(path, { recursive: true, force: true, maxRetries: 3, retryDelay: 150 }).catch(() => {});
}
