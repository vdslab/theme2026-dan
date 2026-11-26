// Record a local VisNotes demo from browser screenshots of reproducible operations.
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { once } from "node:events";
import { mkdir, mkdtemp, readFile, realpath, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

const ROOT = fileURLToPath(new URL("../", import.meta.url));
const LIMIT = 25 * 1024 * 1024;
const WIDTH = 1280;
const HEIGHT = 790;

function sha256(bytes) {
    return createHash("sha256").update(bytes).digest("hex");
}

async function checkedLayout(record) {
    const bytes = await readFile(resolve(ROOT, record.path));
    if (sha256(bytes) !== record.sha256) {
        throw new Error("Layout digest differs from the figure manifest: " + record.path);
    }
    return bytes;
}

async function waitFor(probe, description, seconds = 25) {
    const deadline = Date.now() + 1000 * seconds;
    while (Date.now() < deadline) {
        try {
            const result = await probe();
            if (result) return result;
        } catch (error) {
            if (!/Execution context was destroyed|Cannot find context|Inspected target navigated/.test(error.message)) {
                throw error;
            }
        }
        await sleep(100);
    }
    throw new Error("Timed out waiting for " + description);
}

async function connect({ webSocketDebuggerUrl }) {
    const socket = new WebSocket(webSocketDebuggerUrl);
    await new Promise((resolveOpen, rejectOpen) => {
        socket.addEventListener("open", resolveOpen, { once: true });
        socket.addEventListener("error", () => rejectOpen(new Error("Chrome WebSocket failed")), { once: true });
    });
    const pending = new Map();
    let nextId = 0;
    socket.addEventListener("message", ({ data }) => {
        const message = JSON.parse(data);
        const handler = pending.get(message.id);
        if (!handler) return;
        pending.delete(message.id);
        if (message.error) handler.reject(new Error(message.error.message));
        else handler.resolve(message.result);
    });
    socket.addEventListener("close", () => {
        for (const handler of pending.values()) handler.reject(new Error("Chrome WebSocket closed"));
        pending.clear();
    });
    function send(method, params = {}, timeout = 12000) {
        const id = ++nextId;
        return new Promise((resolveCall, rejectCall) => {
            const timer = setTimeout(() => {
                pending.delete(id);
                rejectCall(new Error("Chrome command timed out: " + method));
            }, timeout);
            pending.set(id, {
                resolve: (value) => { clearTimeout(timer); resolveCall(value); },
                reject: (error) => { clearTimeout(timer); rejectCall(error); },
            });
            socket.send(JSON.stringify({ id, method, params }));
        });
    }
    return {
        send,
        async evaluate(expression, timeout) {
            const reply = await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true }, timeout);
            if (reply.exceptionDetails) {
                throw new Error(reply.exceptionDetails.exception?.description || reply.exceptionDetails.text);
            }
            return reply.result.value;
        },
        close: () => socket.close(),
    };
}

async function captureScenes(cdp, baseURL, manifest, addFrame) {
    await cdp.send("Page.enable");
    await cdp.send("Emulation.setDeviceMetricsOverride",
        { width: WIDTH, height: HEIGHT, deviceScaleFactor: 1, mobile: false });
    await cdp.send("Page.navigate", { url: baseURL + "/viewer_visnotes.html?a=first-fit.json&b=proposal.json" });
    const expected = [
        ["a", "first-fit.json", manifest.layouts.first_fit.used_layer_count],
        ["b", "proposal.json", manifest.layouts.proposal.used_layer_count],
    ];
    await waitFor(() => cdp.evaluate("(" + ((panes, edges) => {
        const format = new Intl.NumberFormat("ja-JP");
        return panes.every(([name, file, layers]) => {
            const pane = document.querySelector("#pane-" + name);
            const layerText = format.format(layers) + " / " + format.format(layers) + " 層";
            const edgeText = format.format(edges) + " 辺 / " + format.format(edges) + " 辺";
            return pane && !pane.hidden && pane.querySelector(".dataset").textContent === file
                && pane.querySelector(".counts").textContent.startsWith(layerText + " · " + edgeText);
        });
    }).toString() + ")(" + JSON.stringify(expected) + "," + manifest.hyperedge_count + ")"),
    "both saved layouts to load");
    const scenes = [];
    const paint = () =>
        cdp.evaluate("new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))");
    async function snapshot(title, note, seconds) {
        await paint();
        const screenshot = await cdp.send("Page.captureScreenshot", { format: "png" });
        const path = "/frames/" + scenes.length + ".png";
        addFrame(path, Buffer.from(screenshot.data, "base64"));
        scenes.push({ path, title, note, seconds });
    }
    await snapshot("Compare two saved layouts",
        "A uses the input-order Qu et al. first-fit rule. B uses the proposed method. Both views share a camera and scale.", 11);
    const result = await cdp.evaluate("(" + ((focus, incidence) => {
        const select = document.querySelector("#node-results");
        const option = [...select.options].find(({ value }) => JSON.parse(value)[0] === focus);
        if (!option) throw new Error("Focus vertex is absent");
        select.value = option.value;
        select.dispatchEvent(new Event("change", { bubbles: true }));
        const detail = document.querySelector("#node-details");
        const membershipLists = [...detail.querySelectorAll(".memberships")];
        if (detail.hidden || membershipLists.length !== 2 ||
            membershipLists.some((list) => list.children.length !== incidence)) {
            throw new Error("Focus interaction did not display the incident edges");
        }
        return detail.querySelector(".selected").textContent;
    }).toString() + ")(" + JSON.stringify(manifest.focus_vertex) + "," + manifest.incident_edge_ids.length + ")");
    await snapshot("Follow a shared vertex",
        result + ". The viewer highlights incident edges and draws the selected vertex as a pillar.", 11);
    const slice = await cdp.evaluate("(" + (() => {
        const upper = document.querySelector("#layer-high");
        const lower = document.querySelector("#layer-low");
        const maximum = upper.valueAsNumber;
        const counts = [...document.querySelectorAll("#pane-a .counts, #pane-b .counts")];
        const before = counts.map((element) => element.textContent);
        upper.value = Math.ceil(maximum * 2 / 3);
        lower.value = Math.floor(maximum / 3);
        lower.dispatchEvent(new Event("change", { bubbles: true }));
        upper.dispatchEvent(new Event("change", { bubbles: true }));
        if (counts.some((element, index) => element.textContent === before[index])) {
            throw new Error("Layer slice did not affect both displayed counts");
        }
        return [lower.valueAsNumber, upper.valueAsNumber];
    }).toString() + ")()");
    await snapshot("Limit the displayed layers",
        "The range control displays layers " + slice.join("–") + " in both arrangements.", 11);
    const drag = await cdp.evaluate("(" + (() => {
        document.querySelector("#all-layers").click();
        const box = document.querySelector("#pane-b .stage canvas").getBoundingClientRect();
        return { x: box.x + box.width * 0.5, y: box.y + box.height * 0.43 };
    }).toString() + ")()");
    async function viewDigest(pane) {
        const clip = await cdp.evaluate("(" + ((id) => {
            const { x, y, width, height } = document.querySelector(id + " .stage canvas").getBoundingClientRect();
            return { x, y, width, height, scale: 1 };
        }).toString() + ")(" + JSON.stringify(pane) + ")");
        const result = await cdp.send("Page.captureScreenshot", { format: "png", clip });
        return sha256(Buffer.from(result.data, "base64"));
    }
    const panes = ["#pane-a", "#pane-b"];
    await paint();
    const before = await Promise.all(panes.map(viewDigest));
    await cdp.send("Input.dispatchMouseEvent", { type: "mousePressed", button: "left", buttons: 1, ...drag, clickCount: 1 });
    for (const offset of [25, 50, 75]) {
        await cdp.send("Input.dispatchMouseEvent", { type: "mouseMoved", button: "none", buttons: 1, x: drag.x + offset, y: drag.y + offset / 5 });
    }
    await cdp.send("Input.dispatchMouseEvent", { type: "mouseReleased", button: "left", buttons: 0, x: drag.x + 75, y: drag.y + 15 });
    await paint();
    const after = await Promise.all(panes.map(viewDigest));
    if (after.some((digest, index) => digest === before[index])) {
        throw new Error("Dragging did not change both comparison views");
    }
    await snapshot("Rotate the shared view",
        "Restoring all layers and dragging the comparison pane rotates both layouts together.", 11);
    return scenes;
}

async function record(scenes, intendedDuration) {
    const images = await Promise.all(scenes.map(async ({ path }) => {
        const clip = new Image();
        clip.src = path;
        await clip.decode();
        return clip;
    }));
    const canvas = document.createElement("canvas");
    canvas.width = 1280;
    canvas.height = 900;
    document.body.append(canvas);
    const painter = canvas.getContext("2d");
    const durationOf = scenes.reduce((sum, scene) => sum + scene.seconds, 0);
    const stream = canvas.captureStream(10);
    const recorder = new MediaRecorder(stream, {
        mimeType: "video/webm;codecs=vp8", videoBitsPerSecond: 1600000,
    });
    const fragments = [];
    const captured = new Promise((resolveCapture, rejectCapture) => {
        recorder.ondataavailable = ({ data }) => { if (data.size) fragments.push(data); };
        recorder.onerror = () => rejectCapture(recorder.error || new Error("Recording failed"));
        recorder.onstop = () => resolveCapture(new Blob(fragments, { type: "video/webm" }));
    });
    function draw(elapsed) {
        let cursor = 0;
        const scaled = elapsed * durationOf / intendedDuration;
        const found = scenes.findIndex(({ seconds }) => {
            cursor += seconds;
            return scaled <= cursor;
        });
        const index = found === -1 ? scenes.length - 1 : found;
        const scene = scenes[index];
        const image = images[index];
        painter.fillStyle = "#101d24";
        painter.fillRect(0, 0, canvas.width, canvas.height);
        const fraction = Math.min(1260 / image.naturalWidth, 780 / image.naturalHeight);
        const width = fraction * image.naturalWidth;
        const height = fraction * image.naturalHeight;
        painter.drawImage(image, (canvas.width - width) / 2, 8 + (780 - height) / 2, width, height);
        painter.fillStyle = "#92d7ce";
        painter.font = "600 25px system-ui";
        painter.fillText(scene.title, 31, 824);
        painter.fillStyle = "#e2eaec";
        painter.font = "18px system-ui";
        painter.fillText(scene.note, 31, 856, 1216);
        painter.fillStyle = "#6db4ae";
        painter.fillRect(31, 879, 1218 * elapsed / intendedDuration, 3);
    }
    draw(0);
    recorder.start();
    const start = performance.now();
    await new Promise((resolveFrames) => {
        function frame() {
            const elapsed = Math.min(intendedDuration, (performance.now() - start) / 1000);
            draw(elapsed);
            if (elapsed < intendedDuration) setTimeout(frame, 100);
            else resolveFrames();
        }
        frame();
    });
    recorder.stop();
    const file = await captured;
    for (const track of stream.getTracks()) track.stop();

    const video = document.createElement("video");
    await new Promise((resolveMetadata, rejectMetadata) => {
        video.onloadeddata = resolveMetadata;
        video.onerror = () => rejectMetadata(new Error("The encoded video did not decode"));
        video.src = URL.createObjectURL(file);
    });
    if (!Number.isFinite(video.duration)) {
        await new Promise((resolveSeek, rejectSeek) => {
            const timer = setTimeout(() => rejectSeek(new Error("Could not seek to the end of the encoded video")), 3000);
            video.onseeked = () => { clearTimeout(timer); resolveSeek(); };
            video.currentTime = 1e6;
        });
    }
    const duration = Number.isFinite(video.duration) ? video.duration : video.currentTime;
    if (Math.abs(duration - intendedDuration) > 1) {
        throw new Error("Decoded duration differs from the requested duration: " + duration);
    }
    const response = await fetch("/recording", { method: "POST", body: file });
    if (!response.ok) throw new Error("The local recording transfer failed");
    return { width: video.videoWidth, height: video.videoHeight, duration, bytes: file.size, codec: recorder.mimeType };
}

async function main() {
    const { values } = parseArgs({ options: {
        "figure-manifest": { type: "string" },
        output: { type: "string" },
        chrome: { type: "string" },
        seconds: { type: "string", default: "60" },
    } });
    const intendedDuration = Number(values.seconds);
    if (!values["figure-manifest"] || !values.output || !Number.isFinite(intendedDuration)
        || intendedDuration < 6 || intendedDuration > 90) {
        throw new Error("Use --figure-manifest FILE --output NEW_DIRECTORY [--seconds 6..90] [--chrome CHROME]");
    }
    if (typeof WebSocket === "undefined") throw new Error("Use Node with --experimental-websocket");

    const manifestPath = resolve(values["figure-manifest"]);
    const manifestBytes = await readFile(manifestPath);
    const manifest = JSON.parse(manifestBytes);
    const teaser = await readFile(join(dirname(manifestPath), "fig1-teaser.png"));
    if (sha256(teaser) !== manifest.figures?.["fig1-teaser.png"]) {
        throw new Error("Teaser image digest differs from the figure manifest");
    }
    const files = new Map([
        ["/viewer_visnotes.html", ["text/html; charset=utf-8", await readFile(join(ROOT, "viewer_visnotes.html"))]],
        ["/viewer_visnotes.js", ["text/javascript; charset=utf-8", await readFile(join(ROOT, "viewer_visnotes.js"))]],
        ["/first-fit.json", ["application/json", await checkedLayout(manifest.layouts.first_fit)]],
        ["/proposal.json", ["application/json", await checkedLayout(manifest.layouts.proposal)]],
        ["/teaser.png", ["image/png", teaser]],
        ["/recorder.html", ["text/html; charset=utf-8", "<!doctype html><title>Local recorder</title>"]],
    ]);
    await mkdir(values.output, { recursive: false });
    let recording;
    const server = createServer(async (request, response) => {
        const path = new URL(request.url, "http://localhost").pathname;
        if (request.method === "POST" && path === "/recording" && !recording) {
            const chunks = [];
            let size = 0;
            for await (const chunk of request) {
                size += chunk.length;
                if (size > LIMIT) { response.writeHead(413).end(); return; }
                chunks.push(chunk);
            }
            recording = Buffer.concat(chunks);
            response.writeHead(201).end();
        } else if (request.method === "GET" && files.has(path)) {
            const [type, content] = files.get(path);
            response.writeHead(200, { "Content-Type": type }).end(content);
        } else {
            response.writeHead(404).end();
        }
    });
    const base = await realpath(tmpdir());
    const profile = await mkdtemp(join(base, "visnotes-chrome-"));
    let browser, cdp;
    try {
        server.listen(0, "127.0.0.1");
        await once(server, "listening");
        const url = "http://127.0.0.1:" + server.address().port;
        const chrome = values.chrome ?? (process.platform === "win32"
            ? join(process.env.ProgramFiles, "Google", "Chrome", "Application", "chrome.exe") : "google-chrome");
        browser = spawn(chrome, [
            "--headless", "--remote-debugging-port=0", "--remote-debugging-address=127.0.0.1",
            "--user-data-dir=" + profile, "--disable-background-timer-throttling",
            "--disable-renderer-backgrounding", "--disable-backgrounding-occluded-windows",
            "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "about:blank",
        ], { windowsHide: true, stdio: "ignore" });
        const launched = once(browser, "spawn");
        const exited = once(browser, "exit");
        exited.catch(() => {});
        await launched;
        const port = await waitFor(async () => {
            try {
                const text = await readFile(join(profile, "DevToolsActivePort"), "utf8");
                return parseInt(text, 10);
            } catch (error) {
                if (error.code !== "ENOENT") throw error;
                return null;
            }
        }, "Chrome debugging endpoint");
        const targets = await fetch("http://127.0.0.1:" + port + "/json/list", { signal: AbortSignal.timeout(5000) });
        cdp = await connect((await targets.json()).find(({ type }) => type === "page"));
        const scenes = await captureScenes(cdp, url, manifest, (name, bytes) => files.set(name, ["image/png", bytes]));
        console.log("Captured the checked comparison, focus, layer range, and rotation states.");
        scenes.unshift({ path: "/teaser.png", title: "VisNotes local comparison",
            note: "Internal draft. The same shared vertex and incident edges appear in the planar, first-fit, and proposed panels.", seconds: 8 });
        scenes.push({ path: "/teaser.png", title: "Trace the same relationships",
            note: "Local case from the existing input. Reading accuracy and task time have yet to be evaluated.", seconds: 8 });
        await cdp.send("Page.navigate", { url: url + "/recorder.html" });
        await waitFor(() => cdp.evaluate("document.title === 'Local recorder'"), "recorder page");
        const measured = await cdp.evaluate("(" + record.toString() + ")(" + JSON.stringify(scenes) + "," + intendedDuration + ")",
            (intendedDuration + 12) * 1000);
        if (recording?.length !== measured.bytes || recording.subarray(0, 4).toString("hex") !== "1a45dfa3") {
            throw new Error("No complete WebM reached the local writer");
        }
        const output = resolve(values.output);
        await writeFile(join(output, "visnotes-demo.webm"), recording, { flag: "wx" });
        await writeFile(join(output, "video-manifest.json"), JSON.stringify({
            figure_manifest: manifestPath,
            figure_manifest_sha256: sha256(manifestBytes),
            viewer_html_sha256: sha256(files.get("/viewer_visnotes.html")[1]),
            viewer_sha256: sha256(files.get("/viewer_visnotes.js")[1]),
            video_sha256: sha256(recording),
            intended_seconds: intendedDuration,
            decoded_seconds: measured.duration,
            width: measured.width,
            height: measured.height,
            codec: measured.codec,
            scenes: scenes.map(({ title, note, seconds }) => ({ title, note, nominal_seconds: seconds })),
        }, null, 2) + "\n", { flag: "wx" });
        console.log("Recorded " + measured.duration.toFixed(2) + " s of local viewer operation at " + output);
        cdp.close();
        browser.kill();
        await exited;
    } finally {
        cdp?.close();
        browser?.kill();
        server.close();
        try {
            const resolvedProfile = await realpath(profile);
            if (dirname(resolvedProfile) === base) {
                await rm(resolvedProfile, { recursive: true, force: true, maxRetries: 3, retryDelay: 200 });
            }
        } catch (error) {
            if (error.code !== "ENOENT") console.warn("Temporary Chrome profile cleanup failed: " + error.message);
        }
    }
}

main().catch((error) => {
    console.error(error.message);
    process.exitCode = 1;
});
