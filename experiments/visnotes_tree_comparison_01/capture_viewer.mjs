import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { mkdtemp, readFile, realpath, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../../", import.meta.url));
const [run, output] = process.argv.slice(2).map((argument) => resolve(argument));
assert.ok(run && output, "Pass the run directory and a new PNG path");

async function poll(action) {
    for (let i = 0; i < 100; i += 1) {
        const result = await action();
        if (result) return result;
        await sleep(200);
    }
    throw new Error("Timed out waiting for the viewer");
}

async function anonymized(method) {
    const layout = JSON.parse(await readFile(join(run, method, "seed-0", "layout.json"), "utf-8"));
    const fields = (items, keys, name) => items.map((item) => ({
        ...Object.fromEntries(keys.map((key) => [key, item[key]])), label: name + " " + item.id,
    }));
    return JSON.stringify({
        nodes: fields(layout.nodes, ["id", "x", "y"], "Vertex"),
        hyperedges: fields(layout.hyperedges, ["id", "nodes", "layer"], "Edge"),
    });
}

const summary = JSON.parse(await readFile(join(run, "summary.json"), "utf-8"));
assert.ok(summary.complete);
const baseline = summary.cases.find((record) => record.condition === "tree-squared" && record.seed === 0);
const focus = baseline.metrics.constrained_hubs[0].node_id;
const files = new Map([
    ["/viewer_visnotes.html", ["text/html; charset=utf-8", await readFile(join(root, "viewer_visnotes.html"))]],
    ["/viewer_visnotes.js", ["text/javascript", await readFile(join(root, "viewer_visnotes.js"))]],
    ["/Tree-SA.json", ["application/json", await anonymized("tree-squared")]],
    ["/Compact-hybrid.json", ["application/json", await anonymized("compact-hybrid")]],
]);
const server = createServer((request, response) => {
    const file = files.get(new URL(request.url, "http://127.0.0.1").pathname);
    if (file) response.writeHead(200, { "Content-Type": file[0] }).end(file[1]);
    else response.writeHead(404).end();
});
const temp = await realpath(tmpdir());
const profile = await mkdtemp(join(temp, "visnotes-tree-chrome-"));
let browser, socket;
try {
    server.listen(0, "127.0.0.1");
    await once(server, "listening");
    const chrome = join(process.env.ProgramFiles, "Google", "Chrome", "Application", "chrome.exe");
    browser = spawn(chrome, [
        "--headless=new", "--remote-debugging-port=0", "--remote-debugging-address=127.0.0.1",
        "--disable-background-networking", "--disable-background-timer-throttling",
        "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
        "--user-data-dir=" + profile, "about:blank",
    ], { windowsHide: true, stdio: "ignore" });
    const port = await poll(() => readFile(join(profile, "DevToolsActivePort"), "utf-8").catch(() => null));
    const tabs = await fetch("http://127.0.0.1:" + port.split("\n")[0] + "/json/list").then((response) => response.json());
    socket = new WebSocket(tabs.find((tab) => tab.type === "page").webSocketDebuggerUrl);
    await once(socket, "open");
    let nextId = 0;
    function send(method, params = {}) {
        const id = ++nextId;
        return new Promise((done, fail) => {
            const timer = setTimeout(() => {
                socket.removeEventListener("message", onMessage);
                fail(new Error(method + " timed out"));
            }, 12000);
            function onMessage(event) {
                const message = JSON.parse(event.data);
                if (message.id !== id) return;
                clearTimeout(timer);
                socket.removeEventListener("message", onMessage);
                if (message.error) fail(new Error(message.error.message));
                else done(message.result);
            }
            socket.addEventListener("message", onMessage);
            socket.send(JSON.stringify({ id, method, params }));
        });
    }
    async function evaluate(expression) {
        const reply = await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
        if (reply.exceptionDetails) throw new Error(reply.exceptionDetails.text);
        return reply.result.value;
    }
    await send("Page.enable");
    await send("Emulation.setDeviceMetricsOverride", { width: 1440, height: 1040, deviceScaleFactor: 1, mobile: false });
    const url = "http://127.0.0.1:" + server.address().port + "/viewer_visnotes.html?a=Tree-SA.json&b=Compact-hybrid.json";
    await send("Page.navigate", { url });
    await poll(() => evaluate(`["a","b"].every(side => document.querySelector("#pane-"+side+" .counts")?.textContent.includes("383"))`));
    const selected = await evaluate("(" + ((id) => {
        const select = document.getElementById("node-results");
        const match = [...select.options].find((option) => JSON.parse(option.value)[0] === id);
        if (!match) return false;
        select.value = match.value;
        select.dispatchEvent(new Event("change", { bubbles: true }));
        const details = document.getElementById("node-details");
        return !details.hidden && details.querySelectorAll(".memberships").length === 2;
    }).toString() + ")(" + JSON.stringify(focus) + ")");
    assert.ok(selected);
    const bounds = await evaluate(`(() => {
        const canvas = document.querySelector("#pane-a .stage canvas").getBoundingClientRect();
        return { x: canvas.x + canvas.width / 2, y: canvas.y + canvas.height / 2 };
    })()`);
    await send("Input.dispatchMouseEvent", { type: "mouseWheel", ...bounds, deltaX: 0, deltaY: -1500 });
    await evaluate("new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done)))");
    const screenshot = await send("Page.captureScreenshot", { format: "png" });
    await writeFile(output, Buffer.from(screenshot.data, "base64"), { flag: "wx" });
    console.log("Saved the synchronized viewer comparison for seed 0, vertex " + focus + ".");
} finally {
    socket?.close();
    browser?.kill();
    server.close();
    const resolved = await realpath(profile).catch(() => null);
    if (resolved && dirname(resolved) === temp) {
        await rm(resolved, { recursive: true, force: true, maxRetries: 3, retryDelay: 200 }).catch(() => {});
    }
}
