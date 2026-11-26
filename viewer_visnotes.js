const byId = (id) => document.getElementById(id);
const UI = {
    status: byId("status"), panes: byId("panes"),
    low: byId("layer-low"), high: byId("layer-high"), spacing: byId("spacing"),
    spacingValue: byId("spacing-value"), query: byId("node-query"),
    results: byId("node-results"), details: byId("node-details"), clear: byId("clear-node"),
    close: byId("remove-b"), reset: byId("reset-view"), all: byId("all-layers"),
};
const FORMAT = new Intl.NumberFormat("ja-JP", { maximumFractionDigits: 3 });
const PALETTE = ["#3f608f", "#2a788e", "#22a884", "#7ad151", "#fde725"];
const FOCUS = "#f4f4c6";
const LAYER_STEP = 0.01;
const ANGLES = { azimuth: Math.PI / 4, elevation: 35 * Math.PI / 180 };
const EPSILON = 1e-8;

function dataset(payload) {
    if (!Array.isArray(payload?.nodes) || !Array.isArray(payload.hyperedges)) {
        throw new Error("nodes と hyperedges の配列が必要です。");
    }
    const validId = (id) => Number.isSafeInteger(id) || (typeof id === "string" && id.length > 0);
    const labelOf = (item) => {
        if (typeof item.label === "string") return item.label;
        return typeof item.name === "string" ? item.name : String(item.id);
    };
    const nodes = new Map();
    const selectable = new Map();
    for (const node of payload.nodes) {
        if (!validId(node?.id) || !Number.isFinite(node.x) || !Number.isFinite(node.y) || nodes.has(node.id)) {
            throw new Error("頂点 ID の重複または無効な座標があります。");
        }
        const label = labelOf(node);
        const key = JSON.stringify([node.id, label]);
        const entry = { id: node.id, label, key, x: node.x, y: node.y, edges: [] };
        nodes.set(node.id, entry);
        selectable.set(key, entry);
    }
    const edges = [];
    const edgeIds = new Set();
    let min = Infinity;
    let max = -Infinity;
    let maxDegree = 1;
    for (const edge of payload.hyperedges) {
        if (!validId(edge?.id) || edgeIds.has(edge.id) || !Array.isArray(edge.nodes) || !Number.isFinite(edge.layer)) {
            throw new Error("ハイパーエッジの ID、所属頂点、または高さが無効です。");
        }
        edgeIds.add(edge.id);
        const boundary = edge.nodes.map((id) => nodes.get(id));
        if (boundary.some((node) => !node)) throw new Error("存在しない頂点への参照があります。");
        const members = Array.from(new Set(boundary));
        const rounded = Math.round(edge.layer);
        const level = Math.abs(edge.layer - rounded) <= EPSILON ? rounded : edge.layer;
        const entry = { label: labelOf(edge), level, members, boundary: members.length > 2 ? boundary : members };
        edges.push(entry);
        for (const node of members) node.edges.push(entry);
        min = Math.min(min, level);
        max = Math.max(max, level);
        maxDegree = Math.max(maxDegree, members.length);
    }
    if (!edges.length) min = max = 0;
    if (!Number.isFinite(max - min)) throw new Error("配置の高さ範囲が大きすぎます。");
    for (const edge of edges) edge.height = edge.level - min;
    return {
        nodes, selectable, edges, maxHeight: max - min, maxDegree,
        levels: new Set(edges.map((edge) => edge.height)),
        integerLevels: edges.every((edge) => Number.isInteger(edge.level)),
    };
}

function notify(message, isError = false) {
    UI.status.textContent = message;
    UI.status.classList.toggle("error", isError);
}

function boundsOf(data) {
    const bounds = { left: Infinity, right: -Infinity, bottom: Infinity, top: -Infinity, degree: 1, height: 0 };
    for (const item of data) {
        bounds.degree = Math.max(bounds.degree, item.maxDegree);
        bounds.height = Math.max(bounds.height, item.maxHeight);
        for (const point of item.nodes.values()) {
            bounds.left = Math.min(bounds.left, point.x);
            bounds.right = Math.max(bounds.right, point.x);
            bounds.bottom = Math.min(bounds.bottom, point.y);
            bounds.top = Math.max(bounds.top, point.y);
        }
    }
    if (!Number.isFinite(bounds.left)) bounds.left = bounds.right = bounds.bottom = bounds.top = 0;
    const width = bounds.right - bounds.left;
    const height = bounds.top - bounds.bottom;
    if (!Number.isFinite(Math.max(width, height))) throw new Error("XY 座標の範囲が大きすぎます。");
    return {
        ...bounds,
        midX: bounds.left + width / 2,
        midY: bounds.bottom + height / 2,
        span: Math.max(width, height) || 1,
    };
}

function orientation(a, b, c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

function crossing(a, b, c, d) {
    if (Math.max(a.x, b.x) < Math.min(c.x, d.x) || Math.max(c.x, d.x) < Math.min(a.x, b.x)
        || Math.max(a.y, b.y) < Math.min(c.y, d.y) || Math.max(c.y, d.y) < Math.min(a.y, b.y)) {
        return false;
    }
    return orientation(a, b, c) * orientation(a, b, d) <= 0
        && orientation(c, d, a) * orientation(c, d, b) <= 0;
}

function simpleRing(points) {
    const size = points.length;
    for (let i = 0; i < size; i++) {
        for (let j = i + 2; j < size; j++) {
            if (i === 0 && j === size - 1) continue;
            if (crossing(points[i], points[(i + 1) % size], points[j], points[(j + 1) % size])) return false;
        }
    }
    return true;
}

function makeLine(THREE, points, color, opacity, closed = false) {
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false });
    return closed ? new THREE.LineLoop(geometry, material) : new THREE.Line(geometry, material);
}

function makeMesh(THREE, geometry, color, opacity) {
    const material = new THREE.MeshBasicMaterial({
        color, transparent: true, opacity, side: THREE.DoubleSide, depthWrite: false, forceSinglePass: true,
    });
    return new THREE.Mesh(geometry, material);
}

function triangleGeometry(THREE, points, triangles) {
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    geometry.setIndex(triangles.flat());
    return geometry;
}

function glyphs(THREE, edge, xy, color) {
    const points = edge.boundary.map((node) => new THREE.Vector3(...xy(node), 0));
    if (!points.length) return { objects: [], outlineOnly: false };
    if (points.length === 1 || (points.length === 2 && points[0].equals(points[1]))) {
        const marker = makeMesh(THREE, new THREE.RingGeometry(0.002, 0.005, 14), color, 0.85);
        marker.position.x = points[0].x;
        marker.position.y = points[0].y;
        return { objects: [marker], outlineOnly: false };
    }
    if (points.length === 2) {
        const [a, b] = points;
        const normal = new THREE.Vector3(a.y - b.y, b.x - a.x, 0).normalize().multiplyScalar(0.0019);
        const vertices = [a.clone().add(normal), b.clone().add(normal), b.clone().sub(normal), a.clone().sub(normal)];
        const ribbon = makeMesh(THREE, triangleGeometry(THREE, vertices, [[0, 1, 2], [0, 2, 3]]), color, 0.85);
        return { objects: [ribbon], outlineOnly: false };
    }
    const ring = points.filter((point, i) => i === 0 || !point.equals(points[i - 1]));
    if (ring.length > 1 && ring.at(-1).equals(ring[0])) ring.pop();
    const outline = makeLine(THREE, points, color, 0.8, true);
    const polygon = ring.map(({ x, y }) => new THREE.Vector2(x, y));
    if (ring.length >= 3 && THREE.ShapeUtils.area(polygon) !== 0 && simpleRing(ring)) {
        const triangles = THREE.ShapeUtils.triangulateShape(polygon, []);
        if (triangles.length) {
            const fill = makeMesh(THREE, triangleGeometry(THREE, ring, triangles), color, 0.39);
            return { objects: [fill, outline], outlineOnly: false };
        }
    }
    return { objects: [outline], outlineOnly: true };
}

function release(group) {
    group.traverse((object) => {
        object.geometry?.dispose();
        object.material?.dispose();
    });
    group.clear();
}

class View {
    constructor(THREE, OrbitControls, slot, changeCamera, choose) {
        this.THREE = THREE;
        this.slot = slot;
        this.element = byId("pane-" + slot);
        this.stage = this.element.querySelector(".stage");
        this.overview = this.element.querySelector(".overview");
        this.context = this.overview.getContext("2d");
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
        this.renderer.setClearColor("#0e151c");
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        this.renderer.domElement.setAttribute("aria-label", "配置 " + slot.toUpperCase() + " の層配置図");
        this.stage.append(this.renderer.domElement);
        this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.001, 10);
        this.camera.up.set(0, 0, 1);
        this.camera.position.set(1, 1, 1);
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = false;
        this.controls.minZoom = 0.1;
        this.controls.maxZoom = 80;
        this.controls.addEventListener("change", () => changeCamera(this));
        this.scene = new THREE.Scene();
        this.objects = new THREE.Group();
        this.highlight = new THREE.Group();
        this.scene.add(this.objects, this.highlight);
        this.entries = [];
        this.points = [];
        this.levelPoints = [];
        this.revision = 0;
        this.data = null;
        this.loading = false;
        this.error = "";

        const canvas = this.renderer.domElement;
        let press = null;
        canvas.addEventListener("pointerdown", (event) => {
            if (event.isPrimary && event.button === 0) press = { id: event.pointerId, x: event.clientX, y: event.clientY };
            else press = null;
        });
        canvas.addEventListener("pointermove", (event) => {
            if (press?.id === event.pointerId && Math.hypot(event.clientX - press.x, event.clientY - press.y) > 4) press = null;
        });
        canvas.addEventListener("pointercancel", () => { press = null; });
        canvas.addEventListener("pointerup", (event) => {
            if (!press || press.id !== event.pointerId) return;
            const distance = Math.hypot(event.clientX - press.x, event.clientY - press.y);
            press = null;
            if (distance <= 4) {
                const node = this.pick(event);
                if (node) choose(node.key);
            }
        });
        this.overview.addEventListener("click", (event) => {
            const rect = this.overview.getBoundingClientRect();
            const node = this.closest(this.mapPoints ?? [], event.clientX - rect.left, event.clientY - rect.top, 10);
            if (node) choose(node.key);
        });
    }

    setDataset(data, name) {
        this.data = data;
        this.element.querySelector(".dataset").textContent = name;
        this.element.hidden = this.slot === "b" && !data;
    }

    rebuild(bounds, palette) {
        release(this.objects);
        release(this.highlight);
        this.entries = [];
        this.points = [];
        this.levelPoints = [];
        this.mapPoints = [];
        this.outlines = 0;
        if (!this.data) {
            this.element.querySelector(".counts").textContent = "";
            return;
        }
        this.xy = (node) => [(node.x - bounds.midX) / bounds.span, (node.y - bounds.midY) / bounds.span];
        const heights = Array.from(this.data.levels).sort((a, b) => a - b);
        const ranks = new Map(heights.map((height, index) => [height, index]));
        const { THREE } = this;
        for (const edge of this.data.edges) {
            const fraction = bounds.degree > 1 ? Math.log(Math.max(edge.members.length, 1)) / Math.log(bounds.degree) : 0;
            const scaled = fraction * (palette.length - 1);
            const index = Math.floor(scaled);
            const color = palette[index].clone().lerp(palette[Math.min(index + 1, palette.length - 1)], scaled - index);
            const { objects, outlineOnly } = glyphs(THREE, edge, this.xy, color);
            for (const object of objects) this.objects.add(object);
            this.entries.push({ edge, rank: ranks.get(edge.height), objects });
            this.outlines += Number(outlineOnly);
        }
        const pointsByHeight = new Map();
        for (const node of this.data.nodes.values()) {
            const [x, y] = this.xy(node);
            const occupied = new Set(node.edges.map((edge) => edge.height));
            if (!occupied.size) occupied.add(0);
            for (const height of occupied) {
                const vertex = { node, x, y, height };
                this.points.push(vertex);
                if (!pointsByHeight.has(height)) pointsByHeight.set(height, []);
                pointsByHeight.get(height).push(vertex);
            }
        }
        for (const [height, vertices] of pointsByHeight) {
            const geometry = new THREE.BufferGeometry().setFromPoints(vertices.map(({ x, y }) => new THREE.Vector3(x, y, 0)));
            const material = new THREE.PointsMaterial({ color: "#b0c4d1", size: 2, sizeAttenuation: false, transparent: true, opacity: 0.64, depthWrite: false });
            const mesh = new THREE.Points(geometry, material);
            this.objects.add(mesh);
            this.levelPoints.push({ height, rank: ranks.get(height) ?? 0, mesh });
        }
    }

    size(unitsPerPixel) {
        const { clientWidth: width, clientHeight: height } = this.stage;
        if (!width || !height) return;
        this.renderer.setSize(width, height, false);
        Object.assign(this.camera, {
            left: -width * unitsPerPixel / 2, right: width * unitsPerPixel / 2,
            top: height * unitsPerPixel / 2, bottom: -height * unitsPerPixel / 2,
        });
        this.camera.updateProjectionMatrix();
    }

    prepare(state) {
        if (!this.data) return;
        this.state = state;
        const node = this.data.selectable.get(state.selected);
        const step = state.step;
        const selectedEdges = new Set(node?.edges ?? []);
        const visibleEdges = this.entries.filter(({ edge }) => this.visible(edge.height));
        const visibleLevels = new Set(visibleEdges.map(({ edge }) => edge.height));
        const heightUnit = this.data.integerLevels ? " 層" : " 高さ位置";
        this.element.querySelector(".counts").textContent =
            FORMAT.format(visibleLevels.size) + " / " + FORMAT.format(this.data.levels.size) + heightUnit + " · "
            + FORMAT.format(visibleEdges.length) + " 辺 / " + FORMAT.format(this.entries.length) + " 辺"
            + (this.outlines ? " · 輪郭のみ " + FORMAT.format(this.outlines) + " 面" : "");
        for (const { edge, objects } of this.entries) {
            const emphasized = selectedEdges.has(edge);
            for (const object of objects) {
                object.visible = this.visible(edge.height);
                object.position.z = edge.height * step;
                if (object.userData.opacity == null) object.userData.opacity = object.material.opacity;
                const alpha = object.userData.opacity;
                object.material.opacity = node ? (emphasized ? Math.min(1, alpha + 0.33) : alpha * 0.19) : alpha;
            }
        }
        for (const { height, mesh } of this.levelPoints) {
            mesh.visible = this.visible(height);
            mesh.position.z = height * step;
            mesh.material.opacity = node ? 0.12 : 0.64;
        }
        release(this.highlight);
        this.highlight.visible = Boolean(node);
        if (!node) return;
        const [x, y] = this.xy(node);
        const occupied = Array.from(new Set(node.edges.map((edge) => edge.height))).sort((a, b) => a - b);
        if (!occupied.length) occupied.push(0);
        const bottom = Math.max(occupied[0], state.low);
        const top = Math.min(occupied.at(-1), state.high);
        if (top - bottom > EPSILON) {
            const line = makeLine(this.THREE, [new this.THREE.Vector3(x, y, bottom * step), new this.THREE.Vector3(x, y, top * step)], FOCUS, 0.95);
            line.renderOrder = 1e7;
            this.highlight.add(line);
        }
        for (const height of occupied) {
            if (!this.visible(height)) continue;
            const dot = makeMesh(this.THREE, new this.THREE.CircleGeometry(0.006, 16), FOCUS, 1);
            dot.position.set(x, y, height * step);
            dot.renderOrder = 1e7 + 1;
            this.highlight.add(dot);
        }
    }

    visible(height) {
        return height >= this.state.low - EPSILON && height <= this.state.high + EPSILON;
    }

    render() {
        if (!this.data || !this.state || !this.stage.clientWidth) return;
        const direction = this.camera.getWorldDirection(new this.THREE.Vector3());
        const order = (rank) => 3 * (direction.z <= 0 ? rank : this.data.levels.size - rank);
        for (const { rank, objects } of this.entries) {
            objects.forEach((object, index) => { object.renderOrder = order(rank) + index; });
        }
        for (const { rank, mesh } of this.levelPoints) mesh.renderOrder = order(rank) + 2;
        this.renderer.render(this.scene, this.camera);
        this.drawOverview(direction);
    }

    drawOverview(direction) {
        const { clientWidth: width, clientHeight: height } = this.overview;
        if (!width || !height) return;
        const density = Math.min(window.devicePixelRatio || 1, 2);
        const canvasWidth = Math.round(width * density);
        const canvasHeight = Math.round(height * density);
        if (this.overview.width !== canvasWidth) this.overview.width = canvasWidth;
        if (this.overview.height !== canvasHeight) this.overview.height = canvasHeight;
        const ctx = this.context;
        ctx.setTransform(density, 0, 0, density, 0, 0);
        ctx.clearRect(0, 0, width, height);
        const scale = Math.min(width, height) - 18;
        const position = (node) => {
            const [x, y] = this.xy(node);
            return [width / 2 + x * scale, height / 2 - y * scale];
        };
        const node = this.data.selectable.get(this.state.selected);
        const selectedEdges = new Set(node?.edges ?? []);
        for (const { edge } of this.entries) {
            if (!this.visible(edge.height) || edge.members.length < 2) continue;
            ctx.beginPath();
            ctx.moveTo(...position(edge.boundary[0]));
            for (const member of edge.boundary.slice(1)) ctx.lineTo(...position(member));
            if (edge.members.length > 2) ctx.closePath();
            ctx.strokeStyle = selectedEdges.has(edge) ? FOCUS : node ? "#273847" : "#465967";
            ctx.lineWidth = selectedEdges.has(edge) ? 1.5 : 0.7;
            ctx.stroke();
        }
        this.mapPoints = [];
        for (const candidate of this.data.nodes.values()) {
            if (candidate.edges.length ? !candidate.edges.some((edge) => this.visible(edge.height)) : !this.visible(0)) continue;
            const [x, y] = position(candidate);
            this.mapPoints.push({ node: candidate, x, y });
            ctx.fillStyle = candidate === node ? FOCUS : "#879ba9";
            ctx.fillRect(x - 0.8, y - 0.8, 1.6, 1.6);
        }
        if (node) {
            ctx.beginPath();
            ctx.arc(...position(node), 3.8, 0, 2 * Math.PI);
            ctx.strokeStyle = FOCUS;
            ctx.lineWidth = 1.3;
            ctx.stroke();
        }
        const targetX = width / 2 + this.controls.target.x * scale;
        const targetY = height / 2 - this.controls.target.y * scale;
        ctx.beginPath();
        ctx.moveTo(targetX - direction.x * 15, targetY + direction.y * 15);
        ctx.lineTo(targetX, targetY);
        ctx.strokeStyle = "#8ee1de";
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }

    closest(points, x, y, tolerance) {
        let closest = null;
        let distance = tolerance ** 2;
        for (const point of points) {
            const candidate = (point.x - x) ** 2 + (point.y - y) ** 2;
            if (candidate < distance) {
                closest = point.node;
                distance = candidate;
            }
        }
        return closest;
    }

    pick(event) {
        if (!this.data || !this.state) return null;
        const rect = this.renderer.domElement.getBoundingClientRect();
        const vector = new this.THREE.Vector3();
        const projected = [];
        for (const { node, x, y, height } of this.points) {
            if (!this.visible(height)) continue;
            vector.set(x, y, height * this.state.step).project(this.camera);
            if (vector.z < -1 || vector.z > 1) continue;
            projected.push({ node, x: (vector.x + 1) * rect.width / 2, y: (1 - vector.y) * rect.height / 2, z: vector.z });
        }
        projected.sort((a, b) => a.z - b.z);
        return this.closest(projected, event.clientX - rect.left, event.clientY - rect.top, 10);
    }
}

async function launch() {
    const [THREE, { OrbitControls }] = await Promise.all([
        import("three"), import("three/addons/controls/OrbitControls.js"),
    ]);
    const palette = PALETTE.map((color) => new THREE.Color(color));
    let frames = null;
    let syncing = false;
    let bounds = boundsOf([]);
    let selected = null;
    let selectable = new Map();
    let views;

    function visibleViews() { return views.filter((view) => !view.element.hidden); }
    function activeViews() { return visibleViews().filter((view) => view.data); }
    function state() {
        return { selected, low: UI.low.valueAsNumber || 0, high: UI.high.valueAsNumber || 0, step: LAYER_STEP * UI.spacing.valueAsNumber };
    }
    function requestRender() {
        if (frames !== null) return;
        frames = requestAnimationFrame(() => {
            frames = null;
            for (const view of activeViews()) view.render();
        });
    }
    function fit() {
        const panes = visibleViews().filter((view) => view.stage.clientWidth > 0 && view.stage.clientHeight > 0);
        if (!panes.length) return;
        const smallest = Math.min(...panes.map(({ stage }) => Math.min(stage.clientWidth, stage.clientHeight)));
        const diagonal = Math.hypot(1, 1, bounds.height * state().step);
        for (const view of panes) view.size(diagonal * 1.12 / smallest);
        requestRender();
    }
    function syncCamera(source) {
        if (syncing || !views) return;
        syncing = true;
        for (const target of views) {
            if (target === source) continue;
            target.camera.position.copy(source.camera.position);
            target.camera.zoom = source.camera.zoom;
            target.camera.updateProjectionMatrix();
            target.controls.target.copy(source.controls.target);
            target.controls.update();
        }
        syncing = false;
        requestRender();
    }
    function resetView() {
        const totalZ = bounds.height * state().step;
        const distance = Math.max(2, Math.hypot(1, 1, totalZ) * 2);
        syncing = true;
        for (const view of views) {
            const target = view.controls.target.set(0, 0, totalZ / 2);
            view.camera.near = distance / 1000;
            view.camera.far = distance * 8;
            view.camera.position.set(
                target.x + distance * Math.cos(ANGLES.azimuth) * Math.cos(ANGLES.elevation),
                target.y + distance * Math.sin(ANGLES.azimuth) * Math.cos(ANGLES.elevation),
                target.z + distance * Math.sin(ANGLES.elevation),
            );
            view.camera.zoom = 1;
            view.camera.updateProjectionMatrix();
            view.controls.update();
        }
        syncing = false;
        requestRender();
    }
    function updateResults() {
        const needle = UI.query.value.trim().toLocaleLowerCase("ja");
        const options = document.createDocumentFragment();
        for (const node of selectable.values()) {
            const label = node.label + " · ID " + String(node.id);
            if (!label.toLocaleLowerCase("ja").includes(needle) && node.key !== selected) continue;
            options.append(new Option(label, node.key, false, node.key === selected));
        }
        UI.results.replaceChildren(options);
        if (selected === null) UI.results.selectedIndex = -1;
    }
    function updateDetails() {
        const node = selectable.get(selected);
        UI.clear.disabled = !node;
        UI.details.hidden = !node;
        UI.details.replaceChildren();
        if (!node) return;
        const title = document.createElement("h3");
        title.className = "selected";
        title.textContent = node.label + " · ID " + String(node.id);
        UI.details.append(title);
        for (const view of activeViews()) {
            const candidate = view.data.selectable.get(selected);
            const heading = document.createElement("p");
            heading.className = "muted";
            heading.textContent = "配置 " + view.slot.toUpperCase() + " · " + (candidate?.edges.length ?? 0) + " 辺";
            UI.details.append(heading);
            const list = document.createElement("ul");
            list.className = "memberships";
            const edges = Array.from(candidate?.edges ?? []).sort((a, b) => a.height - b.height);
            for (const edge of edges) {
                const item = document.createElement("li");
                const position = view.data.integerLevels ? FORMAT.format(edge.height) + " 層" : "高さ " + FORMAT.format(edge.height);
                item.textContent = position + " · " + edge.label;
                if (!view.visible(edge.height)) item.className = "muted";
                list.append(item);
            }
            if (edges.length) UI.details.append(list);
        }
    }
    function refresh() {
        const settings = state();
        for (const view of activeViews()) view.prepare(settings);
        updateDetails();
        requestRender();
    }
    function choose(key) {
        selected = selectable.has(key) ? key : null;
        updateResults();
        refresh();
    }
    function resetRange() {
        UI.low.value = 0;
        UI.high.value = bounds.height;
    }
    function updateBounds() {
        bounds = boundsOf(activeViews().map((view) => view.data));
        UI.panes.classList.toggle("compare", !views[1].element.hidden);
        UI.close.hidden = views[1].element.hidden;
        resetRange();
        selectable = new Map();
        for (const view of activeViews()) {
            for (const [key, node] of view.data.selectable) {
                if (!selectable.has(key)) selectable.set(key, node);
            }
        }
        if (!selectable.has(selected)) selected = null;
        for (const view of views) view.rebuild(bounds, palette);
        updateResults();
        fit();
        resetView();
        refresh();
    }
    function updateStatus() {
        const errors = views.map((view) => view.error).filter(Boolean);
        const loading = views.filter((view) => view.loading).map((view) => view.slot.toUpperCase());
        const message = [...errors];
        if (loading.length) message.push("配置 " + loading.join("・") + " を読み込み中…");
        notify(message.join(" "), errors.length > 0);
    }
    async function load(view, filename, read) {
        const revision = ++view.revision;
        view.loading = true;
        view.error = "";
        updateStatus();
        try {
            const parsed = dataset(await read());
            if (revision !== view.revision) return;
            boundsOf([parsed, ...views.filter((other) => other !== view && other.data).map((other) => other.data)]);
            view.setDataset(parsed, filename);
            updateBounds();
        } catch (error) {
            if (revision === view.revision) view.error = "配置 " + view.slot.toUpperCase() + " を開けません。 " + error.message;
        } finally {
            if (revision === view.revision) {
                view.loading = false;
                updateStatus();
            }
        }
    }
    function loadURL(view, value) {
        return load(view, value, async () => {
            const url = new URL(value, window.location.href);
            if (url.origin !== window.location.origin || !/^https?:$/.test(url.protocol)) {
                throw new Error("同じ HTTP サーバー上の JSON を指定してください。");
            }
            const response = await fetch(url, { mode: "same-origin", redirect: "error" });
            if (!response.ok) throw new Error("JSON の取得に失敗しました (HTTP " + response.status + ")。");
            return response.json();
        });
    }
    views = ["a", "b"].map((slot) => new View(THREE, OrbitControls, slot, syncCamera, choose));
    const observer = new ResizeObserver(fit);
    for (const view of views) {
        observer.observe(view.stage);
        observer.observe(view.overview);
        const fileInput = byId("file-" + view.slot);
        fileInput.addEventListener("change", () => {
            const file = fileInput.files[0];
            fileInput.value = "";
            if (file) load(view, file.name, async () => JSON.parse(await file.text()));
        });
        fileInput.disabled = false;
    }
    UI.close.addEventListener("click", () => {
        views[1].revision++;
        views[1].loading = false;
        views[1].error = "";
        views[1].setDataset(null, "");
        updateBounds();
        updateStatus();
    });
    UI.low.addEventListener("change", () => {
        UI.low.value = Math.min(Math.max(UI.low.valueAsNumber || 0, 0), UI.high.valueAsNumber || 0);
        refresh();
    });
    UI.high.addEventListener("change", () => {
        UI.high.value = Math.max(Math.min(UI.high.valueAsNumber || 0, bounds.height), UI.low.valueAsNumber || 0);
        refresh();
    });
    UI.all.addEventListener("click", () => { resetRange(); refresh(); });
    UI.spacing.addEventListener("input", () => {
        UI.spacingValue.value = UI.spacing.valueAsNumber.toFixed(1) + "×";
        fit();
        const center = bounds.height * state().step / 2;
        syncing = true;
        for (const view of views) {
            view.camera.position.z += center - view.controls.target.z;
            view.controls.target.z = center;
            const distance = view.camera.position.distanceTo(view.controls.target);
            view.camera.far = distance * 8 + 2 * center;
            view.camera.updateProjectionMatrix();
            view.controls.update();
        }
        syncing = false;
        refresh();
    });
    UI.query.addEventListener("input", updateResults);
    UI.results.addEventListener("change", () => choose(UI.results.value));
    UI.clear.addEventListener("click", () => choose(null));
    UI.reset.addEventListener("click", resetView);
    const params = new URLSearchParams(window.location.search);
    const loads = [loadURL(views[0], params.get("a") || "layered_hypergraph.json")];
    if (params.has("b")) loads.push(loadURL(views[1], params.get("b")));
    await Promise.all(loads);
}

launch().catch((error) => {
    notify("Three.js ビューアを起動できません。ローカル HTTP サーバー、WebGL 2、CDN への接続を確認してください。 " + error.message, true);
});
