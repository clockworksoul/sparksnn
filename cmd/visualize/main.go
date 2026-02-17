// Command visualize runs a C. elegans network simulation and generates
// a self-contained HTML file that animates signal propagation through
// the connectome.
//
// Usage:
//
//	go run ./cmd/visualize -stimulus PLML,PLMR -ticks 30 -o sim.html
//	open sim.html
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"strings"

	"github.com/clockworksoul/biomimetic-network/celegans"
)

// SimData is the complete simulation output embedded in the HTML.
type SimData struct {
	Neurons     []NeuronInfo   `json:"neurons"`
	Connections []ConnInfo     `json:"connections"`
	Ticks       []TickSnapshot `json:"ticks"`
	Stimulus    []string       `json:"stimulus"`
	Params      ParamInfo      `json:"params"`
}

type NeuronInfo struct {
	Name  string  `json:"name"`
	X     float64 `json:"x"`
	Y     float64 `json:"y"`
	Class string  `json:"class"` // sensory, motor_a, motor_b, inter, command, gaba
}

type ConnInfo struct {
	From   int  `json:"from"`
	To     int  `json:"to"`
	Weight int  `json:"weight"`
	IsGap  bool `json:"isGap"`
}

type TickSnapshot struct {
	Tick        int            `json:"tick"`
	Activations []int32        `json:"activations"`
	Fired       []int          `json:"fired"` // indices of neurons that fired
	TotalFired  int            `json:"totalFired"`
	Pending     int            `json:"pending"`
}

type ParamInfo struct {
	Threshold      int32  `json:"threshold"`
	Baseline       int32  `json:"baseline"`
	RefractoryPeriod uint32 `json:"refractoryPeriod"`
}

// classifyNeuron returns a class string for visualization coloring.
func classifyNeuron(name string) string {
	// Sensory neurons
	sensory := map[string]bool{
		"PLML": true, "PLMR": true, "ALML": true, "ALMR": true,
		"AVM": true, "PVM": true, "PVDL": true, "PVDR": true,
		"FLPL": true, "FLPR": true, "ASHL": true, "ASHR": true,
		"ADEL": true, "ADER": true, "PDEL": true, "PDER": true,
		"ASEL": true, "ASER": true, "AWCL": true, "AWCR": true,
		"AWAL": true, "AWAR": true, "AFDL": true, "AFDR": true,
		"ASKL": true, "ASKR": true, "ADLL": true, "ADLR": true,
		"ADFL": true, "ADFR": true, "CEPDL": true, "CEPDR": true,
		"CEPVL": true, "CEPVR": true, "OLLL": true, "OLLR": true,
		"OLQDL": true, "OLQDR": true, "OLQVL": true, "OLQVR": true,
		"IL1DL": true, "IL1DR": true, "IL1L": true, "IL1R": true,
		"IL1VL": true, "IL1VR": true, "IL2DL": true, "IL2DR": true,
		"IL2L": true, "IL2R": true, "IL2VL": true, "IL2VR": true,
		"BAGL": true, "BAGR": true, "URXL": true, "URXR": true,
		"AIAL": true, "AIAR": true, "AIBL": true, "AIBR": true,
		"AINL": true, "AINR": true, "AIYL": true, "AIYR": true,
		"AIZL": true, "AIZR": true, "PHAL": true, "PHAR": true,
		"PHBL": true, "PHBR": true, "PHCL": true, "PHCR": true,
	}

	// Command interneurons
	command := map[string]bool{
		"AVAL": true, "AVAR": true, "AVBL": true, "AVBR": true,
		"AVDL": true, "AVDR": true, "AVEL": true, "AVER": true,
		"PVCL": true, "PVCR": true,
	}

	// GABA interneurons
	if celegans.IsGABANeuron(name) {
		return "gaba"
	}
	if command[name] {
		return "command"
	}
	if sensory[name] {
		return "sensory"
	}
	// B-class motors (forward)
	if strings.HasPrefix(name, "DB") || strings.HasPrefix(name, "VB") {
		return "motor_b"
	}
	// A-class motors (backward)
	if strings.HasPrefix(name, "DA") || strings.HasPrefix(name, "VA") {
		return "motor_a"
	}
	// D-type motors (inhibitory, but caught by GABA above)
	if strings.HasPrefix(name, "DD") || strings.HasPrefix(name, "VD") {
		return "gaba"
	}
	return "inter"
}

// layoutNeurons assigns 2D positions using a simple anatomical layout.
// Neurons are positioned along the anterior-posterior axis (x) and
// dorsal-ventral axis (y) based on their class and name.
func layoutNeurons(names []string) map[string][2]float64 {
	pos := make(map[string][2]float64, len(names))

	// Group neurons by class for structured layout
	// Use a circular layout with class-based clustering
	classGroups := make(map[string][]string)
	for _, name := range names {
		c := classifyNeuron(name)
		classGroups[c] = append(classGroups[c], name)
	}

	// Arrange in concentric rings:
	// Inner: command interneurons
	// Middle: regular interneurons + sensory
	// Outer: motor neurons + GABA
	ringConfig := map[string]struct {
		radius float64
		offset float64 // angular offset in radians
	}{
		"command": {80, 0},
		"sensory": {200, 0.1},
		"inter":   {300, 0.2},
		"motor_b": {420, 0.3},
		"motor_a": {420, math.Pi + 0.3}, // opposite side from B-class
		"gaba":    {350, math.Pi / 2},
	}

	for class, neurons := range classGroups {
		cfg, ok := ringConfig[class]
		if !ok {
			cfg = ringConfig["inter"]
		}
		n := len(neurons)
		// Sort for determinism
		sortStrings(neurons)
		for i, name := range neurons {
			angle := cfg.offset + 2*math.Pi*float64(i)/float64(n)
			pos[name] = [2]float64{
				500 + cfg.radius*math.Cos(angle),
				500 + cfg.radius*math.Sin(angle),
			}
		}
	}

	return pos
}

func sortStrings(s []string) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j] < s[j-1]; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}

func main() {
	stimFlag := flag.String("stimulus", "PLML,PLMR", "comma-separated neuron names to stimulate")
	stimWeight := flag.Int("weight", 5000, "stimulus weight (int32)")
	ticks := flag.Int("ticks", 40, "number of ticks to simulate")
	output := flag.String("o", "sim.html", "output HTML file")
	flag.Parse()

	// Load connectome
	records, err := celegans.LoadCSV("data/celegans_connectome.csv")
	if err != nil {
		log.Fatalf("LoadCSV: %v", err)
	}

	params := celegans.DefaultParams()
	net, nameMap := celegans.BuildNetwork(records, params)

	// Build reverse map
	indexToName := make(map[uint32]string, len(nameMap))
	for name, idx := range nameMap {
		indexToName[idx] = name
	}

	// Build sorted name list (matching index order)
	names := make([]string, len(net.Neurons))
	for name, idx := range nameMap {
		names[idx] = name
	}

	// Layout
	positions := layoutNeurons(names)

	// Build neuron info
	neuronInfos := make([]NeuronInfo, len(names))
	for i, name := range names {
		p := positions[name]
		neuronInfos[i] = NeuronInfo{
			Name:  name,
			X:     p[0],
			Y:     p[1],
			Class: classifyNeuron(name),
		}
	}

	// Build connection info
	var connInfos []ConnInfo
	for i, n := range net.Neurons {
		for _, c := range n.Connections {
			connInfos = append(connInfos, ConnInfo{
				From:   i,
				To:     int(c.Target),
				Weight: int(c.Weight),
				IsGap:  false, // we could track this but it's not critical for viz
			})
		}
	}

	// Stimulate
	stimNames := strings.Split(*stimFlag, ",")
	for _, name := range stimNames {
		name = strings.TrimSpace(name)
		idx, ok := nameMap[name]
		if !ok {
			log.Fatalf("unknown neuron: %s", name)
		}
		net.Stimulate(idx, int32(*stimWeight))
	}

	// Run simulation, capture each tick
	snapshots := make([]TickSnapshot, 0, *ticks+1)

	// Tick 0: initial state after stimulus
	snap0 := TickSnapshot{
		Tick:        0,
		Activations: make([]int32, len(net.Neurons)),
		Pending:     net.Pending(),
	}
	for i, n := range net.Neurons {
		snap0.Activations[i] = n.Activation
	}
	// Find neurons that fired from direct stimulus
	for i, n := range net.Neurons {
		if n.LastFired > 0 {
			snap0.Fired = append(snap0.Fired, i)
			snap0.TotalFired++
		}
	}
	snapshots = append(snapshots, snap0)

	for t := 0; t < *ticks; t++ {
		fired := net.Tick()

		snap := TickSnapshot{
			Tick:        t + 1,
			Activations: make([]int32, len(net.Neurons)),
			TotalFired:  fired,
			Pending:     net.Pending(),
		}
		for i, n := range net.Neurons {
			snap.Activations[i] = n.Activation
			if n.LastFired == net.Counter {
				snap.Fired = append(snap.Fired, i)
			}
		}
		snapshots = append(snapshots, snap)
	}

	simData := SimData{
		Neurons:     neuronInfos,
		Connections: connInfos,
		Ticks:       snapshots,
		Stimulus:    stimNames,
		Params: ParamInfo{
			Threshold:        params.Threshold,
			Baseline:         params.Baseline,
			RefractoryPeriod: params.RefractoryPeriod,
		},
	}

	dataJSON, err := json.Marshal(simData)
	if err != nil {
		log.Fatalf("json marshal: %v", err)
	}

	html := generateHTML(dataJSON)

	if err := os.WriteFile(*output, []byte(html), 0644); err != nil {
		log.Fatalf("write %s: %v", *output, err)
	}

	fmt.Printf("Wrote %s (%d neurons, %d connections, %d ticks)\n",
		*output, len(names), len(connInfos), len(snapshots))
	fmt.Printf("Stimulus: %v (weight=%d)\n", stimNames, *stimWeight)
}

func generateHTML(dataJSON []byte) string {
	return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>C. elegans Connectome — Biomimetic Network Visualization</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    background: #0a0a0f;
    color: #ccc;
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    overflow: hidden;
}
#container { display: flex; height: 100vh; }
#canvas-wrap { flex: 1; position: relative; }
canvas { display: block; width: 100%; height: 100%; }
#sidebar {
    width: 320px;
    background: #111118;
    border-left: 1px solid #222;
    padding: 16px;
    overflow-y: auto;
    font-size: 13px;
}
h1 { font-size: 16px; color: #9b59b6; margin-bottom: 8px; }
h2 { font-size: 13px; color: #888; margin: 12px 0 6px; text-transform: uppercase; letter-spacing: 1px; }
.stat { display: flex; justify-content: space-between; padding: 2px 0; }
.stat-label { color: #666; }
.stat-value { color: #eee; font-weight: bold; }
.controls { margin: 12px 0; }
.controls button {
    background: #1a1a25;
    color: #9b59b6;
    border: 1px solid #333;
    padding: 6px 14px;
    margin: 2px;
    cursor: pointer;
    border-radius: 4px;
    font-family: inherit;
    font-size: 12px;
}
.controls button:hover { background: #252535; }
.controls button.active { background: #9b59b6; color: #fff; }
input[type="range"] { width: 100%; margin: 4px 0; accent-color: #9b59b6; }
.legend { margin: 8px 0; }
.legend-item { display: flex; align-items: center; padding: 2px 0; }
.legend-dot {
    width: 10px; height: 10px; border-radius: 50%;
    margin-right: 8px; flex-shrink: 0;
}
#motor-bars { margin: 8px 0; }
.bar-container { margin: 4px 0; }
.bar-label { font-size: 11px; color: #888; margin-bottom: 2px; }
.bar-track { height: 14px; background: #1a1a25; border-radius: 3px; overflow: hidden; position: relative; }
.bar-fill { height: 100%; border-radius: 3px; transition: width 0.15s; }
.bar-fill.forward { background: #2ecc71; }
.bar-fill.backward { background: #e74c3c; }
#tooltip {
    position: absolute; display: none; background: #1a1a25;
    border: 1px solid #333; padding: 8px 12px; border-radius: 6px;
    font-size: 12px; pointer-events: none; z-index: 10;
    max-width: 250px;
}
#tooltip .name { color: #9b59b6; font-weight: bold; }
#tooltip .detail { color: #888; margin-top: 4px; }
#fired-list { max-height: 150px; overflow-y: auto; }
#fired-list div { padding: 1px 0; font-size: 11px; }
</style>
</head>
<body>
<div id="container">
    <div id="canvas-wrap">
        <canvas id="network"></canvas>
        <div id="tooltip"></div>
    </div>
    <div id="sidebar">
        <h1>🟣 C. elegans Connectome</h1>
        <div class="stat"><span class="stat-label">Neurons</span><span class="stat-value" id="s-neurons">—</span></div>
        <div class="stat"><span class="stat-label">Connections</span><span class="stat-value" id="s-conns">—</span></div>
        <div class="stat"><span class="stat-label">Stimulus</span><span class="stat-value" id="s-stim">—</span></div>

        <h2>Playback</h2>
        <div class="controls">
            <button id="btn-play">▶ Play</button>
            <button id="btn-pause">⏸</button>
            <button id="btn-reset">⏮</button>
            <button id="btn-step">⏭ Step</button>
        </div>
        <div class="stat"><span class="stat-label">Tick</span><span class="stat-value" id="s-tick">0</span></div>
        <input type="range" id="tick-slider" min="0" max="0" value="0">
        <div class="stat"><span class="stat-label">Speed</span><span class="stat-value" id="s-speed">1x</span></div>
        <input type="range" id="speed-slider" min="1" max="20" value="4">

        <h2>Activity</h2>
        <div class="stat"><span class="stat-label">Fired this tick</span><span class="stat-value" id="s-fired">0</span></div>
        <div class="stat"><span class="stat-label">Active neurons</span><span class="stat-value" id="s-active">0</span></div>
        <div class="stat"><span class="stat-label">Pending signals</span><span class="stat-value" id="s-pending">0</span></div>

        <h2>Motor Output</h2>
        <div id="motor-bars">
            <div class="bar-container">
                <div class="bar-label">Forward (B-class) <span id="b-count">0</span></div>
                <div class="bar-track"><div class="bar-fill forward" id="b-bar"></div></div>
            </div>
            <div class="bar-container">
                <div class="bar-label">Backward (A-class) <span id="a-count">0</span></div>
                <div class="bar-track"><div class="bar-fill backward" id="a-bar"></div></div>
            </div>
        </div>

        <h2>Legend</h2>
        <div class="legend">
            <div class="legend-item"><div class="legend-dot" style="background:#e74c3c"></div>Sensory</div>
            <div class="legend-item"><div class="legend-dot" style="background:#f1c40f"></div>Command Interneuron</div>
            <div class="legend-item"><div class="legend-dot" style="background:#3498db"></div>Interneuron</div>
            <div class="legend-item"><div class="legend-dot" style="background:#2ecc71"></div>Motor B (forward)</div>
            <div class="legend-item"><div class="legend-dot" style="background:#e67e22"></div>Motor A (backward)</div>
            <div class="legend-item"><div class="legend-dot" style="background:#9b59b6"></div>GABAergic (inhibitory)</div>
        </div>

        <h2>Fired Neurons</h2>
        <div id="fired-list"></div>
    </div>
</div>

<script>
const SIM = ` + string(dataJSON) + `;

const classColors = {
    sensory:  [231, 76, 60],
    command:  [241, 196, 15],
    inter:    [52, 152, 219],
    motor_b:  [46, 204, 113],
    motor_a:  [230, 126, 34],
    gaba:     [155, 89, 182],
};

const canvas = document.getElementById('network');
const ctx = canvas.getContext('2d');
const tooltip = document.getElementById('tooltip');

let dpr = window.devicePixelRatio || 1;
let W, H;
let currentTick = 0;
let playing = false;
let playInterval = null;
let speedMs = 250;
let hoveredNeuron = -1;
let showConnections = true;

// Transform: pan/zoom
let viewX = 0, viewY = 0, viewScale = 1;
let dragging = false, dragStartX, dragStartY, dragViewX, dragViewY;

function resize() {
    const wrap = document.getElementById('canvas-wrap');
    W = wrap.clientWidth;
    H = wrap.clientHeight;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = W + 'px';
    canvas.style.height = H + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Fit neurons to canvas
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const n of SIM.neurons) {
        minX = Math.min(minX, n.x); maxX = Math.max(maxX, n.x);
        minY = Math.min(minY, n.y); maxY = Math.max(maxY, n.y);
    }
    const pad = 60;
    const scaleX = (W - 2*pad) / (maxX - minX || 1);
    const scaleY = (H - 2*pad) / (maxY - minY || 1);
    viewScale = Math.min(scaleX, scaleY);
    viewX = pad - minX * viewScale + (W - 2*pad - (maxX - minX) * viewScale) / 2;
    viewY = pad - minY * viewScale + (H - 2*pad - (maxY - minY) * viewScale) / 2;

    draw();
}

function toScreen(x, y) {
    return [x * viewScale + viewX, y * viewScale + viewY];
}

function fromScreen(sx, sy) {
    return [(sx - viewX) / viewScale, (sy - viewY) / viewScale];
}

function draw() {
    ctx.clearRect(0, 0, W, H);

    const tick = SIM.ticks[currentTick] || SIM.ticks[0];
    const firedSet = new Set(tick.fired || []);

    // Draw connections (dimmed)
    if (showConnections) {
        ctx.globalAlpha = 0.04;
        ctx.lineWidth = 0.5;
        for (const c of SIM.connections) {
            const from = SIM.neurons[c.from];
            const to = SIM.neurons[c.to];
            const [x1, y1] = toScreen(from.x, from.y);
            const [x2, y2] = toScreen(to.x, to.y);
            ctx.strokeStyle = c.weight < 0 ? '#9b59b6' : '#445';
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
        }
        ctx.globalAlpha = 1;
    }

    // Highlight connections for hovered neuron
    if (hoveredNeuron >= 0) {
        ctx.lineWidth = 1;
        for (const c of SIM.connections) {
            if (c.from !== hoveredNeuron && c.to !== hoveredNeuron) continue;
            const from = SIM.neurons[c.from];
            const to = SIM.neurons[c.to];
            const [x1, y1] = toScreen(from.x, from.y);
            const [x2, y2] = toScreen(to.x, to.y);
            const outgoing = c.from === hoveredNeuron;
            ctx.globalAlpha = 0.5;
            ctx.strokeStyle = c.weight < 0 ? '#9b59b6' : (outgoing ? '#2ecc71' : '#3498db');
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
        }
        ctx.globalAlpha = 1;
    }

    // Draw fired connections (bright)
    if (firedSet.size > 0) {
        ctx.lineWidth = 1.5;
        ctx.globalAlpha = 0.3;
        for (const c of SIM.connections) {
            if (!firedSet.has(c.from)) continue;
            const from = SIM.neurons[c.from];
            const to = SIM.neurons[c.to];
            const [x1, y1] = toScreen(from.x, from.y);
            const [x2, y2] = toScreen(to.x, to.y);
            ctx.strokeStyle = c.weight < 0 ? '#c0f' : '#ff0';
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
        }
        ctx.globalAlpha = 1;
    }

    // Draw neurons
    for (let i = 0; i < SIM.neurons.length; i++) {
        const n = SIM.neurons[i];
        const [sx, sy] = toScreen(n.x, n.y);
        const act = tick.activations[i];
        const fired = firedSet.has(i);
        const rgb = classColors[n.class] || [100, 100, 100];

        // Size: base 3, bigger if active
        let r = 3;
        if (fired) r = 8;
        else if (act > 0) r = 3 + Math.min(act / 200, 4);

        // Glow for fired neurons
        if (fired) {
            ctx.beginPath();
            ctx.arc(sx, sy, r + 8, 0, Math.PI * 2);
            const glow = ctx.createRadialGradient(sx, sy, r, sx, sy, r + 8);
            glow.addColorStop(0, 'rgba(' + rgb.join(',') + ', 0.6)');
            glow.addColorStop(1, 'rgba(' + rgb.join(',') + ', 0)');
            ctx.fillStyle = glow;
            ctx.fill();
        }

        // Neuron circle
        ctx.beginPath();
        ctx.arc(sx, sy, r, 0, Math.PI * 2);

        let brightness = 0.3;
        if (fired) brightness = 1.0;
        else if (act > 0) brightness = 0.3 + Math.min(act / 500, 0.7);
        else if (act < 0) brightness = 0.15;

        ctx.fillStyle = 'rgba(' + rgb.map(c => Math.round(c * brightness)).join(',') + ', 1)';
        ctx.fill();

        // Border for hovered
        if (i === hoveredNeuron) {
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            ctx.stroke();
        }

        // Label for fired or hovered neurons
        if (fired || i === hoveredNeuron) {
            ctx.fillStyle = '#fff';
            ctx.font = '10px SF Mono, Consolas, monospace';
            ctx.textAlign = 'center';
            ctx.fillText(n.name, sx, sy - r - 4);
        }
    }

    updateSidebar(tick, firedSet);
}

function updateSidebar(tick, firedSet) {
    document.getElementById('s-tick').textContent = currentTick + ' / ' + (SIM.ticks.length - 1);
    document.getElementById('s-fired').textContent = tick.totalFired;
    document.getElementById('s-pending').textContent = tick.pending;

    let activeCount = 0;
    for (const a of tick.activations) if (a > 0) activeCount++;
    document.getElementById('s-active').textContent = activeCount + ' / ' + SIM.neurons.length +
        ' (' + Math.round(activeCount / SIM.neurons.length * 100) + '%)';

    // Motor bars
    let bFired = 0, aFired = 0;
    for (const idx of (tick.fired || [])) {
        const cls = SIM.neurons[idx].class;
        if (cls === 'motor_b') bFired++;
        if (cls === 'motor_a') aFired++;
    }
    const maxMotor = Math.max(bFired, aFired, 1);
    document.getElementById('b-bar').style.width = (bFired / 20 * 100) + '%';
    document.getElementById('a-bar').style.width = (aFired / 20 * 100) + '%';
    document.getElementById('b-count').textContent = bFired;
    document.getElementById('a-count').textContent = aFired;

    // Fired list
    const list = document.getElementById('fired-list');
    if (firedSet.size === 0) {
        list.innerHTML = '<div style="color:#555">None</div>';
    } else {
        const names = [...firedSet].map(i => SIM.neurons[i].name).sort();
        list.innerHTML = names.map(n => '<div>' + n + '</div>').join('');
    }
}

// Playback controls
function play() {
    if (playing) return;
    playing = true;
    document.getElementById('btn-play').classList.add('active');
    playInterval = setInterval(() => {
        if (currentTick >= SIM.ticks.length - 1) { pause(); return; }
        currentTick++;
        document.getElementById('tick-slider').value = currentTick;
        draw();
    }, speedMs);
}
function pause() {
    playing = false;
    document.getElementById('btn-play').classList.remove('active');
    clearInterval(playInterval);
}
function reset() {
    pause();
    currentTick = 0;
    document.getElementById('tick-slider').value = 0;
    draw();
}
function step() {
    pause();
    if (currentTick < SIM.ticks.length - 1) {
        currentTick++;
        document.getElementById('tick-slider').value = currentTick;
        draw();
    }
}

document.getElementById('btn-play').onclick = play;
document.getElementById('btn-pause').onclick = pause;
document.getElementById('btn-reset').onclick = reset;
document.getElementById('btn-step').onclick = step;

document.getElementById('tick-slider').max = SIM.ticks.length - 1;
document.getElementById('tick-slider').oninput = function() {
    pause();
    currentTick = parseInt(this.value);
    draw();
};

document.getElementById('speed-slider').oninput = function() {
    const v = parseInt(this.value);
    speedMs = 500 / v;
    document.getElementById('s-speed').textContent = v + 'x';
    if (playing) { pause(); play(); }
};

// Mouse interaction
canvas.addEventListener('mousemove', function(e) {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    if (dragging) {
        viewX = dragViewX + (mx - dragStartX);
        viewY = dragViewY + (my - dragStartY);
        draw();
        return;
    }

    // Find hovered neuron
    let closest = -1, closestDist = 20;
    for (let i = 0; i < SIM.neurons.length; i++) {
        const [sx, sy] = toScreen(SIM.neurons[i].x, SIM.neurons[i].y);
        const d = Math.hypot(mx - sx, my - sy);
        if (d < closestDist) { closest = i; closestDist = d; }
    }

    if (closest !== hoveredNeuron) {
        hoveredNeuron = closest;
        draw();
    }

    if (closest >= 0) {
        const n = SIM.neurons[closest];
        const tick = SIM.ticks[currentTick];
        const act = tick.activations[closest];
        const fired = (tick.fired || []).includes(closest);
        tooltip.style.display = 'block';
        tooltip.style.left = (mx + 15) + 'px';
        tooltip.style.top = (my + 15) + 'px';
        tooltip.innerHTML = '<div class="name">' + n.name + '</div>' +
            '<div class="detail">Class: ' + n.class + '</div>' +
            '<div class="detail">Activation: ' + act + '</div>' +
            '<div class="detail">Fired: ' + (fired ? 'YES' : 'no') + '</div>';
    } else {
        tooltip.style.display = 'none';
    }
});

canvas.addEventListener('mousedown', function(e) {
    dragging = true;
    dragStartX = e.clientX - canvas.getBoundingClientRect().left;
    dragStartY = e.clientY - canvas.getBoundingClientRect().top;
    dragViewX = viewX;
    dragViewY = viewY;
});
canvas.addEventListener('mouseup', () => dragging = false);
canvas.addEventListener('mouseleave', () => { dragging = false; tooltip.style.display = 'none'; });

canvas.addEventListener('wheel', function(e) {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const zoom = e.deltaY < 0 ? 1.1 : 0.9;
    viewX = mx - (mx - viewX) * zoom;
    viewY = my - (my - viewY) * zoom;
    viewScale *= zoom;
    draw();
});

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    if (e.key === ' ') { e.preventDefault(); playing ? pause() : play(); }
    if (e.key === 'ArrowRight') step();
    if (e.key === 'ArrowLeft' && currentTick > 0) { pause(); currentTick--; document.getElementById('tick-slider').value = currentTick; draw(); }
    if (e.key === 'r') reset();
    if (e.key === 'c') { showConnections = !showConnections; draw(); }
});

// Init
document.getElementById('s-neurons').textContent = SIM.neurons.length;
document.getElementById('s-conns').textContent = SIM.connections.length;
document.getElementById('s-stim').textContent = SIM.stimulus.join(', ');

window.addEventListener('resize', resize);
resize();
</script>
</body>
</html>`
}
