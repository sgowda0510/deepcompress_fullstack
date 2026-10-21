// app.js — DeepCompress Frontend Logic

let selectedDataset = 'cifar10';
let pollTimer = null;
let charts = {};

// ── INIT ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    loadDatasets();
    setupChips();
    checkStatus();
});

// ── PAGE NAVIGATION ───────────────────────────────────────────────────────────
function showPage(name) {
    const pages = ['dashboard', 'insert', 'models', 'about'];
    pages.forEach(p => {
        const el = document.getElementById('page-' + p);
        if (el) el.style.display = (p === name) ? '' : 'none';
    });
    document.querySelectorAll('.nav-tab').forEach((t, i) => {
        t.classList.toggle('active', pages[i] === name);
    });
    if (name === 'models') loadModelsList();
    if (name === 'insert') loadCustomDatasetsList();
}

// ── DATASET SELECTION ─────────────────────────────────────────────────────────
async function loadDatasets() {
    const res  = await fetch('/api/datasets');
    const data = await res.json();
    const grid = document.getElementById('dataset-grid');
    grid.innerHTML = data.map(d => `
        <div class="ds-card ${d.id===selectedDataset?'selected':''}" data-id="${d.id}"
             onclick="selectDataset('${d.id}','${d.name}','${d.samples} · ${d.classes} classes', this)">
            <div class="ds-type">${d.type} ${d.custom ? '⭐ Custom' : ''}</div>
            <div class="ds-name">${d.name}</div>
            <div class="ds-desc">${d.desc}</div>
            <div class="ds-stats">
                <span class="ds-stat">${d.samples}</span>
                <span class="ds-stat">${d.classes} classes</span>
                <span class="ds-stat">${d.size}</span>
            </div>
        </div>
    `).join('');
}

function selectDataset(id, name, sub, el) {
    selectedDataset = id;
    document.querySelectorAll('.ds-card').forEach(c => c.classList.remove('selected'));
    if (el) {
        el.classList.add('selected');
    } else {
        const found = document.querySelector(`.ds-card[data-id="${id}"]`);
        if (found) found.classList.add('selected');
    }
    document.getElementById('mc-name').textContent = name;
    document.getElementById('mc-sub').textContent  = sub;
}

// ── CHIP TOGGLES ──────────────────────────────────────────────────────────────
function setupChips() {
    document.querySelectorAll('.chip').forEach(chip => {
        chip.addEventListener('click', () => chip.classList.toggle('on'));
    });
}

// ── RUN PIPELINE ──────────────────────────────────────────────────────────────
async function runPipeline() {
    const pruning  = [...document.querySelectorAll('.chip input[name=prune]')]
        .filter(cb => cb.closest('.chip').classList.contains('on'))
        .map(cb => parseFloat(cb.value));
    const device   = document.getElementById('device-select').value;
    const skipTrain= document.getElementById('skip-train').checked;

    if (!pruning.length) { alert('Select at least one pruning level'); return; }

    document.getElementById('run-btn').disabled = true;
    document.getElementById('card-progress').style.display = 'block';
    document.getElementById('card-results').style.display  = 'none';
    document.getElementById('log-box').innerHTML = '';
    setStep(3);

    await fetch('/api/run', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ dataset: selectedDataset, skip_train: skipTrain,
                               pruning, device })
    });

    pollTimer = setInterval(pollStatus, 2000);
}

// ── POLL STATUS ───────────────────────────────────────────────────────────────
async function checkStatus() {
    const res  = await fetch('/api/status');
    const data = await res.json();
    if (data.complete) renderResults(data.results);
}

async function pollStatus() {
    const res  = await fetch('/api/status');
    const data = await res.json();

    // Progress bars
    updateProgress(data);

    // Logs
    const box = document.getElementById('log-box');
    box.innerHTML = (data.logs||[]).map(l =>
        `<div class="log-${l.level}">[${l.time}] ${l.msg}</div>`
    ).join('');
    box.scrollTop = box.scrollHeight;

    // Live KPIs from latest result
    if (data.results && data.results.length > 0) {
        const last = data.results[data.results.length - 1];
        document.getElementById('lk-acc').textContent    = parseFloat(last['accuracy_%']).toFixed(2)+'%';
        document.getElementById('lk-size').textContent   = parseFloat(last['model_size_mb']).toFixed(2)+' MB';
        document.getElementById('lk-speed').textContent  = parseFloat(last['inference_time_ms']).toFixed(0)+' ms';
        document.getElementById('lk-energy').textContent = parseFloat(last['energy_kwh']).toExponential(1)+' kWh';
        updateLiveChart(data.results);
    }

    // Done
    if (data.complete || data.error) {
        clearInterval(pollTimer);
        document.getElementById('run-btn').disabled = false;
        document.getElementById('pipeline-tag').textContent  = data.error ? '✗ Error' : '✓ Complete';
        document.getElementById('pipeline-tag').className    = 'tag '+(data.error?'tag-red':'tag-green');
        if (data.error) { alert('Error: ' + data.error); return; }
        if (data.results) renderResults(data.results);
        setStep(4);
    }
}

// ── PROGRESS BAR ──────────────────────────────────────────────────────────────
const STEPS = [
    {label:'Loading dataset',      pct:[0,10]},
    {label:'Training model',       pct:[10,42]},
    {label:'Evaluating baseline',  pct:[42,55]},
    {label:'Pruning models',       pct:[55,78]},
    {label:'Quantizing INT8',      pct:[78,92]},
    {label:'Saving results',       pct:[92,100]},
];

function updateProgress(data) {
    const pct  = data.percent || 0;
    const wrap = document.getElementById('prog-steps');
    wrap.innerHTML = STEPS.map(s => {
        const done    = pct >= s.pct[1];
        const running = pct >= s.pct[0] && pct < s.pct[1];
        const fill    = done ? 100 : running ? Math.round((pct-s.pct[0])/(s.pct[1]-s.pct[0])*100) : 0;
        const color   = done ? 'linear-gradient(90deg,#059669,#34d399)' : 'linear-gradient(90deg,#4f46e5,#7c3aed)';
        const status  = done ? '<span style="color:#059669">✓ Done</span>'
                       : running ? `<span style="color:#4f46e5">${fill}%</span>`
                       : '<span style="color:#94a3b8">Waiting...</span>';
        return `<div class="prog-wrap">
            <div class="prog-top"><span>${s.label}</span>${status}</div>
            <div class="prog-bg"><div class="prog-fill" style="width:${fill}%;background:${color}"></div></div>
        </div>`;
    }).join('');
}

// ── LIVE CHART ────────────────────────────────────────────────────────────────
function updateLiveChart(results) {
    const ctx = document.getElementById('live-chart');
    const labels = results.map(r => r.label);
    const acc    = results.map(r => parseFloat(r['accuracy_%']));
    const sizes  = results.map(r => parseFloat(r['model_size_mb']));
    const colors = ['#4f46e5','#059669','#d97706','#7c3aed','#dc2626'];

    if (charts.live) charts.live.destroy();
    charts.live = new Chart(ctx, {
        type: 'bar',
        data: { labels, datasets: [
            {label:'Accuracy %', data:acc,   backgroundColor:colors.map(c=>c+'aa'), yAxisID:'y'},
            {label:'Size MB',    data:sizes,  backgroundColor:colors, borderRadius:4, yAxisID:'y1'},
        ]},
        options: {
            responsive:true, maintainAspectRatio:false,
            plugins:{legend:{labels:{color:'#4a5578',font:{size:10},boxWidth:10,padding:10}}},
            scales:{
                x:{grid:{display:false},ticks:{color:'#8892a8',font:{size:9}}},
                y:{position:'left',ticks:{color:'#8892a8',font:{size:9},callback:v=>v+'%'}},
                y1:{position:'right',grid:{display:false},ticks:{color:'#8892a8',font:{size:9},callback:v=>v+' MB'}},
            }
        }
    });
}

// ── RENDER FINAL RESULTS ──────────────────────────────────────────────────────
function renderResults(results) {
    document.getElementById('card-results').style.display = 'block';
    window.scrollTo({top: document.getElementById('card-results').offsetTop - 80, behavior:'smooth'});

    const base  = results[0];
    const best  = results.reduce((a,b) => parseFloat(a['model_size_mb']) < parseFloat(b['model_size_mb']) ? a : b);
    const fastst= results.reduce((a,b) => parseFloat(a['inference_time_ms']) < parseFloat(b['inference_time_ms']) ? a : b);
    const baseS = parseFloat(base['model_size_mb']);
    const bestS = parseFloat(best['model_size_mb']);
    const compression = baseS > 0 ? (baseS/bestS).toFixed(2) : '—';
    const accDrop = Math.abs(parseFloat(base['accuracy_%']) - Math.min(...results.map(r=>parseFloat(r['accuracy_%'])))).toFixed(2);

    // Summary pills
    document.getElementById('summary-pills').innerHTML = `
        <div class="sp gr"><div class="sp-lbl">Size reduction</div><div class="sp-val">${compression}×</div><div class="sp-sub">${baseS.toFixed(2)} MB → ${bestS.toFixed(2)} MB</div></div>
        <div class="sp re"><div class="sp-lbl">Max acc drop</div><div class="sp-val">${accDrop}%</div><div class="sp-sub">Across all models</div></div>
        <div class="sp am"><div class="sp-lbl">Fastest model</div><div class="sp-val">${parseFloat(fastst['inference_time_ms']).toFixed(0)} ms</div><div class="sp-sub">${fastst.label}</div></div>
        <div class="sp pu"><div class="sp-lbl">Models tested</div><div class="sp-val">${results.length}</div><div class="sp-sub">Including baseline</div></div>
    `;

    // Charts
    const labels = results.map(r => r.label);
    const colors = ['#4f46e5','#059669','#d97706','#7c3aed','#dc2626'];
    const TICK = {color:'#8892a8', font:{size:9}};
    const GRID = 'rgba(0,0,0,0.05)';
    const tipCfg = {backgroundColor:'#0f1729',titleColor:'#eef0f5',bodyColor:'#8892a8',
                    titleFont:{size:12},bodyFont:{size:10},padding:10,cornerRadius:8};

    const makeBar = (id, data, minY) => {
        if(charts[id]) charts[id].destroy();
        charts[id] = new Chart(document.getElementById(id),{
            type:'bar',
            data:{labels,datasets:[{data,backgroundColor:colors,borderRadius:5,borderSkipped:false}]},
            options:{responsive:true,maintainAspectRatio:false,
                plugins:{legend:{display:false},tooltip:tipCfg},
                scales:{x:{grid:{display:false},ticks:{...TICK,maxRotation:12}},
                        y:{min:minY,grid:{color:GRID},ticks:TICK}}}
        });
    };

    const accMin = Math.floor(Math.min(...results.map(r=>parseFloat(r['accuracy_%'])))*2)/2 - 1;
    makeBar('chart-size',   results.map(r=>parseFloat(r['model_size_mb'])),     0);
    makeBar('chart-acc',    results.map(r=>parseFloat(r['accuracy_%'])),         accMin);
    makeBar('chart-speed',  results.map(r=>parseFloat(r['inference_time_ms'])),  0);
    makeBar('chart-energy', results.map(r=>parseFloat(r['energy_kwh'])),         0);

    // Find best per column
    const bestAcc   = Math.max(...results.map(r=>parseFloat(r['accuracy_%'])));
    const bestSize  = Math.min(...results.map(r=>parseFloat(r['model_size_mb'])));
    const bestSpeed = Math.min(...results.map(r=>parseFloat(r['inference_time_ms'])));
    const bestEnergy= Math.min(...results.map(r=>parseFloat(r['energy_kwh'])));

    const chips = ['#4f46e5bb','#059669bb','#d97706bb','#7c3aedbb','#dc2626bb'];
    document.getElementById('result-tbody').innerHTML = results.map((r,i) => `
        <tr>
            <td><span style="display:inline-block;font-size:10px;padding:2px 8px;border-radius:6px;font-weight:600;margin-right:5px;background:${chips[i%chips.length]}20;color:${chips[i%chips.length]};border:1px solid ${chips[i%chips.length]}44">${r.label.split(' ')[0]}</span>${r.label}</td>
            <td class="${parseFloat(r['accuracy_%'])===bestAcc?'best':''}">${parseFloat(r['accuracy_%']).toFixed(2)}%</td>
            <td class="${parseFloat(r['model_size_mb'])===bestSize?'best':''}">${parseFloat(r['model_size_mb']).toFixed(4)}</td>
            <td class="${parseFloat(r['inference_time_ms'])===bestSpeed?'best':''}">${parseFloat(r['inference_time_ms']).toFixed(2)}</td>
            <td class="${parseFloat(r['energy_kwh'])===bestEnergy?'best':''}">${parseFloat(r['energy_kwh']).toExponential(2)}</td>
            <td>${parseFloat(r['co2_kg']).toExponential(2)}</td>
            <td><button class="dl-btn-sm" onclick="dlModel('${r.label}')">⬇ Download</button></td>
        </tr>
    `).join('');
}

// ── DOWNLOAD ──────────────────────────────────────────────────────────────────
function dlModel(label) {
    const name = label.toLowerCase().replace(/\s+/g,'_').replace('%','') + `_${selectedDataset}.pt`;
    window.location.href = `/api/download/${name}`;
}

// ── RESET ─────────────────────────────────────────────────────────────────────
async function resetPipeline() {
    if(pollTimer) clearInterval(pollTimer);
    await fetch('/api/reset', {method:'POST'});
    document.getElementById('card-progress').style.display = 'none';
    document.getElementById('card-results').style.display  = 'none';
    document.getElementById('run-btn').disabled = false;
    document.getElementById('log-box').innerHTML = '';
    document.getElementById('pipeline-tag').textContent = 'Running...';
    document.getElementById('pipeline-tag').className   = 'tag tag-blue';
    ['lk-acc','lk-size','lk-speed','lk-energy'].forEach(id => {
        document.getElementById(id).textContent = '—';
    });
    setStep(2);
}

// ── STEP INDICATOR ────────────────────────────────────────────────────────────
function setStep(n) {
    for(let i=1;i<=5;i++){
        const el = document.getElementById('step-'+i);
        const ln = document.getElementById('line-'+i);
        el.className = 'step-item '+(i<n?'done':i===n?'active':'wait');
        if(ln) ln.className = 'step-line '+(i<n?'done':'');
    }
}

// ── MODELS LIST ───────────────────────────────────────────────────────────────
async function loadModelsList() {
    const res   = await fetch('/api/models');
    const data  = await res.json();
    document.getElementById('models-list').innerHTML = data.length
        ? data.map(m => `
            <div class="model-item">
                <div class="mi-name">${m.name}</div>
                <div class="mi-size">${m.size_mb} MB · ${m.modified}</div>
                <button class="mi-dl" onclick="window.location.href='/api/download/${m.name}'">⬇ Download</button>
            </div>`).join('')
        : '<p style="color:var(--text3)">No saved models yet. Run the pipeline first.</p>';
}

// ── CUSTOM DATASET UPLOAD ─────────────────────────────────────────────────────
function onFileSelected(input) {
    if (input.files && input.files[0]) {
        const f = input.files[0];
        document.getElementById('file-drop-text').innerHTML = `
            <span style="font-size:24px; display:block; margin-bottom:6px;">✅</span>
            <strong style="color:var(--green); font-size:13px;">${escapeHtml(f.name)}</strong>
            <div style="font-size:11px; color:var(--text3); margin-top:4px;">Size: ${(f.size/1024).toFixed(1)} KB</div>
        `;
        const nameInput = document.getElementById('up-name');
        if (!nameInput.value.trim()) {
            nameInput.value = f.name.replace('.csv', '').replace(/[-_]/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        }
    }
}

async function handleDatasetUpload(e) {
    e.preventDefault();
    const fileInput = document.getElementById('up-file');
    if (!fileInput.files || !fileInput.files[0]) {
        alert('Please choose a .csv file to upload.');
        return;
    }
    const name = document.getElementById('up-name').value.trim();
    const desc = document.getElementById('up-desc').value.trim();
    const btn = document.getElementById('up-submit-btn');
    const statusDiv = document.getElementById('upload-status');

    btn.disabled = true;
    btn.textContent = '⏳ Uploading & Analyzing...';
    statusDiv.style.display = 'none';

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('name', name);
    formData.append('desc', desc);

    try {
        const res = await fetch('/api/datasets/upload', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        if (!res.ok) {
            throw new Error(data.error || 'Upload failed');
        }

        statusDiv.style.display = 'block';
        statusDiv.style.background = 'var(--green-l)';
        statusDiv.style.color = 'var(--green)';
        statusDiv.style.border = '1px solid var(--green-b)';
        statusDiv.innerHTML = `✅ Successfully registered <strong>${escapeHtml(data.dataset.name)}</strong> (${data.dataset.samples} samples, ${data.dataset.classes} classes)!`;

        document.getElementById('dataset-upload-form').reset();
        document.getElementById('file-drop-text').innerHTML = `
            <span style="font-size:24px; display:block; margin-bottom:6px;">📄</span>
            <strong style="color:var(--accent); font-size:13px;">Click to select CSV file</strong>
            <div style="font-size:11px; color:var(--text3); margin-top:4px;">Comma-separated values with target/label in the final column</div>
        `;

        await loadDatasets();
        await loadCustomDatasetsList();

        setTimeout(() => {
            selectDataset(data.dataset.id, data.dataset.name, `${data.dataset.samples} · ${data.dataset.classes} classes`, null);
            showPage('dashboard');
        }, 1500);

    } catch (err) {
        statusDiv.style.display = 'block';
        statusDiv.style.background = 'var(--red-l)';
        statusDiv.style.color = 'var(--red)';
        statusDiv.style.border = '1px solid #fecaca';
        statusDiv.textContent = 'Error: ' + err.message;
    } finally {
        btn.disabled = false;
        btn.textContent = '➕ Upload & Register Dataset';
    }
}

async function loadCustomDatasetsList() {
    const container = document.getElementById('custom-datasets-container');
    if (!container) return;
    try {
        const res = await fetch('/api/datasets');
        const list = await res.json();
        const custom = list.filter(d => d.custom);
        const countTag = document.getElementById('custom-count-tag');
        if (countTag) countTag.textContent = `${custom.length} Custom`;

        if (custom.length === 0) {
            container.innerHTML = `<div style="text-align:center; padding:20px; color:var(--text3); font-size:12px;">No custom datasets uploaded yet. Upload your first CSV above!</div>`;
            return;
        }

        container.innerHTML = `
            <table class="rtbl">
                <thead>
                    <tr><th>Name</th><th>Type</th><th>Samples</th><th>Classes</th><th>File Size</th><th>Action</th></tr>
                </thead>
                <tbody>
                    ${custom.map(d => `
                        <tr>
                            <td><strong>${escapeHtml(d.name)}</strong><br><small style="color:var(--text3)">${escapeHtml(d.desc)}</small></td>
                            <td><span class="tag tag-blue">${d.type}</span></td>
                            <td>${d.samples}</td>
                            <td>${d.classes}</td>
                            <td>${d.size}</td>
                            <td>
                                <button class="dl-btn-sm" onclick="selectDataset('${d.id}','${escapeHtml(d.name)}','${d.samples} · ${d.classes} classes', null); showPage('dashboard');">
                                    ▶ Select for Compression
                                </button>
                            </td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
    } catch (e) {
        container.innerHTML = `<div style="color:var(--red); font-size:12px;">Failed to load custom datasets.</div>`;
    }
}

function escapeHtml(text) {
    return (text || '').replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
