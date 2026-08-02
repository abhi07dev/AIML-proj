
const API_BASE = '';
let mode = 'image';
let selectedFile = null;

const dropzone   = document.getElementById('dropzone');
const fileInput  = document.getElementById('fileInput');
const fileChip   = document.getElementById('fileChip');
const fileName   = document.getElementById('fileName');
const clearBtn   = document.getElementById('clearFile');
const runBtn     = document.getElementById('runBtn');
const loading    = document.getElementById('loading');
const loadingText= document.getElementById('loadingText');
const results     = document.getElementById('results');
const errorBox    = document.getElementById('errorBox');
const dzPrimary   = document.getElementById('dzPrimary');
const dzSecondary = document.getElementById('dzSecondary');

document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    mode = tab.dataset.mode;
    clearSelection();
    if (mode === 'image') {
      fileInput.accept = 'image/*';
      dzPrimary.textContent = 'Drop an image, or click to browse';
      dzSecondary.textContent = 'JPG · PNG · WEBP — analyzed at 224×224';
    } else {
      fileInput.accept = 'video/*';
      dzPrimary.textContent = 'Drop a video, or click to browse';
      dzSecondary.textContent = 'MP4 · MOV · AVI — 24 frames sampled evenly';
    }
  });
});

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('dragover'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  if (e.dataTransfer.files.length) selectFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files.length) selectFile(fileInput.files[0]); });
clearBtn.addEventListener('click', clearSelection);

function selectFile(file){
  selectedFile = file;
  fileName.textContent = file.name;
  fileChip.classList.add('show');
  runBtn.disabled = false;
  results.classList.remove('show');
  errorBox.classList.remove('show');
}

function clearSelection(){
  selectedFile = null;
  fileInput.value = '';
  fileChip.classList.remove('show');
  runBtn.disabled = true;
  results.classList.remove('show');
  errorBox.classList.remove('show');
}

runBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  loading.classList.add('show');
  results.classList.remove('show');
  errorBox.classList.remove('show');
  runBtn.disabled = true;
  loadingText.textContent = mode === 'image'
    ? 'Running spatial + frequency streams…'
    : 'Sampling frames and scoring each one…';

  const formData = new FormData();
  formData.append('file', selectedFile);
  const endpoint = mode === 'image' ? '/api/predict-image' : '/api/predict-video';

  try {
    const res = await fetch(API_BASE + endpoint, { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Analysis failed');
    mode === 'image' ? renderImageResult(data) : renderVideoResult(data);
  } catch (err) {
    errorBox.textContent = '⚠ ' + err.message;
    errorBox.classList.add('show');
  } finally {
    loading.classList.remove('show');
    runBtn.disabled = false;
  }
});

function badgeMeta(label){
  return label === 'FAKE'
    ? { cls: 'fake', color: 'var(--fake)' }
    : { cls: 'real', color: 'var(--real)' };
}

function drawSpectrum(fakeProb){
  const bars = document.getElementById('spectrumBars');
  bars.innerHTML = '';
  const n = 40;
  for (let i = 0; i < n; i++) {
    const bar = document.createElement('div');
    bar.className = 'bar';
    const noise = Math.sin(i * 1.3) * 0.15 + Math.random() * 0.1;
    const h = Math.max(4, Math.min(100, (fakeProb * 100) * (0.55 + noise) + (i % 5 === 0 ? 6 : 0)));
    bar.style.height = '2px';
    bar.style.background = fakeProb >= 0.5 ? 'var(--fake)' : 'var(--real)';
    bars.appendChild(bar);
    setTimeout(() => { bar.style.height = h + '%'; }, 15 * i);
  }
  document.getElementById('spectrumPct').textContent = (fakeProb * 100).toFixed(1) + '% fake signal';
}

function renderImageResult(data){
  const { cls } = badgeMeta(data.label);
  const badge = document.getElementById('verdictBadge');
  badge.className = 'verdict-badge ' + cls;
  badge.textContent = data.verdict;

  document.getElementById('verdictMeta').innerHTML =
    `<b>${data.filename}</b><br>Confidence ${data.confidence}%`;

  document.getElementById('frameStrip').style.display = 'none';
  const img = document.getElementById('previewImg');
  img.src = URL.createObjectURL(selectedFile);
  img.style.display = 'block';

  drawSpectrum(data.fake_prob);

  document.getElementById('metricsGrid').innerHTML = `
    <div class="metric"><div class="k">Fake probability</div><div class="v">${(data.fake_prob*100).toFixed(1)}%</div></div>
    <div class="metric"><div class="k">Real probability</div><div class="v">${(data.real_prob*100).toFixed(1)}%</div></div>
    <div class="metric"><div class="k">Verdict</div><div class="v">${data.label}</div></div>
  `;
  results.classList.add('show');
}

function renderVideoResult(data){
  const { cls } = badgeMeta(data.label);
  const badge = document.getElementById('verdictBadge');
  badge.className = 'verdict-badge ' + cls;
  badge.textContent = data.label === 'FAKE' ? 'LIKELY FAKE' : 'LIKELY REAL';

  document.getElementById('verdictMeta').innerHTML =
    `<b>${data.filename}</b><br>${data.duration_sec}s · ${data.total_frames} frames · ${data.frames_analyzed} sampled`;

  document.getElementById('previewImg').style.display = 'none';

  const strip = document.getElementById('frameStrip');
  strip.style.display = 'flex';
  strip.innerHTML = '';
  // note: we don't have per-frame thumbnails from the server response (video bytes
  // aren't re-sent), so we render a compact per-frame probability timeline instead.
  data.frame_results.forEach(r => {
    const cell = document.createElement('div');
    cell.className = 'frame';
    cell.innerHTML = `
      <div style="width:88px;height:88px;border-radius:6px;border:2px solid var(--hairline);
        display:flex;align-items:center;justify-content:center;font-family:'IBM Plex Mono';
        font-size:11px;color:${r.fake_prob >= 0.5 ? 'var(--fake)' : 'var(--real)'};">
        ${(r.fake_prob*100).toFixed(0)}%
      </div>
      <div class="fp">${r.timestamp}s</div>`;
    strip.appendChild(cell);
  });

  drawSpectrum(data.mean_fake_prob);

  document.getElementById('metricsGrid').innerHTML = `
    <div class="metric"><div class="k">Mean fake prob</div><div class="v">${(data.mean_fake_prob*100).toFixed(1)}%</div></div>
    <div class="metric"><div class="k">Suspicious frames</div><div class="v">${data.suspicious_frame_count}/${data.frames_analyzed}</div></div>
    <div class="metric"><div class="k">FPS</div><div class="v">${data.fps}</div></div>
  `;
  results.classList.add('show');
}

// backend health check
(async () => {
  const dot = document.getElementById('statusDot');
  const text = document.getElementById('statusText');
  try {
    const res = await fetch(API_BASE + '/api/health');
    const data = await res.json();
    if (data.model_loaded) {
      dot.classList.add('on');
      text.textContent = 'model online · ' + data.device;
    } else {
      dot.classList.add('off');
      text.textContent = 'model not loaded';
    }
  } catch {
    dot.classList.add('off');
    text.textContent = 'backend unreachable';
  }
})();
