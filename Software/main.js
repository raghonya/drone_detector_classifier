const WS_URL = 'ws://192.168.16.151:7000/receive';

const els = {
  statusBox   : document.getElementById('statusBox'),
  signalInfo  : document.getElementById('signalInfo'),
  intervalSel : document.getElementById('intervalSel'),
  reconnect   : document.getElementById('reconnectBtn'),
  exportBtn   : document.getElementById('exportBtn'),
};

let socket = null;
let reconnectAttempts = 0;
let requestDelay = +els.intervalSel.value;
let waitingForResponse = false;

let circleMarker = null;

function highlightArea(location, type = '') {
  if (!location || typeof location.lat !== 'number' || typeof location.lng !== 'number') {
    console.warn('⚠️ Invalid location object — skipping highlight.');
    return;
  }

  const radius = typeof location.radius === 'number' ? location.radius : 200;

  // Remove previous marker if needed
  if (circleMarker) {
    circleMarker.remove();
    circleMarker = null;
  }

  console.log(`✅ Drawing red circle at (${location.lat}, ${location.lng}) with radius ${radius}`);

  circleMarker = L.circle([location.lat, location.lng], {
    color: '#ff0033',
    fillColor: '#ff0033',
    fillOpacity: 0.25,
    radius: radius
  }).addTo(map);

  const label = type ? `🚁 Drone Detected: <b>${type}</b>` : `🚁 Drone Detected`;
  circleMarker.bindPopup(label).openPopup();

  map.setView([location.lat, location.lng], 15, { animate: false });
}

function removeHighlightArea() {
  if (circleMarker) {
    console.log('🧽 Drone no longer detected — removing circle.');
    circleMarker.remove();
    circleMarker = null;
  }
}

function setStatus(text, cls) {
  els.statusBox.textContent = text;
  document.body.classList.remove('connected', 'error');
  document.body.classList.add(cls);
}

function updateSpectrum(freq, amp, type = '', detected = false) {
  chart.data.labels = freq;
  chart.data.datasets[0].data = amp;
  chart.update('none');

  const maxI = amp.reduce((iMax, x, i) => x > amp[iMax] ? i : iMax, 0);

  const detectionText = detected ? 'Yes' : 'No';
  const typeText = detected && type?.trim() ? type : 'N/A';

  els.signalInfo.innerHTML =
    `Peak&nbsp;Freq: <b>${freq[maxI].toFixed(1)} MHz</b><br>` +
    `Peak&nbsp;Amp : <b>${amp[maxI].toFixed(1)} dB</b><br>` +
    `Detected&nbsp;&nbsp;&nbsp;: <b>${detectionText}</b><br>` +
    `Drone&nbsp;Type : <b>${typeText}</b>`;
}

function handleIncomingData(jsonString) {
  try {
    console.log('📩 Raw WebSocket message:', jsonString);

    const wrapper = JSON.parse(jsonString);
    const data = JSON.parse(wrapper.Payload);

    if (typeof data.Location === 'string') {
      data.Location = JSON.parse(data.Location);
    }

    console.log('📦 Parsed data:', data);

    const freq = data.Frequency;
    const amp = data.Amplitude;
    const detected = data.Detected === true || data.Detected === 'true';
    const droneType = data.Type || data.type || '';

    if (!Array.isArray(freq) || !Array.isArray(amp) || freq.length !== amp.length || freq.length === 0) {
      console.warn("⚠️ Invalid or mismatched frequency/amplitude data.");
      return;
    }

    console.log(`📊 Frequency count: ${freq.length}, Type: "${data.Type || 'N/A'}"`);
    console.log("🧭 Location received:", data.Location);

    //const detected = data.Detected === true || data.Detected === 'true';
    //const droneType = data.Type || data.type || '';
    
    updateSpectrum(freq, amp, droneType, detected);
    
    // Show or hide the circle based on detection state
    if (detected && data.Location?.lat && data.Location?.lng) {
      data.Location.radius = +data.Location.radius || 200;
      highlightArea(data.Location, droneType);
    } else {
      removeHighlightArea();
    }
    

  } catch (err) {
    console.error("❌ Error parsing incoming data:", err);
  }
}

function connect() {
  console.log("🔌 Connecting to WebSocket:", WS_URL);
  socket = new WebSocket(WS_URL);

  socket.addEventListener('open', () => {
    console.log("✅ WebSocket connected.");
    setStatus('Connected', 'connected');
    reconnectAttempts = 0;
    waitingForResponse = false;
    requestNext();
  });

  socket.addEventListener('message', e => {
    console.log("📨 WebSocket message received.");
    handleIncomingData(e.data);
    setTimeout(() => {
      waitingForResponse = false;
      requestNext();
    }, requestDelay);
  });

  socket.addEventListener('error', () => {
    console.warn("⚠️ WebSocket error");
    setStatus('Error', 'error');
  });

  socket.addEventListener('close', () => {
    console.warn("❌ WebSocket closed");
    setStatus('Disconnected', 'error');
    if (reconnectAttempts < 5) {
      const delay = 2000;
      console.log(`🔄 Reconnecting in ${delay}ms...`);
      //reconnectAttempts++;
      setTimeout(connect, delay);
    }
  });
}

function requestNext() {
  if (socket && socket.readyState === WebSocket.OPEN && !waitingForResponse) {
    console.log("📡 Requesting next data packet...");
    waitingForResponse = true;
    socket.send(JSON.stringify({ CMD: "Get", Payload: {} }));
  }
}

const map = L.map('map', { attributionControl: false }).setView([40.1792, 44.4991], 13);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19 }).addTo(map);

const chart = new Chart(document.getElementById('fftChart').getContext('2d'), {
  type: 'line',
  data: {
    labels: [],
    datasets: [{
      data: [],
      borderColor: '#00ffcc',
      backgroundColor: 'rgba(0,255,204,0.08)',
      borderWidth: 1,
      pointRadius: 0,
      tension: 0.3
    }]
  },
  options: {
    responsive: true,
    animation: false,
    plugins: {
      legend: { display: false },
      tooltip: { enabled: false, mode: 'index', intersect: false }
    },
    hover: { mode: 'index', intersect: false },
    scales: {
      x: {
        title: { display: true, text: 'Frequency (MHz)', color: '#66ffc2' },
        ticks: { color: '#66ffc2' },
        grid: { color: '#003a32' }
      },
      y: {
        title: { display: true, text: 'Amplitude (dB)', color: '#66ffc2' },
        suggestedMin: -120,
        suggestedMax: 0,
        ticks: { color: '#66ffc2' },
        grid: { color: '#003a32' }
      }
    }
  }
});

els.intervalSel.addEventListener('change', e => {
  requestDelay = +e.target.value;
});

els.reconnect.addEventListener('click', () => {
  if (socket) socket.close(1000, 'manual reconnect');
  connect();
});

els.exportBtn.addEventListener('click', () => {
  const freqs = chart.data.labels;
  const amps = chart.data.datasets[0].data;
  if (!freqs?.length || !amps?.length) {
    alert('No data to export');
    return;
  }
  const rows = ['Frequency (Hz),Amplitude (dB)'];
  for (let i = 0; i < freqs.length; i++) {
    rows.push(`${freqs[i]},${amps[i]}`);
  }
  const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `spectrum_${Date.now()}.csv`;
  link.click();
  URL.revokeObjectURL(url);
});

connect();
