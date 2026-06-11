async function apiGet(path) {
  const res = await fetch(path);
  const data = await res.json();
  if (!res.ok) throw new Error(data?.error || "Request failed");
  return data;
}

async function apiPost(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data?.error || "Request failed");
  return data;
}

function tag(ok, text) {
  const cls = ok === true ? "good" : ok === false ? "bad" : "warn";
  return `<span class="tag ${cls}">${text}</span>`;
}

function flightsTable(flights) {
  const rows = flights
    .slice()
    .sort((a, b) => a.fid.localeCompare(b.fid))
    .map((f) => {
      const status = f.delayed ? tag(false, "DELAYED") : f.on_time ? tag(true, "ON TIME") : tag(null, "UNKNOWN");
      return `<tr>
        <td>${f.fid}</td>
        <td>${f.origin}</td>
        <td>${f.destination}</td>
        <td>${f.depart_hour}</td>
        <td>${f.capacity_kg}</td>
        <td>${f.remaining_capacity_kg}</td>
        <td>${status}</td>
      </tr>`;
    })
    .join("");

  return `<table>
    <thead>
      <tr>
        <th>Flight</th><th>From</th><th>To</th><th>Dep(h)</th><th>Cap(kg)</th><th>Rem(kg)</th><th>Status</th>
      </tr>
    </thead>
    <tbody>${rows}</tbody>
  </table>`;
}

function cargoTable(cargo) {
  const rows = cargo
    .slice()
    .sort((a, b) => a.cid.localeCompare(b.cid))
    .map((c) => {
      const pr = c.priority === "urgent" ? tag(null, "urgent") : c.priority === "high" ? tag(null, "high") : tag(null, "normal");
      return `<tr>
        <td>${c.cid}</td>
        <td>${c.destination}</td>
        <td>${c.weight_kg}</td>
        <td>${pr}</td>
        <td>${c.deadline_hour}</td>
      </tr>`;
    })
    .join("");

  return `<table>
    <thead>
      <tr>
        <th>Cargo</th><th>Dest</th><th>Weight(kg)</th><th>Priority</th><th>Deadline(h)</th>
      </tr>
    </thead>
    <tbody>${rows}</tbody>
  </table>`;
}

function renderState(state) {
  document.querySelector("#flightsTable").innerHTML = flightsTable(state.flights || []);
  document.querySelector("#cargoTable").innerHTML = cargoTable(state.cargo || []);
}

function setBox(id, text) {
  document.querySelector(id).textContent = text || "";
}

async function refresh() {
  const state = await apiGet("/api/state");
  renderState(state);
}

function fmtTrace(trace) {
  if (!trace || trace.length === 0) return "(no trace)";
  const tag = (k) => {
    if (k === "default") return "[Default]";
    if (k === "rule") return "[Rule]";
    if (k === "conflict") return "[Conflict]";
    if (k === "query") return "[Query]";
    return "[Info]";
  };
  return trace.map((t) => `${tag(t.kind)} ${t.message}`).join("\n");
}

function fmtAssignments(asg) {
  if (!asg || asg.length === 0) return "No assignments could be made.";
  return asg.map((a) => `${a.cargo_id} -> ${a.flight_id}`).join("\n");
}

function wire() {
  document.querySelector("#btnRefresh").addEventListener("click", async () => {
    try {
      await refresh();
    } catch (e) {
      alert(e.message);
    }
  });

  document.querySelector("#btnAddCargo").addEventListener("click", async () => {
    try {
      const cid = document.querySelector("#cargoId").value.trim();
      const destination = document.querySelector("#cargoDest").value.trim().toUpperCase();
      const weight_kg = Number(document.querySelector("#cargoWeight").value);
      const priority = document.querySelector("#cargoPriority").value;
      const deadline_hour = Number(document.querySelector("#cargoDeadline").value);

      const r = await apiPost("/api/add_cargo", { cid, destination, weight_kg, priority, deadline_hour });
      renderState(r.state);
      setBox("#scheduleResult", `Added cargo ${cid}.`);
    } catch (e) {
      alert(e.message);
    }
  });

  document.querySelector("#btnSetWeather").addEventListener("click", async () => {
    try {
      const airport = document.querySelector("#weatherAirport").value.trim().toUpperCase();
      const condition = document.querySelector("#weatherCondition").value;
      const r = await apiPost("/api/weather", { airport, condition });
      renderState(r.state);
      setBox("#scheduleResult", `Weather updated: weather(${airport}, ${condition}).`);
    } catch (e) {
      alert(e.message);
    }
  });

  document.querySelector("#btnDelay").addEventListener("click", async () => {
    try {
      const flight_id = document.querySelector("#delayFlight").value.trim().toUpperCase();
      const reason = document.querySelector("#delayReason").value.trim() || "unknown";
      const r = await apiPost("/api/delay", { flight_id, reason });
      renderState(r.state);
      setBox("#scheduleResult", `Event recorded: event_delay(${flight_id}, ${reason}).`);
    } catch (e) {
      alert(e.message);
    }
  });

  async function runSchedule() {
    const explanation = document.querySelector("#explainToggle").checked;
    const r = await apiPost("/api/schedule", { explanation });
    renderState(r.state);
    const out = `Assignments:\n${fmtAssignments(r.assignments)}\n\nTrace:\n${fmtTrace(r.trace)}`;
    setBox("#scheduleResult", out);
  }

  document.querySelector("#btnSchedule").addEventListener("click", async () => {
    try {
      await runSchedule();
    } catch (e) {
      alert(e.message);
    }
  });

  document.querySelector("#btnQuery").addEventListener("click", async () => {
    try {
      const query = document.querySelector("#queryText").value.trim();
      const r = await apiPost("/api/query", { query });
      const out = `Answers:\n${(r.answers || []).join("\n") || "(none)"}\n\nTrace:\n${fmtTrace(r.trace)}`;
      setBox("#queryResult", out);
    } catch (e) {
      alert(e.message);
    }
  });

  document.querySelector("#btnUnify").addEventListener("click", async () => {
    try {
      const a = document.querySelector("#unifyA").value.trim();
      const b = document.querySelector("#unifyB").value.trim();
      const r = await apiPost("/api/unify", { a, b });
      const out = r.ok
        ? `Unification succeeded.\nSubstitution: ${r.substitution}\nReason: ${r.reason}`
        : `Unification failed.\nReason: ${r.reason}`;
      setBox("#unifyResult", out);
    } catch (e) {
      alert(e.message);
    }
  });
}

async function main() {
  wire();
  try {
    await refresh();
    setBox("#scheduleResult", "Ready. Click “Run Scheduling” to derive assignments + trace.");
  } catch (e) {
    setBox("#scheduleResult", "Failed to load initial state: " + e.message);
  }
}

main();

