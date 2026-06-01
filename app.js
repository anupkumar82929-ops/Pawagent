const $ = (id) => document.getElementById(id);

const statusPill = $("statusPill");
const dropzone = $("dropzone");
const fileInput = $("fileInput");
const cameraInput = $("cameraInput");
const cameraBtn = $("cameraBtn");
const previewWrap = $("previewWrap");
const preview = $("preview");
const clearBtn = $("clearBtn");
const predictBtn = $("predictBtn");
const predictError = $("predictError");
const results = $("results");
const topLabel = $("topLabel");
const topConf = $("topConf");
const predList = $("predList");
const breedHint = $("breedHint");
const chatLog = $("chatLog");
const chatForm = $("chatForm");
const chatInput = $("chatInput");
const sendBtn = $("sendBtn");
const stepAssistant = $("stepAssistant");
const stepTracker = $("stepTracker");
const stepTeaser = $("stepTeaser");
const newDogBtn = $("newDogBtn");

let selectedFile = null;
let lastTopBreed = null;
let assistantUnlocked = false;
/** Last 10 turns (20 messages) for advanced chat context */
const chatHistory = [];

async function refreshStatus() {
  try {
    const r = await fetch("/api/status");
    const j = await r.json();
    if (j.model_ready) {
      statusPill.textContent = "Model ready";
      statusPill.className = "pill pill-ok";
    } else {
      statusPill.textContent = "Model missing — train first";
      statusPill.className = "pill pill-bad";
    }
  } catch {
    statusPill.textContent = "Server offline";
    statusPill.className = "pill pill-bad";
  }
}

function setFile(file, source = "file") {
  selectedFile = file;
  predictError.classList.add("hidden");
  if (file) {
    if (source === "camera") fileInput.value = "";
    else cameraInput.value = "";
  } else {
    fileInput.value = "";
    cameraInput.value = "";
  }
  if (!file) {
    previewWrap.classList.add("hidden");
    predictBtn.disabled = true;
    if (preview.src) URL.revokeObjectURL(preview.src);
    preview.removeAttribute("src");
    return;
  }
  const url = URL.createObjectURL(file);
  preview.src = url;
  previewWrap.classList.remove("hidden");
  predictBtn.disabled = false;
}

dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  const f = fileInput.files?.[0];
  if (f) setFile(f, "file");
});

cameraBtn.addEventListener("click", () => cameraInput.click());
cameraInput.addEventListener("change", () => {
  const f = cameraInput.files?.[0];
  if (f) setFile(f, "camera");
});

["dragenter", "dragover"].forEach((ev) => {
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag");
  });
});
["dragleave", "drop"].forEach((ev) => {
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag");
  });
});
dropzone.addEventListener("drop", (e) => {
  const f = e.dataTransfer?.files?.[0];
  if (f && f.type.startsWith("image/")) setFile(f, "file");
});

clearBtn.addEventListener("click", () => {
  setFile(null);
  results.classList.add("hidden");
  newDogBtn.hidden = true;
});

function humanizeBreed(s) {
  return String(s || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function unlockAssistant() {
  assistantUnlocked = true;
  breedHint.value = humanizeBreed(lastTopBreed);
  chatLog.innerHTML = "";
  appendBubble(
    "bot",
    `Breed context is set to ${humanizeBreed(lastTopBreed)} (from your photo). Ask me anything about care, training, exercise, or health basics — I will keep this breed in mind.`
  );
  stepAssistant.classList.remove("hidden");
  stepAssistant.setAttribute("aria-hidden", "false");
  stepAssistant.classList.remove("card-chat--enter");
  void stepAssistant.offsetWidth;
  stepAssistant.classList.add("card-chat--enter");
  stepTracker.textContent = "Step 2 of 2 — Care assistant";
  stepTeaser.classList.add("completed");
  stepTeaser.querySelector(".step-teaser-inner").textContent =
    "Step 2 unlocked — you can chat with the assistant below.";
  newDogBtn.hidden = false;
  requestAnimationFrame(() => {
    stepAssistant.scrollIntoView({ behavior: "smooth", block: "start" });
  });
}

function lockAssistantFlow() {
  assistantUnlocked = false;
  lastTopBreed = null;
  breedHint.value = "";
  chatLog.innerHTML = "";
  chatHistory.length = 0;
  stepAssistant.classList.add("hidden");
  stepAssistant.setAttribute("aria-hidden", "true");
  stepAssistant.classList.remove("card-chat--enter");
  stepTracker.textContent = "Step 1 of 2 — Identify the breed";
  stepTeaser.classList.remove("completed");
  const inner = stepTeaser.querySelector(".step-teaser-inner");
  if (inner) {
    inner.innerHTML =
      '<span class="step-dot" aria-hidden="true"></span> Step 2 unlocks after a successful breed match — your care assistant will appear below.';
  }
  newDogBtn.hidden = true;
  results.classList.add("hidden");
  setFile(null);
}

newDogBtn.addEventListener("click", () => {
  lockAssistantFlow();
});

predictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  predictError.classList.add("hidden");
  results.classList.add("hidden");
  predictBtn.disabled = true;
  predictBtn.textContent = "Analyzing…";
  try {
    const fd = new FormData();
    fd.append("file", selectedFile);
    const r = await fetch("/api/predict", { method: "POST", body: fd });
    const j = await r.json();
    if (!r.ok) {
      const err =
        typeof j.detail === "string" ? j.detail : j.detail != null ? JSON.stringify(j.detail) : r.statusText;
      throw new Error(err);
    }
    const preds = j.predictions || [];
    if (!preds.length) throw new Error("No predictions returned.");
    lastTopBreed = preds[0].label;
    topLabel.textContent = humanizeBreed(preds[0].label);
    topConf.textContent = `Confidence about ${Math.round(preds[0].confidence * 1000) / 10}% — ranked alternatives below.`;
    predList.innerHTML = "";
    preds.forEach((p) => {
      const li = document.createElement("li");
      li.innerHTML = `<span>${humanizeBreed(p.label)}</span><span class="conf">${(p.confidence * 100).toFixed(1)}%</span>`;
      predList.appendChild(li);
    });
    results.classList.remove("hidden");
    unlockAssistant();
  } catch (e) {
    predictError.textContent = e.message || String(e);
    predictError.classList.remove("hidden");
  } finally {
    predictBtn.disabled = !selectedFile;
    predictBtn.textContent = "Analyze breed";
  }
});

function appendBubble(role, text) {
  const div = document.createElement("div");
  div.className = `bubble ${role}`;
  div.textContent = text;
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!assistantUnlocked || !lastTopBreed) return;
  const msg = chatInput.value.trim();
  if (!msg) return;
  const hint = lastTopBreed;
  appendBubble("user", msg);
  chatInput.value = "";
  sendBtn.disabled = true;
  appendBubble("bot", "…");
  const placeholder = chatLog.lastChild;
  try {
    const r = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({
        message: msg,
        breed_hint: hint,
        history: chatHistory.slice(-20),
      }),
    });
    if (!r.ok || !r.body) {
      const t = await r.text();
      throw new Error(t || r.statusText);
    }
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buffer = "";
    let full = "";
    placeholder.textContent = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += dec.decode(value, { stream: true });
      let sep;
      while ((sep = buffer.indexOf("\n\n")) >= 0) {
        const block = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        for (const line of block.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          const raw = line.slice(6).trim();
          if (!raw || raw === "[DONE]") continue;
          try {
            const j = JSON.parse(raw);
            if (j.error) throw new Error(j.error);
            if (j.done) continue;
            if (j.c) {
              full += j.c;
              placeholder.textContent = full;
              chatLog.scrollTop = chatLog.scrollHeight;
            }
          } catch (parseErr) {
            if (parseErr instanceof SyntaxError) continue;
            throw parseErr;
          }
        }
      }
    }
    if (!full.trim()) full = "(No text returned.)";
    placeholder.textContent = full;
    chatHistory.push({ role: "user", content: msg });
    chatHistory.push({ role: "assistant", content: full });
    while (chatHistory.length > 20) chatHistory.shift();
  } catch (err) {
    placeholder.textContent = `Error: ${err.message || err}`;
  } finally {
    sendBtn.disabled = false;
  }
});

refreshStatus();

/* Mini yard: tap / touch moves ball; puppy chases */
function initPuppyPlayground() {
  const yard = document.getElementById("playYard");
  const ball = document.getElementById("playBall");
  const pup = document.getElementById("playPuppy");
  const flip = document.getElementById("playPuppyFlip");
  if (!yard || !ball || !pup || !flip) return;

  const BALL_R = 18;
  const PW = 76;
  const PH = 76;
  let puppyX = 8;
  let puppyY = 40;
  let ballX = 140;
  let ballY = 72;
  let resizeTimer;

  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  function applyInitial() {
    const r = yard.getBoundingClientRect();
    if (r.width < 48) return;
    pup.style.transition = "none";
    ball.style.transition = "none";
    puppyX = r.width * 0.06;
    puppyY = r.height * 0.48;
    ballX = r.width * 0.76;
    ballY = r.height * 0.56;
    pup.style.left = `${puppyX}px`;
    pup.style.top = `${puppyY}px`;
    ball.style.left = `${ballX}px`;
    ball.style.top = `${ballY}px`;
    ball.style.transform = "rotate(0deg)";
    void yard.offsetWidth;
    requestAnimationFrame(() => {
      pup.style.transition = "";
      ball.style.transition = "";
    });
  }

  function toss(clientX, clientY) {
    const r = yard.getBoundingClientRect();
    const cx = clamp(clientX - r.left, BALL_R, r.width - BALL_R);
    const cy = clamp(clientY - r.top, BALL_R, r.height - BALL_R);
    ballX = cx;
    ballY = cy;

    const prevX = puppyX;
    const prevY = puppyY;
    const tx = clamp(cx - 40, 4, r.width - PW - 4);
    const ty = clamp(cy - 34, 4, r.height - PH - 4);
    const dist = Math.hypot(tx - prevX, ty - prevY);

    const prevRotMatch = /rotate\(([-\d.]+)deg\)/.exec(ball.style.transform || "");
    const prevRot = prevRotMatch ? parseFloat(prevRotMatch[1]) : 0;
    const roll = prevRot + (Math.random() * 70 - 35) + dist * 0.12;
    ball.style.transform = `rotate(${roll}deg)`;
    ball.style.left = `${cx}px`;
    ball.style.top = `${cy}px`;

    if (tx < prevX - 6) flip.classList.add("facing-left");
    else if (tx > prevX + 6) flip.classList.remove("facing-left");

    const dur = dist < 14 ? 0.2 : Math.min(1.35, 0.28 + dist / 280);
    pup.classList.remove("play-puppy--idle", "play-puppy--wiggle");
    pup.style.transitionDuration = `${dur}s`;
    pup.style.left = `${tx}px`;
    pup.style.top = `${ty}px`;
    puppyX = tx;
    puppyY = ty;

    if (dist < 4) {
      pup.classList.add("play-puppy--wiggle");
      window.setTimeout(() => {
        pup.classList.remove("play-puppy--wiggle");
        pup.classList.add("play-puppy--idle");
      }, 700);
      return;
    }

    let ended = false;
    const onEnd = (e) => {
      if (e.target !== pup || (e.propertyName !== "left" && e.propertyName !== "top")) return;
      if (ended) return;
      ended = true;
      pup.removeEventListener("transitionend", onEnd);
      pup.classList.add("play-puppy--wiggle");
      window.setTimeout(() => {
        pup.classList.remove("play-puppy--wiggle");
        pup.classList.add("play-puppy--idle");
      }, 700);
    };
    pup.addEventListener("transitionend", onEnd);
  }

  yard.addEventListener("pointerdown", (e) => {
    if (e.pointerType === "mouse" && e.button !== 0) return;
    e.preventDefault();
    toss(e.clientX, e.clientY);
  });

  window.addEventListener(
    "resize",
    () => {
      window.clearTimeout(resizeTimer);
      resizeTimer = window.setTimeout(applyInitial, 100);
    },
    { passive: true }
  );

  applyInitial();
  if (document.fonts?.ready) void document.fonts.ready.then(applyInitial);
}

initPuppyPlayground();
