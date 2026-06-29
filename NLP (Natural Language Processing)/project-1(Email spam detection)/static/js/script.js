/* ── Spam Detector — Frontend Logic ─────────────────────────────────────── */

const textarea     = document.getElementById("message-input");
const predictBtn   = document.getElementById("predict-btn");
const clearBtn     = document.getElementById("clear-btn");
const resultCard   = document.getElementById("result-card");
const resultVerdict= document.getElementById("result-verdict");
const resultConf   = document.getElementById("result-confidence");
const resultProb   = document.getElementById("result-prob");
const probBar      = document.getElementById("prob-bar");
const exampleBtns  = document.querySelectorAll(".example-btn");

// ── Predict ────────────────────────────────────────────────────────────────

async function runPrediction(message) {
  if (!message.trim()) {
    shake(textarea);
    return;
  }

  // Loading state
  predictBtn.disabled = true;
  predictBtn.innerHTML = `<span class="spinner"></span> Analysing…`;

  try {
    const res  = await fetch("/predict", {
      method : "POST",
      headers: { "Content-Type": "application/json" },
      body   : JSON.stringify({ message }),
    });

    if (!res.ok) {
      const err = await res.json();
      showError(err.error || "Server error.");
      return;
    }

    const data = await res.json();
    showResult(data);
  } catch (e) {
    showError("Could not reach the server. Is Flask running?");
  } finally {
    predictBtn.disabled = false;
    predictBtn.innerHTML = `<span class="btn-icon">⚡</span> Analyse`;
  }
}

// ── Result display ─────────────────────────────────────────────────────────

function showResult(data) {
  const isSpam = data.is_spam;
  const prob   = data.raw_prob;                  // 0–1 probability of spam
  const conf   = (data.confidence * 100).toFixed(1) + "%";

  resultVerdict.textContent = isSpam ? "🔴 SPAM" : "🟢 HAM";
  resultVerdict.className   = "result-verdict " + (isSpam ? "spam" : "ham");

  resultConf.textContent = conf;
  resultProb.textContent = prob.toFixed(4);

  // Animate the probability bar (width = P(spam))
  setTimeout(() => {
    probBar.style.width = (prob * 100).toFixed(1) + "%";
    probBar.className   = "prob-bar " + (prob >= 0.5 ? "high" : "low");
  }, 50);

  resultCard.classList.remove("hidden");
  resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function showError(msg) {
  resultVerdict.textContent = "⚠ " + msg;
  resultVerdict.className   = "result-verdict";
  resultConf.textContent    = "—";
  resultProb.textContent    = "—";
  probBar.style.width       = "0%";
  resultCard.classList.remove("hidden");
}

// ── Event listeners ────────────────────────────────────────────────────────

predictBtn.addEventListener("click", () => {
  runPrediction(textarea.value);
});

textarea.addEventListener("keydown", (e) => {
  // Ctrl+Enter or Cmd+Enter to submit
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    runPrediction(textarea.value);
  }
});

clearBtn.addEventListener("click", () => {
  textarea.value = "";
  resultCard.classList.add("hidden");
  probBar.style.width = "0%";
  textarea.focus();
});

exampleBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    const msg = btn.dataset.msg;
    textarea.value = msg;
    runPrediction(msg);
    textarea.scrollIntoView({ behavior: "smooth" });
  });
});

// ── Shake animation helper ────────────────────────────────────────────────

function shake(el) {
  el.style.animation = "none";
  el.offsetHeight;  // reflow
  el.style.animation = "shake .35s ease";
  el.addEventListener("animationend", () => {
    el.style.animation = "";
  }, { once: true });
}

// Inject keyframe for shake if not already in CSS
const style = document.createElement("style");
style.textContent = `
  @keyframes shake {
    0%,100%{transform:translateX(0)}
    20%{transform:translateX(-6px)}
    40%{transform:translateX(6px)}
    60%{transform:translateX(-4px)}
    80%{transform:translateX(4px)}
  }
`;
document.head.appendChild(style);
