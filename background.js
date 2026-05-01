// Service worker — manages offscreen document and routes messages.

async function ensureOffscreen() {
  const contexts = await chrome.runtime.getContexts({
    contextTypes: ["OFFSCREEN_DOCUMENT"],
  });
  if (contexts.length > 0) return;

  await chrome.offscreen.createDocument({
    url: "offscreen.html",
    reasons: ["WORKERS"],
    justification: "ML model inference with WebGPU",
  });
}

ensureOffscreen();

let runCounter = 0;

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "SAVE_METRICS") {
    if (message.format === "json") runCounter++;
    const mimeType = message.format === "json" ? "application/json" : "text/csv";
    const dataUrl = `data:${mimeType};base64,${btoa(unescape(encodeURIComponent(message.data)))}`;
    const filename = `vitgpt2-tgif-benchmark-run${runCounter}.${message.format}`;

    chrome.downloads.download({
      url: dataUrl,
      filename: `gif-benchmarks/${filename}`,
      conflictAction: "overwrite",
      saveAs: false,
    }, () => {
      sendResponse({ ok: true, runId: runCounter });
    });

    return true;
  }

  if (message.type !== "DESCRIBE_GIF") return;
  if (message.target === "offscreen") return;

  ensureOffscreen()
    .then(() =>
      chrome.runtime.sendMessage({ ...message, target: "offscreen" })
    )
    .then((response) => sendResponse(response))
    .catch((err) => {
      console.error("[background] Error:", err);
      sendResponse({ error: err.message });
    });

  return true;
});
