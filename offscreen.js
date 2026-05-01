import { pipeline, env, RawImage } from "@huggingface/transformers";
import Tesseract from "tesseract.js";

env.backends.onnx.wasm.wasmPaths = "./";

// TGIF fine-tuned model — change this after running upload_to_hf.py
const MODEL_ID = "Patricijia/vitgpt2-gif-descriptor";

let captioner = null;
let loadPromise = null;
let ocrWorker = null;
let modelLoadTime = 0;
let ocrLoadTime = 0;

// Grid params (must match training: 6 frames, 3x2 grid)
const NUM_FRAMES = 6;
const FRAME_INDICES = [0, 3, 6, 9, 12, 15]; // from 16 evenly-spaced
const GRID_ROWS = 2;
const GRID_COLS = 3;
const CELL_SIZE = 128;
const GRID_PAD = 4;
const GRID_FINAL_W = 512;
const GRID_FINAL_H = 512;

// === Load TGIF fine-tuned model ===
async function loadModel() {
  if (captioner) return;
  if (loadPromise) return loadPromise;

  loadPromise = (async () => {
    const t0 = performance.now();
    console.log("[offscreen] Loading ViT-GPT2 TGIF on WebGPU...");
    captioner = await pipeline(
      "image-to-text",
      MODEL_ID,
      { device: "webgpu", dtype: "fp32" },
    );
    modelLoadTime = performance.now() - t0;
    console.log(`[offscreen] ViT-GPT2 TGIF ready in ${modelLoadTime.toFixed(0)}ms`);
  })();

  return loadPromise;
}

// === Load Tesseract OCR worker ===
let ocrLoadPromise = null;

async function loadOCR() {
  if (ocrWorker) return;
  if (ocrLoadPromise) return ocrLoadPromise;

  ocrLoadPromise = (async () => {
    const t0 = performance.now();
    try {
      ocrWorker = await Tesseract.createWorker("eng", 1, {
        workerPath: chrome.runtime.getURL("tesseract/worker.min.js"),
        langPath: chrome.runtime.getURL("tesseract"),
        corePath: chrome.runtime.getURL("tesseract/"),
        workerBlobURL: false,
        gzip: false,
      });
      ocrLoadTime = performance.now() - t0;
      console.log(`[offscreen] Tesseract OCR ready in ${ocrLoadTime.toFixed(0)}ms`);
    } catch (e) {
      console.error("[offscreen] OCR init failed:", e.message);
      ocrWorker = null;
    }
  })();

  return ocrLoadPromise;
}

loadModel();
loadOCR();

// === Build 6-frame 3x2 grid (same as training) ===
async function buildGrid(url) {
  const response = await fetch(url);
  const buffer = await response.arrayBuffer();

  // Decode all frames
  if (typeof ImageDecoder === "undefined") {
    // Fallback: single frame
    const blob = new Blob([buffer], { type: "image/gif" });
    const bmp = await createImageBitmap(blob);
    const canvas = new OffscreenCanvas(GRID_FINAL_W, GRID_FINAL_H);
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, GRID_FINAL_W, GRID_FINAL_H);
    ctx.drawImage(bmp, 0, 0, CELL_SIZE, CELL_SIZE);
    const gridBlob = await canvas.convertToBlob({ type: "image/png" });
    return { grid: await RawImage.fromBlob(gridBlob), frameCount: 1, framesUsed: 1 };
  }

  const decoder = new ImageDecoder({
    data: new Uint8Array(buffer),
    type: "image/gif",
  });
  await decoder.tracks.ready;
  const frameCount = decoder.tracks.selectedTrack.frameCount;

  // Select 16 evenly spaced, then pick 6 at FRAME_INDICES
  const step16 = Math.max(1, (frameCount - 1) / 15);
  const all16 = [];
  for (let i = 0; i < 16; i++) {
    const idx = Math.min(Math.floor(i * step16), frameCount - 1);
    all16.push(idx);
  }
  const selectedIndices = FRAME_INDICES.map(i => all16[Math.min(i, all16.length - 1)]);

  // Decode selected frames
  const frames = [];
  for (const idx of selectedIndices) {
    const { image } = await decoder.decode({ frameIndex: idx });
    frames.push(image);
  }
  decoder.close();

  // Build 3x2 grid
  const gw = GRID_COLS * CELL_SIZE + (GRID_COLS - 1) * GRID_PAD;
  const gh = GRID_ROWS * CELL_SIZE + (GRID_ROWS - 1) * GRID_PAD;
  const gridCanvas = new OffscreenCanvas(gw, gh);
  const ctx = gridCanvas.getContext("2d");
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, gw, gh);

  for (let k = 0; k < frames.length; k++) {
    const r = Math.floor(k / GRID_COLS);
    const c = k % GRID_COLS;
    const x = c * (CELL_SIZE + GRID_PAD);
    const y = r * (CELL_SIZE + GRID_PAD);
    ctx.drawImage(frames[k], x, y, CELL_SIZE, CELL_SIZE);
  }

  // Scale to 512x512 final
  const finalCanvas = new OffscreenCanvas(GRID_FINAL_W, GRID_FINAL_H);
  const fCtx = finalCanvas.getContext("2d");
  fCtx.fillStyle = "black";
  fCtx.fillRect(0, 0, GRID_FINAL_W, GRID_FINAL_H);

  // Center the grid
  const scale = Math.min(GRID_FINAL_W / gw, GRID_FINAL_H / gh);
  const sw = Math.round(gw * scale);
  const sh = Math.round(gh * scale);
  const ox = Math.round((GRID_FINAL_W - sw) / 2);
  const oy = Math.round((GRID_FINAL_H - sh) / 2);
  fCtx.drawImage(gridCanvas, ox, oy, sw, sh);

  const gridBlob = await finalCanvas.convertToBlob({ type: "image/png" });
  return {
    grid: await RawImage.fromBlob(gridBlob),
    frameCount,
    framesUsed: frames.length,
  };
}

// === Extract OCR frames ===
async function extractOCRFrames(url) {
  const response = await fetch(url);
  const buffer = await response.arrayBuffer();

  if (typeof ImageDecoder === "undefined") {
    const blob = new Blob([buffer], { type: "image/png" });
    const bmp = await createImageBitmap(blob);
    const canvas = new OffscreenCanvas(bmp.width, bmp.height);
    canvas.getContext("2d").drawImage(bmp, 0, 0);
    const outBlob = await canvas.convertToBlob({ type: "image/png" });
    return [URL.createObjectURL(outBlob)];
  }

  const decoder = new ImageDecoder({
    data: new Uint8Array(buffer),
    type: "image/gif",
  });
  await decoder.tracks.ready;
  const frameCount = decoder.tracks.selectedTrack.frameCount;

  const ocrIndices = [0, Math.max(0, Math.floor(frameCount / 2) - 1)];
  const urls = [];

  for (const idx of ocrIndices) {
    const { image } = await decoder.decode({ frameIndex: idx });
    const canvas = new OffscreenCanvas(image.displayWidth, image.displayHeight);
    canvas.getContext("2d").drawImage(image, 0, 0);
    const blob = await canvas.convertToBlob({ type: "image/png" });
    urls.push(URL.createObjectURL(blob));
  }

  decoder.close();
  return urls;
}

// === OCR helpers (same as base extension) ===
function cleanOcrText(rawText) {
  if (!rawText) return "";
  let text = rawText.replace(/[\r\n]+/g, " ").replace(/\s+/g, " ").trim();
  const allWords = text.split(" ").filter(w => w.length > 0);
  if (allWords.length === 0) return "";
  const shortCount = allWords.filter(w => w.replace(/[^a-zA-Z0-9]/g, "").length <= 2).length;
  if (allWords.length > 2 && shortCount / allWords.length > 0.4) return "";
  const words = allWords.filter(w => {
    const clean = w.replace(/[^a-zA-Z0-9]/g, "");
    if (clean.length < 2) return false;
    if (clean.length >= 3 && !/[aeiouyAEIOUY]/.test(clean)) return false;
    return true;
  });
  text = words.join(" ").trim();
  return text.length < 3 ? "" : text;
}

async function isolateTextByColor(blobUrl) {
  const response = await fetch(blobUrl);
  const blob = await response.blob();
  const bmp = await createImageBitmap(blob);
  const w = bmp.width, h = bmp.height;

  const srcCanvas = new OffscreenCanvas(w, h);
  const srcCtx = srcCanvas.getContext("2d");
  srcCtx.drawImage(bmp, 0, 0);
  const src = srcCtx.getImageData(0, 0, w, h);

  const whiteCanvas = new OffscreenCanvas(w, h);
  const whiteCtx = whiteCanvas.getContext("2d");
  const whiteData = whiteCtx.createImageData(w, h);

  const blackCanvas = new OffscreenCanvas(w, h);
  const blackCtx = blackCanvas.getContext("2d");
  const blackData = blackCtx.createImageData(w, h);

  for (let i = 0; i < src.data.length; i += 4) {
    const r = src.data[i], g = src.data[i + 1], b = src.data[i + 2];
    const isLight = r > 200 && g > 200 && b > 200;
    whiteData.data[i]     = isLight ? 0 : 255;
    whiteData.data[i + 1] = isLight ? 0 : 255;
    whiteData.data[i + 2] = isLight ? 0 : 255;
    whiteData.data[i + 3] = 255;
    const isDark = r < 55 && g < 55 && b < 55;
    blackData.data[i]     = isDark ? 0 : 255;
    blackData.data[i + 1] = isDark ? 0 : 255;
    blackData.data[i + 2] = isDark ? 0 : 255;
    blackData.data[i + 3] = 255;
  }

  whiteCtx.putImageData(whiteData, 0, 0);
  blackCtx.putImageData(blackData, 0, 0);

  const [whiteBlob, blackBlob] = await Promise.all([
    whiteCanvas.convertToBlob({ type: "image/png" }),
    blackCanvas.convertToBlob({ type: "image/png" }),
  ]);

  return {
    white: URL.createObjectURL(whiteBlob),
    black: URL.createObjectURL(blackBlob),
  };
}

async function extractTextFromFrame(blobUrl) {
  if (!ocrWorker) return "";
  try {
    const isolated = await isolateTextByColor(blobUrl);
    const [whiteResult, blackResult] = await Promise.all([
      ocrWorker.recognize(isolated.white),
      ocrWorker.recognize(isolated.black),
    ]);
    URL.revokeObjectURL(isolated.white);
    URL.revokeObjectURL(isolated.black);

    const whiteText = cleanOcrText(whiteResult.data.text?.trim() || "");
    const blackText = cleanOcrText(blackResult.data.text?.trim() || "");
    return whiteText.length >= blackText.length ? whiteText : blackText;
  } catch {
    return "";
  }
}

async function extractTextBestOf(blobUrls) {
  const results = await Promise.all(blobUrls.map(u => extractTextFromFrame(u)));
  blobUrls.forEach(u => URL.revokeObjectURL(u));
  return results.reduce((best, t) => t.length > best.length ? t : best, "");
}

// === Describe GIF — single grid caption (no summarization needed) ===
async function describeGif(url) {
  const t0 = performance.now();

  await Promise.all([loadModel(), loadOCR()]);
  const tModelsReady = performance.now();

  // Build grid and extract OCR frames in parallel
  const [{ grid, frameCount, framesUsed }, ocrBlobUrls] = await Promise.all([
    buildGrid(url),
    extractOCRFrames(url),
  ]);
  const tFrames = performance.now();

  // Run captioning and OCR in parallel
  const captionPromise = (async () => {
    const ct0 = performance.now();
    const result = await captioner(grid, { max_length: 20, num_beams: 4, no_repeat_ngram_size: 3 });
    const captionTime = performance.now() - ct0;
    return { caption: result[0].generated_text.trim(), captionTime };
  })();

  const ocrPromise = extractTextBestOf(ocrBlobUrls);

  const [{ caption: rawCaption, captionTime }, ocrText] = await Promise.all([captionPromise, ocrPromise]);
  const tDone = performance.now();

  let caption = rawCaption;
  if (ocrText && ocrText.length > 3) {
    caption += ". Text: " + ocrText;
  }

  const metrics = {
    modelLoadMs: Math.round(modelLoadTime),
    ocrLoadMs: Math.round(ocrLoadTime),
    frameExtractionMs: Math.round(tFrames - tModelsReady),
    totalInferenceMs: Math.round(captionTime),
    totalMs: Math.round(tDone - t0),
    framesExtracted: framesUsed,
    totalGifFrames: frameCount,
    ocrDetected: ocrText.length > 0,
    ocrText: ocrText || null,
    device: "webgpu",
  };

  console.log(
    `[offscreen] ${metrics.totalMs}ms total | ` +
    `grid: ${metrics.frameExtractionMs}ms, ` +
    `inference: ${metrics.totalInferenceMs}ms | ` +
    `${framesUsed}/${frameCount} frames | ` +
    `OCR: ${ocrText ? `"${ocrText}"` : "none"} | ` +
    `"${caption}"`
  );

  return { caption, metrics };
}

// === Message handler ===
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type !== "DESCRIBE_GIF" || message.target !== "offscreen") return;

  describeGif(message.url)
    .then(({ caption, metrics }) => sendResponse({ caption, metrics }))
    .catch((err) => {
      console.error("[offscreen] Error:", err);
      sendResponse({ error: err.message });
    });

  return true;
});
