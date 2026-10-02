// 日英二言語対応。静的テキストは data-i18n / data-i18n-html / data-i18n-placeholder、
// 動的テキストは t(key, vars) で引く。バックエンドの進捗メッセージは日本語で届くので
// translateServerMessage() でクライアント側の対訳に写像する(未知の文はそのまま表示)。
const I18N = {
  ja: {
    "status.checking": "APIを確認中",
    "status.connected": "API接続済み",
    "status.disconnected": "API未接続",
    "status.model.ready": "モデル準備完了",
    "status.model.loading": "モデル読込中",
    "status.model.error": "モデルエラー",
    "status.model.missing": "モデル未配置",
    "hero.title": "言葉とイメージから、<br /><em>次の一枚</em>をつくる。",
    "hero.lede": "Qwen-Image-2.1をローカルGPUで実行。参照画像を追加すると、そのまま編集モードになります。",
    "sec.prompt": "プロンプト",
    "sec.prompt.sub": "作りたい画像、または編集内容を入力",
    "prompt.placeholder": "例：雨の東京、ネオンの反射、映画的な構図。看板には『QWEN IMAGE』と書かれている。",
    "negative.label": "ネガティブプロンプト",
    "negative.note": "CFG使用時のみ",
    "sec.reference": "参照画像",
    "reference.help": "任意・最大10枚",
    "reference.help.turbo": "任意・Turboでは最大3枚",
    "drop.title": "画像をドロップ",
    "drop.sub": "またはクリックして選択",
    "sec.settings": "生成設定",
    "sec.settings.sub": "サイズとサンプリング",
    "field.width": "幅",
    "field.height": "高さ",
    "field.steps": "ステップ",
    "field.cfg": "CFG",
    "field.seed": "シード",
    "field.seed.note": "−1でランダム",
    "generate": "生成する",
    "preview": "プレビュー",
    "empty": "生成した画像がここに表示されます",
    "progress.preparing": "準備中",
    "progress.submitting": "ジョブを登録しています",
    "job.connecting": "APIサーバーへの接続を確認しています。",
    "job.needDownload": "モデルのダウンロードが必要です。",
    "job.resident": "モデル常駐中",
    "job.loading": "モデルをGPUへ読み込んでいます。",
    "job.lazy": "初回生成時にモデルをGPUへ読み込みます。",
    "job.startApi": "APIを起動してください: {api}",
    "job.label": "ジョブ {id} · {message}",
    "job.done": "完了 · ジョブ {id}",
    "job.error": "エラー: {message}",
    "model.load": "モデルをGPUへ読み込む",
    "steps.turboFixed": "Viggle Turboでは6ステップ固定です",
    "toast.nonImage": "画像以外のファイルは除外しました。",
    "toast.maxImages": "参照画像は最大{limit}枚です。",
    "toast.multiple16": "幅と高さは16の倍数にしてください。",
    "error.generationFailed": "生成に失敗しました。",
    "server.loadingModel": "モデルを読み込んでいます",
    "server.generating": "画像を生成しています",
    "server.step": "生成中 {current} / {total} ステップ",
    "server.done": "完了",
    "server.failed": "生成に失敗しました",
  },
  en: {
    "status.checking": "Checking API",
    "status.connected": "API connected",
    "status.disconnected": "API offline",
    "status.model.ready": "Model ready",
    "status.model.loading": "Loading model",
    "status.model.error": "Model error",
    "status.model.missing": "Model not found",
    "hero.title": "Turn words and images<br />into <em>the next picture</em>.",
    "hero.lede": "Qwen-Image-2.1 on your local GPU. Add reference images to switch into editing mode.",
    "sec.prompt": "Prompt",
    "sec.prompt.sub": "Describe the image to create, or the edit to make",
    "prompt.placeholder": "e.g. Rainy Tokyo at night, neon reflections, cinematic composition. A sign reads “QWEN IMAGE”.",
    "negative.label": "Negative prompt",
    "negative.note": "used only with CFG",
    "sec.reference": "Reference images",
    "reference.help": "optional · up to 10",
    "reference.help.turbo": "optional · up to 3 with Turbo",
    "drop.title": "Drop images",
    "drop.sub": "or click to browse",
    "sec.settings": "Generation settings",
    "sec.settings.sub": "size and sampling",
    "field.width": "Width",
    "field.height": "Height",
    "field.steps": "Steps",
    "field.cfg": "CFG",
    "field.seed": "Seed",
    "field.seed.note": "−1 for random",
    "generate": "Generate",
    "preview": "Preview",
    "empty": "Generated images will appear here",
    "progress.preparing": "Preparing",
    "progress.submitting": "Submitting job",
    "job.connecting": "Checking the connection to the API server.",
    "job.needDownload": "The model needs to be downloaded first.",
    "job.resident": "model resident",
    "job.loading": "Loading the model onto the GPU.",
    "job.lazy": "The model loads onto the GPU on the first generation.",
    "job.startApi": "Start the API server: {api}",
    "job.label": "Job {id} · {message}",
    "job.done": "Done · job {id}",
    "job.error": "Error: {message}",
    "model.load": "Load model onto GPU",
    "steps.turboFixed": "Viggle Turbo runs a fixed 6 steps",
    "toast.nonImage": "Non-image files were skipped.",
    "toast.maxImages": "Up to {limit} reference images.",
    "toast.multiple16": "Width and height must be multiples of 16.",
    "error.generationFailed": "Generation failed.",
    "server.loadingModel": "Loading the model",
    "server.generating": "Generating the image",
    "server.step": "Step {current} / {total}",
    "server.done": "Done",
    "server.failed": "Generation failed",
  },
};

let currentLang = localStorage.getItem("q21.lang")
  || ((navigator.language || "en").toLowerCase().startsWith("ja") ? "ja" : "en");

function t(key, vars = {}) {
  let text = (I18N[currentLang] && I18N[currentLang][key]) || I18N.ja[key] || key;
  for (const [name, value] of Object.entries(vars)) {
    text = text.replaceAll(`{${name}}`, value);
  }
  return text;
}

// バックエンドの日本語メッセージをUI言語へ写像する
function translateServerMessage(message) {
  if (!message) return message;
  const step = message.match(/^生成中 (\d+) \/ (\d+) ステップ$/);
  if (step) return t("server.step", { current: step[1], total: step[2] });
  const map = {
    "モデルを読み込んでいます": "server.loadingModel",
    "画像を生成しています": "server.generating",
    "完了": "server.done",
    "生成に失敗しました": "server.failed",
  };
  return map[message] ? t(map[message]) : message;
}

function applyStaticI18n() {
  document.documentElement.lang = currentLang;
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    el.textContent = t(el.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-html]").forEach((el) => {
    el.innerHTML = t(el.dataset.i18nHtml);
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
    el.placeholder = t(el.dataset.i18nPlaceholder);
  });
  const toggle = document.getElementById("langToggle");
  if (toggle) toggle.textContent = currentLang === "ja" ? "EN" : "日本語";
}

function setLang(lang) {
  currentLang = lang;
  localStorage.setItem("q21.lang", lang);
  applyStaticI18n();
  document.dispatchEvent(new CustomEvent("langchange"));
}

document.addEventListener("DOMContentLoaded", () => {
  applyStaticI18n();
  const toggle = document.getElementById("langToggle");
  if (toggle) toggle.addEventListener("click", () => setLang(currentLang === "ja" ? "en" : "ja"));
});
