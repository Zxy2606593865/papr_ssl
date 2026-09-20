/* Agent Wave presentation only: no network calls or recognition state changes. */
(() => {
  "use strict";
  const video = document.getElementById("agentWave");
  const button = document.getElementById("motionToggle");
  const status = document.getElementById("motionStatus");
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  let pausedByUser = reduced.matches;
  let visible = true;
  let failed = false;
  let playbackRevision = 0;

  function renderButton() {
    button.textContent = failed ? "动效未加载" : pausedByUser ? "播放动效 ▷" : "暂停动效 Ⅱ";
    button.setAttribute("aria-pressed", String(pausedByUser));
    button.disabled = failed;
    status.textContent = failed ? "显示静态画面 · 不影响语音识别" : "Agent Wave · 原版动效 / 非音频数据";
  }
  async function syncVideo() {
    const revision = ++playbackRevision;
    if (failed || pausedByUser || document.hidden || !visible) {
      video.pause();
      renderButton();
      return;
    }
    try {
      await video.play();
      if (revision !== playbackRevision && (pausedByUser || document.hidden || !visible)) video.pause();
    } catch {
      if (revision === playbackRevision) pausedByUser = true;
    }
    renderButton();
  }
  button.addEventListener("click", () => { pausedByUser = !pausedByUser; syncVideo(); });
  reduced.addEventListener("change", () => { pausedByUser = reduced.matches; syncVideo(); });
  document.addEventListener("visibilitychange", syncVideo);
  video.addEventListener("error", () => { failed = true; syncVideo(); });
  video.querySelector("source").addEventListener("error", () => { failed = true; syncVideo(); });
  new IntersectionObserver(entries => {
    visible = entries[0].isIntersecting;
    syncVideo();
  }, { threshold: 0.05 }).observe(video);
  video.muted = true;
  if (reduced.matches) video.removeAttribute("autoplay");
  syncVideo();

  document.querySelectorAll(".appear").forEach(element => {
    element.addEventListener("animationend", event => {
      if (event.target === element) element.classList.add("is-in");
    });
  });
  requestAnimationFrame(() => requestAnimationFrame(() => {
    document.querySelectorAll(".appear").forEach(element => {
      if (!element.getAnimations().some(animation => ["running", "finished"].includes(animation.playState))) element.classList.add("is-in");
    });
  }));
})();
