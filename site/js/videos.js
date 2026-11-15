// Phone clips: load and play only while on screen; under reduced motion, show controls instead.
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

export function mountVideos() {
  const videos = [...document.querySelectorAll("video[data-src]")];
  const attach = (v) => { if (!v.src) v.src = v.dataset.src; };

  if (reduced || !("IntersectionObserver" in window)) {
    videos.forEach((v) => { v.controls = true; attach(v); });
    return;
  }
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      const v = e.target;
      if (e.isIntersecting) {
        attach(v);
        v.play().catch(() => { v.controls = true; });
      } else if (!v.paused) {
        v.pause();
      }
    }
  }, { rootMargin: "120px 0px", threshold: 0.25 });
  videos.forEach((v) => io.observe(v));
}
