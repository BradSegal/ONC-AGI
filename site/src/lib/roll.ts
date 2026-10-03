/**
 * Mechanical digit counters: each digit is a column of 0–9 that rolls to its value,
 * so a number changing reads as a physical event rather than a fading label.
 */
const DIGITS = "0123456789";

export function rollTo(el: HTMLElement, text: string): void {
  if (el.dataset.rollText === text) return;
  el.dataset.rollText = text;
  if (!el.querySelector(".roll") || el.querySelectorAll(".roll__col").length !== text.length) {
    el.textContent = "";
    const wrap = document.createElement("span");
    wrap.className = "roll";
    wrap.setAttribute("aria-hidden", "true");
    for (const ch of text) {
      const col = document.createElement("span");
      col.className = "roll__col";
      if (DIGITS.includes(ch)) {
        for (const d of DIGITS) {
          const s = document.createElement("span");
          s.textContent = d;
          col.append(s);
        }
      } else {
        const s = document.createElement("span");
        s.textContent = ch;
        col.append(s);
      }
      wrap.append(col);
    }
    const sr = document.createElement("span");
    sr.className = "sr";
    el.append(wrap, sr);
  }
  const cols = el.querySelectorAll<HTMLElement>(".roll__col");
  [...text].forEach((ch, i) => {
    const d = DIGITS.indexOf(ch);
    cols[i].style.transform = d >= 0 ? `translateY(${-d}em)` : "none";
    cols[i].style.transitionDelay = `${(text.length - i) * 35}ms`;
  });
  const sr = el.querySelector<HTMLElement>(".sr");
  if (sr) sr.textContent = text;
}
