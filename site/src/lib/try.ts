/**
 * "Try one": buy patients, read the evidence, list what drives the outcome (or nothing), and get
 * scored. The score follows the benchmark's scorer exactly for this world: a list scores by its
 * first representative, a post-outcome measurement voids it, and spending beyond the reference
 * study costs efficiency. The build checks this rule against the real scorer on every list of up
 * to three measurements (scripts/prepare_data.py).
 */
import play from "../data/play.json";

const MAX = Math.max(...play.stages.flatMap((s) => Object.values(s.evidence)));
const height = (v: number) => Math.sqrt(Math.max(0, v) / MAX);
const money = (n: number) => Math.round(n).toLocaleString("en-GB");
const S = play.scoring as {
  reject: string[];
  neutral: string[];
  clusters: Record<string, number>;
  find_first: Record<string, { find: number }>;
};

/** The benchmark's score for one list on this world (depth 1). */
export function score(list: string[], spent: number) {
  const leaked = list.some((f) => S.reject.includes(f));
  const seen = new Set<number>();
  const reps: string[] = [];
  for (const f of list) {
    if (S.neutral.includes(f) || seen.has(S.clusters[f])) continue;
    seen.add(S.clusters[f]);
    reps.push(f);
  }
  const abstained = reps.length === 0;
  const find = leaked || abstained ? 0 : S.find_first[reps[0]].find;
  const efficiency = spent <= 0 ? 1 : Math.min(1, play.reference_cost / spent);
  return { leaked, abstained, find, efficiency, first: reps[0] };
}

export function tryOne(root: HTMLElement): void {
  const bars = [...root.querySelectorAll<HTMLButtonElement>(".bar")];
  const buys = [...root.querySelectorAll<HTMLButtonElement>(".buy")];
  const listEl = root.querySelector<HTMLOListElement>(".try__list")!;
  const spendEl = root.querySelector<HTMLElement>("[data-spend]")!;
  const result = root.querySelector<HTMLElement>(".try__result")!;
  const submit = root.querySelector<HTMLButtonElement>("[data-submit]")!;
  const nothing = root.querySelector<HTMLButtonElement>("[data-submit-empty]")!;
  const reset = root.querySelector<HTMLButtonElement>("[data-reset]")!;
  const truth = root.dataset.truth!;
  const twin = root.dataset.twin!;
  let stage = -1;
  let list: string[] = [];
  let done = false;

  const render = () => {
    root.classList.toggle("has-data", stage >= 0);
    buys.forEach((b, i) => b.setAttribute("aria-pressed", String(i === stage)));
    const ev = stage >= 0 ? (play.stages[stage].evidence as Record<string, number>) : null;
    bars.forEach((b) => {
      const id = b.dataset.id!;
      b.style.setProperty("--h", ev ? height(ev[id]).toFixed(4) : "0");
      b.disabled = !ev || done;
      const rank = list.indexOf(id);
      b.setAttribute("aria-pressed", String(rank >= 0));
      b.querySelector(".bar__rank")!.textContent = rank >= 0 ? String(rank + 1) : "";
      b.title = ev ? `${id}: −log10 p = ${ev[id]}` : id;
      // only what rises above the line turns red
      b.classList.toggle("is-over", !!ev && ev[id] > play.agent_threshold && !b.classList.contains("bar--post"));
    });
    spendEl.textContent = stage >= 0 ? `$${money(play.stages[stage].spent)} of $${money(play.budget)} spent on ${play.stages[stage].patients} patients` : `Nothing bought yet · $${money(play.budget)} available`;
    listEl.innerHTML = list.length ? list.map((id) => `<li><code>${id}</code></li>`).join("") : "<li class=\"try__list-empty\">Your list is empty</li>";
    submit.disabled = stage < 0 || !list.length || done;
    nothing.disabled = stage < 0 || done;
  };

  buys.forEach((b, i) =>
    b.addEventListener("click", () => {
      if (done) return;
      // buying more is allowed; patients already bought stay bought, so the spend only grows
      stage = Math.max(stage, i);
      render();
    }),
  );
  bars.forEach((b) =>
    b.addEventListener("click", () => {
      if (done || stage < 0) return;
      const id = b.dataset.id!;
      list = list.includes(id) ? list.filter((x) => x !== id) : [...list, id];
      render();
    }),
  );
  const finish = (submitted: string[]) => {
    done = true;
    const spent = play.stages[stage].spent;
    const s = score(submitted, spent);
    root.classList.add("is-revealed");
    bars.forEach((b) => {
      const id = b.dataset.id!;
      b.classList.toggle("is-truth", id === truth);
      b.classList.toggle("is-twin", id === twin);
    });
    const why = s.leaked
      ? `Your list includes a measurement taken after the outcome (${submitted.find((f) => S.reject.includes(f))}). It predicts the outcome only because it comes after it, so the answer earns no credit.`
      : s.abstained
        ? "You returned nothing. This world has a planted cause, so the empty answer misses it; on a world with no signal, it would be exactly right."
        : s.find > 0
          ? `Your first measurement, ${s.first}, ${s.first === truth ? "is the planted driver" : "is its near-duplicate, which the data cannot tell apart from the driver, so it counts"}.`
          : `Your first measurement, ${s.first}, is not the cause. Only the first counts here, because this world has one recoverable cause.`;
    result.hidden = false;
    result.innerHTML = `
      <div class="try__score"><span>Find</span><b>${s.find.toFixed(2)}</b></div>
      <div class="try__score"><span>Efficiency</span><b>${s.efficiency.toFixed(2)}</b></div>
      <p>${why}</p>
      <p>You spent $${money(spent)}; the reference study costs $${money(play.reference_cost)}${spent > play.reference_cost ? ", so efficiency falls" : ""}. The planted driver is <code>${truth}</code>; its near-duplicate <code>${twin}</code> is accepted too.</p>
      <p><a href="#journey">See how a recorded baseline played this world ↓</a></p>`;
    render();
  };
  submit.addEventListener("click", () => finish(list));
  nothing.addEventListener("click", () => finish([]));
  reset.addEventListener("click", () => {
    stage = -1;
    list = [];
    done = false;
    result.hidden = true;
    root.classList.remove("is-revealed");
    bars.forEach((b) => b.classList.remove("is-truth", "is-twin"));
    render();
  });
  render();
}
