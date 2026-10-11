/**
 * The plan strip (advanced-commands-plan.md 5.2): one line above the scene while a plan of
 * several steps is previewed or running.
 *
 *     Nemotron's plan:  [1 red onto blue ✓]  [2 green onto red ● doing]  [3 go home ○]
 *
 * Each box lights as it runs. A step the arm could not do turns amber, and the plan Nemotron
 * sends instead is labelled as such - which is what makes recovery legible without a word of
 * explanation. One step needs no strip: the robot says that one out loud.
 */

import type { Plan, StepState } from '../flow/flow';

/** The mark on a step's box: done, doing, to do, could not be done. */
export const MARK: Record<StepState, string> = {
  done: '✓', // check
  doing: '●', // filled circle
  todo: '○', // empty circle
  failed: '✗', // cross
};

export interface PlanLine {
  title: string;
  steps: { text: string; mark: string; state: StepState; why?: string }[];
}

/** The strip in plain data, so it can be checked without a browser. */
export function planLine(plan: Plan): PlanLine {
  return {
    title: plan.replanned ? 'Nemotron re-planned:' : 'Nemotron’s plan:',
    steps: plan.steps.map((s, i) => ({
      text: `${i + 1} ${s.label}`,
      mark: MARK[s.state],
      state: s.state,
      why: s.why,
    })),
  };
}

/** Draw it into `el` (or hide `el` when there is no plan of several steps to show). */
export function drawPlan(el: HTMLElement, plan: Plan | null): void {
  el.hidden = !plan || plan.steps.length < 2;
  if (el.hidden) {
    el.replaceChildren();
    return;
  }
  const line = planLine(plan!);
  const title = document.createElement('b');
  title.textContent = line.title;
  const boxes = line.steps.map((s) => {
    const box = document.createElement('span');
    box.className = `plan-step ${s.state}`;
    box.textContent = `${s.text} ${s.mark}`;
    if (s.why) box.title = s.why;
    return box;
  });
  el.replaceChildren(title, ...boxes);
}
