/**
 * "What runs this demo" (plan 8.4): every model the demo uses, from the server's `/api/info`,
 * so the page always matches the server.
 */

import type { Info } from '../api';

function link(text: string, href: string): HTMLAnchorElement {
  const a = document.createElement('a');
  a.href = href;
  a.target = '_blank';
  a.rel = 'noopener';
  a.textContent = text;
  return a;
}

export function fillModels(list: HTMLElement, info: Info): void {
  const items = info.models.map((m) => {
    const li = document.createElement('li');
    li.dataset.id = m.id;
    const role = document.createElement('b');
    role.textContent = `${m.role}: `;
    li.append(role, link(m.name, m.url), ` (${m.by}), ${m.where}.`);
    setOff(li, m.off ?? null);
    return li;
  });
  for (const n of info.not_used) {
    const li = document.createElement('li');
    li.className = 'not-used';
    const name = document.createElement('b');
    name.textContent = `${n.name}: `;
    li.append(name, n.why);
    items.push(li);
  }
  const ul = document.createElement('ul');
  ul.append(...items);
  const core = document.createElement('p');
  core.className = 'small';
  core.textContent =
    `The deixis core ${info.core.version} (messages ${info.core.schema_version}), ` +
    (info.core.over_mcp ? 'reached over MCP. ' : 'called directly (MCP is down right now). ') +
    `This page ${info.version}.`;
  list.replaceChildren(ul, core);
}

function setOff(li: HTMLElement, text: string | null): void {
  li.querySelector('.off')?.remove();
  if (!text) return;
  const off = document.createElement('span');
  off.className = 'off';
  off.textContent = ` (${text})`;
  li.append(off);
}

/** Mark one model as off right now (simple mode: "off right now: today's budget is used
 * up"), or on again (`null`). */
export function setModelOff(list: HTMLElement, id: string, text: string | null): void {
  const li = list.querySelector<HTMLElement>(`li[data-id="${id}"]`);
  if (li) setOff(li, text);
}

export function modelsUnavailable(list: HTMLElement): void {
  const p = document.createElement('p');
  p.textContent = 'The server isn’t answering, so this list can’t load right now.';
  list.replaceChildren(p);
}
