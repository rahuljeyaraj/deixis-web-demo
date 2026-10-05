/**
 * The robot's voice (plan 4.1): the browser's own speech synthesis. Each new line cuts off
 * the one before. Silent where the browser has none.
 */

let muted = false;

export function setMuted(on: boolean): void {
  muted = on;
  if (on && 'speechSynthesis' in window) speechSynthesis.cancel();
}

export function speak(text: string): void {
  if (muted || typeof window === 'undefined' || !('speechSynthesis' in window)) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.lang = 'en-US';
  u.rate = 1.05;
  speechSynthesis.speak(u);
}
