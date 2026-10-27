/**
 * The page: the arm, the table and the blocks; the finger as a pointer (plan W4: the webcam,
 * the fingertip dot, samples in table mm), or the mouse; the voice (W5: push-to-talk, the clip
 * on the page clock); one command to the server (W6, W8: words, Nemotron's calls, the core's
 * places over MCP); and the flow (W9: preview, "This one, here?", yes, the arm moves).
 *
 * W10 adds the page around it: the start card, the thinking strip, the wiring panel, the
 * try-saying chips, the models list (plan 8.4), and typing instead of talking. W11: the day's
 * limits on our key, the banner when it is off, and the visitor's own key (plan 8.1, 8.2).
 *
 * 6 Oct (test set-up, final/web-demo.md items 1 to 20): the robot says what it understood
 * ("Move the red block here?"), the coach line, the hover outline, "Done.", one prompt for
 * the camera and the microphone, plain error lines, the hand tracker loading during the start
 * card, one look (amber you, dashed blue guess), tokens and cost, the word-rules line, and
 * events for the server's log (the guess shown, yes, done) so M30 and M31 come from it.
 *
 * `?hands=demo` replays a made-up pointing hand instead of the camera (screenshots, tests);
 * it and `?start=skip` skip the start card.
 */

import {
  type CommandMeta,
  type CommandReply,
  ServerDown,
  getInfo,
  sendAnswer,
  sendCommand,
  sendEvent,
  sendReplan,
  sendStep,
} from './api';
import { MicCapture } from './audio/capture';
import type { Clip } from './audio/clip';
import { PushToTalk } from './audio/talk';
import { Flow, type FlowServer, type Plan } from './flow/flow';
import { MousePointer } from './hand/mouse';
import { Arm, BLOCK_IDS, type BlockId } from './sim/arm';
import { fetchPack, loadScene, mujoco } from './sim/mujoco';
import { type Hit, pointAt } from './sim/pick';
import { TableView } from './sim/render';
import { BASE_MM, BLOCK_MM, CLEAR_OF_SQUARE_MM, REACH_MM, SQUARES, type Vec3, onMat, tableToWorld } from './sim/table';
import { COACH, Coach, onPhone } from './ui/coach';
import { summaryLine } from './ui/summary';
import { KeyPanel, ownKey } from './ui/key';
import { fillModels, modelsUnavailable } from './ui/models';
import { drawPlan } from './ui/plan';
import { type StripModel, drawStrip, stripModel } from './ui/strip';
import { Wiring } from './ui/wiring';
import { setMuted, speak } from './voice';

/** A clip whose loudest sample is below this is silence (a muted or wrong microphone). */
const SILENT_DB = -45;

/**
 * Random spots on the reach mat, at least 60 mm apart and clear of the painted squares, so
 * "put each one on its own colour" always starts with three blocks on bare mat and ends with
 * three blocks on paint.
 */
function layout(n: number, rand = Math.random): [number, number][] {
  const out: [number, number][] = [];
  const clearOfPaint = (x: number, y: number) =>
    SQUARES.every((s) => Math.hypot(s.xMm - x, s.yMm - y) > CLEAR_OF_SQUARE_MM);
  for (let tries = 0; out.length < n && tries < 2000; tries++) {
    const r = REACH_MM.min + 15 + rand() * (REACH_MM.max - REACH_MM.min - 30);
    const a = (rand() - 0.5) * Math.PI * 0.9;
    const x = BASE_MM.x + r * Math.sin(a);
    const y = BASE_MM.y - r * Math.cos(a);
    if (!onMat(x, y) || !clearOfPaint(x, y)) continue;
    if (out.every(([u, v]) => Math.hypot(u - x, v - y) > 60)) out.push([x, y]);
  }
  return out;
}

async function start(): Promise<void> {
  const status = document.getElementById('status')!;
  const stage = document.getElementById('stage')!;
  const canvas = document.getElementById('scene') as HTMLCanvasElement;
  const talkBtn = document.getElementById('talk-btn') as HTMLButtonElement;
  const goBtn = document.getElementById('go-btn') as HTMLButtonElement;
  const bubble = document.getElementById('bubble')!;
  const now = () => performance.now() / 1000;
  const say = (text: string) => {
    status.textContent = text;
  };
  const params = new URLSearchParams(location.search);
  // `?quiet`: the robot writes but does not speak (screenshots, the film, a quiet room).
  if (params.has('quiet')) setMuted(true);
  document.getElementById('phone-line')!.hidden = !onPhone();
  const coach = new Coach();
  const coachEl = document.getElementById('coach')!;
  let coachShown = '';
  const keys = new KeyPanel(document.getElementById('models-list')!);
  showInfo(keys, (ok) => coach.server(ok));
  const [mj, files] = await Promise.all([
    mujoco(),
    fetchPack('/so101.pack', (f) =>
      say(f < 1 ? `Loading the arm… ${Math.round(f * 100)} %` : 'Starting the arm…'),
    ),
  ]);
  const { model, data } = loadScene(mj, files);
  const arm = new Arm(mj, model, data);
  const view = new TableView(canvas, mj, model);
  const blockOfGeom = new Map<number, BlockId>([...arm.blocks.values()].map((b) => [b.geom, b.id]));

  const reset = () => {
    arm.goHomeNow();
    layout(BLOCK_IDS.length).forEach(([x, y], i) =>
      arm.putBlock(BLOCK_IDS[i], x, y, Math.random() * Math.PI),
    );
  };
  reset();

  const fit = () => view.resize(canvas.clientWidth, canvas.clientHeight);
  new ResizeObserver(fit).observe(canvas);
  fit();

  // The flow (plan 5.4, W9): what the server resolved, checked robot-side, shown, then done.
  const glowing = new Set<BlockId>();
  // The bubble floats over the scene, so it fades after a while, except while the robot waits
  // for an answer (a preview or a question).
  let quiet = 0;
  const robotSays = (text: string) => {
    bubble.textContent = text;
    bubble.classList.add('on');
    speak(text);
    clearTimeout(quiet);
    quiet = window.setTimeout(() => {
      if (flow.state.kind !== 'preview' && flow.state.kind !== 'asking') bubble.classList.remove('on');
    }, 6000);
  };
  // The live wiring panel and the plan strip (advanced-commands-plan.md 5.1, 5.2).
  const wiring = new Wiring(
    document.getElementById('wiring')!,
    document.getElementById('wiring-replay') as HTMLButtonElement,
  );
  const planStrip = document.getElementById('plan-strip')!;

  /* "Scroll down to see what each part said" (rahul, 11 Oct). The panel is under the scene, so
     when a command has finished its work, say that there is something to see - unless it is
     already on the screen, which it is on a tall window. */
  const wiringEl = document.getElementById('wiring')!;
  const seeInside = document.getElementById('see-inside') as HTMLButtonElement;
  let panelOnScreen = false;
  /** Something actually moved during this command, so there is a story worth reading. */
  let somethingHappened = false;
  new IntersectionObserver(
    ([seen]) => {
      panelOnScreen = seen.isIntersecting;
      if (panelOnScreen) seeInside.hidden = true;
    },
    { threshold: 0.3 },
  ).observe(wiringEl);
  seeInside.addEventListener('click', () => {
    seeInside.hidden = true;
    wiringEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });

  /**
   * The parts of a plan the page asks for as it goes (`/api/step`, `/api/replan`): each one
   * sends the table as it is at that moment, and its trace rows go straight on the panel.
   */
  const server: FlowServer = {
    step: async (index, things, context, showing) => {
      const reply = await sendStep(index, things, context, showing);
      wiring.add(reply.trace);
      return reply;
    },
    replan: async (index, why, things, context) => {
      const reply = await sendReplan(index, why, things, context, ownKey());
      wiring.add(reply.trace);
      keys.reply(reply);
      wiring.summary(summaryLine(reply));
      return reply;
    },
  };

  const flow = new Flow(
    arm,
    {
      speak: robotSays,
      glow: (ids) => {
        for (const id of BLOCK_IDS) {
          const on = ids.includes(id);
          if (on === glowing.has(id)) continue;
          view.glow(arm.blocks.get(id)!.geom, on);
          if (on) glowing.add(id);
          else glowing.delete(id);
        }
      },
      ghosts: (items) => {
        view.showGhosts(
          items.map(({ block, place }) => {
            const [gx, gy] = tableToWorld(place.xMm, place.yMm);
            return { at: [gx, gy, (place.zMm + BLOCK_MM / 2) / 1000] as Vec3, geom: arm.blocks.get(block)!.geom };
          }),
        );
      },
      plan: (plan) => drawPlan(planStrip, plan),
      trace: (rows) => wiring.add(rows),
      changed: (s) => {
        stage.dataset.flow = s.kind;
        goBtn.disabled = s.kind !== 'preview';
        if (s.kind === 'acting') bubble.classList.remove('on'); // the question is answered
        // The command is over and the arm moved something: there is a diagram to read.
        if (s.kind === 'idle' && somethingHappened) {
          somethingHappened = false;
          seeInside.hidden = panelOnScreen;
        }
      },
      event: (kind, data) => {
        // For the server's log: M30 (right first time) and M31 (last word to the guess).
        const extra: { latency_s?: number; via?: string } = {};
        if ((kind === 'preview' || kind === 'plan') && lastWordT1 !== null) {
          extra.latency_s = Math.round((now() - lastWordT1) * 1000) / 1000;
        }
        if (kind === 'go') extra.via = goVia;
        if (kind === 'go' || kind === 'cancel' || kind === 'stop') goVia = 'voice';
        sendEvent({ kind, ...data, ...extra });
        if (kind === 'done') somethingHappened = true;
        if (kind === 'done' && coach.moved(now())) {
          document.getElementById('chips-label')!.textContent = 'Now say it your own way:';
        }
      },
    },
    { confirmFirst: true, now, server },
  );
  /** How the next "yes" came: by voice unless a button or a key says otherwise. */
  let goVia = 'voice';
  /** The page time the last spoken command's last word ended (null for typed commands). */
  let lastWordT1: number | null = null;

  // The thinking strip (plan 5.2), inside the wiring panel.
  const stripEl = document.getElementById('strip')!;
  const stripSvg = document.getElementById('strip-svg') as unknown as SVGSVGElement;
  // Only what the person said and where they pointed; the robot's words go in its bubble.
  const HINT = 'Speak while you point: your words and your pointing show here, on one timeline.';
  let shown: StripModel | string = HINT;
  const showStrip = (what: StripModel | string, busy = false) => {
    shown = what;
    stripEl.classList.toggle('busy', busy);
    drawStrip(stripSvg, what);
  };
  new ResizeObserver(() => drawStrip(stripSvg, shown)).observe(stripSvg);
  showStrip(shown);

  const failed = (e: unknown) => {
    console.error(e);
    showStrip(HINT);
    if (e instanceof ServerDown) {
      coach.server(false); // item 14: "The robot's brain isn't answering right now."
      sendEvent({ kind: 'server_down' });
      void flow.handle({ result: null, say: COACH.serverDown });
      say(COACH.serverDown);
      return;
    }
    void flow.handle({ result: null, say: "Sorry, that didn't work. Please try again." });
    say(`Something went wrong (${e instanceof Error ? e.message : e}).`);
  };

  let lastReply: CommandReply | null = null;
  /** Clicks after this page time belong to the next typed command. */
  let usedClicksUntil = -Infinity;
  /** One command, spoken (with its clip) or typed: the server, then the strip, the panel and
   * the flow. */
  const runCommand = async (meta: CommandMeta, wav?: ArrayBuffer, prefix = '') => {
    usedClicksUntil = meta.t ?? now();
    flow.thinking();
    coach.sent();
    wiring.clear(); // the panel shows this command, from the first word to the last move
    somethingHappened = false;
    seeInside.hidden = true;
    showStrip(wav ? 'Turning your voice into words…' : 'Thinking…', true);
    const reply = await sendCommand(meta, wav, ownKey(), () => say('One moment…'));
    coach.server(true);
    lastReply = reply;
    lastWordT1 = wav && reply.words.length ? reply.words[reply.words.length - 1].t1 : null;
    const keyNote = keys.reply(reply);
    const heard = reply.words.map((w) => w.text).join(' ');
    wiring.add(reply.trace);
    wiring.summary(summaryLine(reply));
    showStrip(heard ? stripModel(reply.words, reply.result) : HINT);
    const line = wav
      ? heard
        ? `${prefix}You said: “${heard}”`
        : `${prefix}I didn't hear any words.`
      : `You typed: “${heard}”`;
    say(keyNote ? `${keyNote} ${line}` : line);
    await flow.handle(heard ? reply : { result: null, say: "Sorry, I didn't hear that. Try again?" });
  };
  const hitOf = (e: MouseEvent): Hit | null => {
    const r = canvas.getBoundingClientRect();
    return pointAt(view, (e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height, blockOfGeom);
  };
  // The mouse is the pointer (plan 5.3): sampled like a fingertip (always, so the 2.5 s
  // before the speech are there too), and each click a hold.
  const mouse = new MousePointer((u, v) => pointAt(view, u, v, blockOfGeom));
  canvas.addEventListener('pointermove', (e) => {
    const r = canvas.getBoundingClientRect();
    mouse.move((e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height);
  });
  canvas.addEventListener('pointerleave', () => mouse.leave());
  setInterval(() => mouse.sample(now()), 1000 / 30);
  canvas.addEventListener('click', async (e) => {
    const hit = hitOf(e);
    if (!hit) return;
    mouse.click(now(), hit);
    // A click on one of the glowing choices answers the robot's question.
    const pick = hit.block ? flow.candidateOf(hit.block) : null;
    if (pick === null) return;
    usedClicksUntil = now(); // this click was the answer, not part of the next command
    flow.thinking();
    try {
      const reply = await sendAnswer(pick);
      wiring.add(reply.trace);
      if (lastReply) showStrip(stripModel(lastReply.words, lastReply.result, reply.result));
      await flow.handle(reply);
    } catch (err) {
      failed(err);
    }
  });

  // Typing instead of talking (plan 5.3): click the blocks and spots in order, then type; the
  // server gives the k-th pointing word the k-th click's time.
  const typeForm = document.getElementById('type-form') as HTMLFormElement;
  const typeBox = document.getElementById('type-box') as HTMLInputElement;
  const typeToggle = document.getElementById('type-toggle') as HTMLButtonElement;
  const isTyping = () => document.body.classList.contains('typing');
  const setTyping = (on: boolean) => {
    document.body.classList.toggle('typing', on);
    typeToggle.textContent = on ? 'Talk instead' : 'Type instead';
    if (on) typeBox.focus();
  };
  typeForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const text = typeBox.value.trim();
    if (!text) return;
    const t = now();
    const holds = mouse.clicksBetween(usedClicksUntil, t);
    typeBox.value = '';
    void runCommand({ text, t, holds, samples: [], things: flow.things(), ...flow.request() }).catch(failed);
  });

  // The voice (plan W5): push-to-talk; the clip with its pre-roll and tail, on the page clock.
  let mic: MicCapture | null = null;
  let micStarting: Promise<string> | null = null;
  let lastClip: Clip | null = null;
  const startMic = (given?: MediaStream): Promise<string> =>
    (micStarting ??= MicCapture.start(given).then(
      (m) => {
        mic = m;
        talkBtn.classList.add('ready');
        return 'Hold Talk (or the space bar) and speak.';
      },
      (e) => {
        micStarting = null;
        console.error(e);
        setTyping(true);
        coach.tell(COACH.noMic, now());
        sendEvent({ kind: 'mic_error', why: e instanceof Error ? e.name : String(e) });
        return COACH.noMic;
      },
    ));
  typeToggle.addEventListener('click', () => {
    if (!isTyping()) return setTyping(true);
    if (mic) return setTyping(false);
    void startMic().then((msg) => {
      if (mic) setTyping(false);
      say(msg);
    });
  });
  for (const chip of document.querySelectorAll<HTMLButtonElement>('#chips .chip')) {
    // A chip only shows the words (plan 5.2); in typing mode it fills the box.
    chip.addEventListener('click', () => {
      const text = chip.textContent ?? '';
      if (isTyping()) {
        typeBox.value = text;
        typeBox.focus();
      } else {
        say(`Point, then hold Talk (or the space bar) and say “${text}”.`);
      }
    });
  }
  const talk = new PushToTalk({
    start: (t) => {
      mic?.begin(t);
      talkBtn.classList.add('on');
      say('Listening… let go to stop (after a quick tap: tap again).');
      showStrip('Listening… point while you speak.', true);
    },
    stop: (t, reason) => {
      talkBtn.classList.remove('on');
      talkBtn.style.setProperty('--level', '0');
      if (!mic) return;
      if (reason === 'cancel') {
        mic.recorder.cancel();
        say('Stopped listening.');
        showStrip(HINT);
        return;
      }
      const long = reason === 'long' ? 'That was long, so I stopped listening. ' : '';
      void mic
        .end(t)
        .then(async (clip) => {
          lastClip = clip;
          if (clip.peakDb < SILENT_DB) {
            // Item 13: a muted or wrong microphone records silence; don't send it.
            coach.tell(COACH.silent, now());
            say(COACH.silent);
            showStrip(HINT);
            sendEvent({ kind: 'silent' });
            return;
          }
          say(`${long}Thinking…`);
          // The pointing from 2.5 s before the speech to 0.8 s after it (plan 4.2); the clip
          // already starts 0.5 s early and ends 0.4 s late.
          const from = clip.t0 - 2;
          const to = clip.t0 + clip.duration + 0.4;
          const pointing = mouse.samples;
          const meta = {
            audio_t0: clip.t0,
            t: now(),
            samples: pointing.between(from, to),
            holds: mouse.clicksBetween(from, to),
            things: flow.things(),
            ...flow.request(),
          };
          await runCommand(meta, clip.wav, long);
        })
        .catch(failed);
    },
  });
  /** A Talk press with no microphone yet asks for it; that press doesn't record. */
  const pressTalk = () => {
    if (mic) talk.press(now());
    else void startMic().then(say);
  };
  talkBtn.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;
    talkBtn.setPointerCapture(e.pointerId);
    pressTalk();
  });
  talkBtn.addEventListener('pointerup', () => talk.release(now()));
  talkBtn.addEventListener('pointercancel', () => talk.release(now()));
  talkBtn.addEventListener('contextmenu', (e) => e.preventDefault());

  // Mouse only (rahul, 27 Oct): no start card and no camera, so the page is ready as soon
  // as the arm is, and the first Talk press is what asks for the microphone.
  sendEvent({ kind: 'start', via: 'mouse' });
  coach.started(now());

  // The buttons (plan 5.2): Go (same as yes), Undo, Reset blocks, Stop (also Esc).
  const stopAll = () => {
    talk.cancel(now());
    flow.stop();
    robotSays('Stopped.');
  };
  const newBlocks = () => {
    flow.stop();
    reset();
    flow.reset();
  };
  goBtn.addEventListener('click', () => {
    goVia = 'button';
    void flow.go();
  });
  document.getElementById('undo-btn')!.addEventListener('click', () => void flow.undo());
  document.getElementById('reset-btn')!.addEventListener('click', newBlocks);
  document.getElementById('stop-btn')!.addEventListener('click', stopAll);
  goBtn.disabled = true;

  const typing = (e: KeyboardEvent) =>
    e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement;
  window.addEventListener('keyup', (e) => {
    if (e.code === 'Space' && !typing(e)) talk.release(now());
  });
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') return stopAll(); // Stop works even while typing
    if (e.code === 'Space' && !typing(e)) {
      e.preventDefault(); // no page scroll, no "click" on a focused button
      if (!e.repeat && !isTyping()) pressTalk();
      return;
    }
    if (typing(e)) return;
    if (e.key === 'Enter') {
      goVia = 'key';
      void flow.go();
    }
    if (e.key === 'r') newBlocks();
  });

  /** The look check holds the hover outline still (`__sim.markBlock`). */
  let marked = false;
  const drawCoach = (t: number) => {
    const line = coach.line(t);
    const key = line ? line.text : '';
    if (key === coachShown) return;
    coachShown = key;
    coachEl.hidden = !line;
    coachEl.replaceChildren();
    if (line) coachEl.append(line.text);
  };

  // Physics in real time: 5 ms steps, as many as the frame took (at most 0.1 s' worth).
  const dt = model.opt.timestep;
  let last = performance.now();
  let owed = 0;
  const loop = () => {
    // performance.now(), not the frame's timestamp: some browsers (headless Chrome) give
    // frame times on another origin, which would stall the physics.
    const now = performance.now();
    owed += Math.max(0, Math.min((now - last) / 1000, 0.1));
    last = now;
    for (; owed >= dt; owed -= dt) {
      arm.update();
      mj.mj_step(model, data);
    }
    talk.tick(now / 1000);
    if (talk.listening && mic) talkBtn.style.setProperty('--level', Math.min(1, mic.level * 8).toFixed(2));
    // What the dot (or the mouse) is on: a thin amber outline (item 3).
    const hit = mouse.hit;
    const on = hit?.block ?? null;
    if (!marked) view.hover(on ? arm.blocks.get(on)!.geom : null);
    coach.hover(on !== null);
    drawCoach(now / 1000);
    view.sync(data);
    view.render();
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
  say('Point at a block, hold Talk (or the space bar) and say “put that here”.');
  Object.assign(window, {
    __sim: {
      mj, model, data, view, arm, mouse, talk, flow, coach, now,
      mic: () => mic,
      /** For the look check: draw a plan strip without the model (scripts/plan-shots.mjs). */
      plan: (plan: Plan | null) => drawPlan(planStrip, plan),
      lastClip: () => lastClip,
      lastReply: () => lastReply,
      startMic,
      /** For the look check (M34): the hover outline and the guess on one block. */
      markBlock: (id: BlockId, guess = true) => {
        marked = true;
        view.hover(arm.blocks.get(id)!.geom);
        for (const other of BLOCK_IDS) view.glow(arm.blocks.get(other)!.geom, guess && other === id);
      },
    },
  });
}

/** The page's name, the models list (plan 8.4) and the day's limits (8.1), from the server. */
function showInfo(keys: KeyPanel, server: (ok: boolean) => void): void {
  const list = document.getElementById('models-list')!;
  getInfo().then(
    (info) => {
      server(true);
      fillModels(list, info);
      // The "use your own Nebius key" box is hidden for the judges' demo (item 16).
      document.getElementById('own-key')!.hidden = !info.show_own_key;
      keys.allowOwnKey = !!info.show_own_key;
      keys.show(info.limits);
      for (const el of document.querySelectorAll('.page-title')) el.textContent = info.title;
      document.title = `${info.title}: put that here`;
    },
    (e) => {
      console.error(e);
      if (e instanceof ServerDown) server(false);
      modelsUnavailable(list);
    },
  );
}

start().catch((e) => {
  const text = `Could not start: ${e instanceof Error ? e.message : e}`;
  document.getElementById('status')!.textContent = text;
  console.error(e);
});
