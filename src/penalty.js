const BEST_KEY = 'mufc-penalty-best';
const STREAK_KEY = 'mufc-penalty-streak-best';
const SOUND_KEY = 'mufc-penalty-sound';
const ROUND_SIZE = 5;
const DT = 1 / 120;
const SWEET_MIN = 0.72;
const SWEET_MAX = 0.86;
const OVERCHARGE_MIN = 0.88;

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const lerp = (a, b, t) => a + (b - a) * t;

export const PENALTY_PHYSICS = Object.freeze({
  goalWidth: 7.32,
  goalHeight: 2.44,
  goalZ: 11,
  netDepth: 1.45,
  ballRadius: 0.11,
  postRadius: 0.06,
  gravity: 9.81,
  drag: 0.0055,
  timeStep: DT,
});

export const PENALTY_MODES = Object.freeze({
  placement: {
    name: '推射',
    baseSpeed: 11,
    addedSpeed: 13,
    dispersion: 0.045,
    spinFactor: 0.22,
    detail: '脚内侧精准推射，落点稳定，适合瞄准死角',
  },
  power: {
    name: '大力',
    baseSpeed: 17,
    addedSpeed: 19,
    dispersion: 0.18,
    spinFactor: 0.16,
    detail: '正脚背全力爆射，球速极快，过载易产生偏差',
  },
  curl: {
    name: '弧线',
    baseSpeed: 13,
    addedSpeed: 14,
    dispersion: 0.085,
    spinFactor: 1,
    detail: '搓出强烈侧旋弧线，绕开守门员扑救手型',
  },
});

export const PENALTY_DIFFICULTIES = Object.freeze({
  standard: {
    name: '标准',
    reaction: 0.18,
    reactionVariation: 0.065,
    error: 0.22,
    moveSpeed: 3.6,
    diveSpeed: 6.8,
    acceleration: 28,
    jumpSpeed: 2.8,
    detail: '常规门将反应与扑救半径',
  },
  worldclass: {
    name: '世界级',
    reaction: 0.135,
    reactionVariation: 0.045,
    error: 0.13,
    moveSpeed: 4.1,
    diveSpeed: 7.55,
    acceleration: 33,
    jumpSpeed: 3.05,
    detail: '预判更快，死角覆盖更广',
  },
  legend: {
    name: '传奇',
    reaction: 0.095,
    reactionVariation: 0.03,
    error: 0.075,
    moveSpeed: 4.5,
    diveSpeed: 8.25,
    acceleration: 38,
    jumpSpeed: 3.25,
    detail: '梦剧场门神，极致反应与臂展考验',
  },
});

const QUICK_ZONES = Object.freeze({
  tl: { label: '左上死角', x: -2.92, y: 1.98 },
  tr: { label: '右上死角', x: 2.92, y: 1.98 },
  bl: { label: '左下死角', x: -3.02, y: 0.42 },
  br: { label: '右下死角', x: 3.02, y: 0.42 },
  panenka: { label: '中路挑射', x: 0, y: 1.86 },
});

function randomGenerator(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6D2B79F5;
    let v = Math.imul(state ^ (state >>> 15), 1 | state);
    v ^= v + Math.imul(v ^ (v >>> 7), 61 | v);
    return ((v ^ (v >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(random) {
  return (
    Math.sqrt(-2 * Math.log(Math.max(0.000001, random()))) *
    Math.cos(2 * Math.PI * random())
  );
}

function freshSeed() {
  try {
    return crypto.getRandomValues(new Uint32Array(1))[0];
  } catch {
    return Math.floor(Math.random() * 4294967296);
  }
}

function readBest() {
  try {
    const v = Number(localStorage.getItem(BEST_KEY));
    return Number.isInteger(v) && v >= 0 && v <= ROUND_SIZE ? v : 0;
  } catch {
    return 0;
  }
}

function readBestStreak() {
  try {
    const v = Number(localStorage.getItem(STREAK_KEY));
    return Number.isInteger(v) && v >= 0 && v <= 999 ? v : 0;
  } catch {
    return 0;
  }
}

function readSoundEnabled() {
  try {
    const raw = localStorage.getItem(SOUND_KEY);
    return raw === null ? true : raw !== '0';
  } catch {
    return true;
  }
}

function initialVelocity(target, speed) {
  const range = Math.hypot(target.x, PENALTY_PHYSICS.goalZ);
  const height = target.y - PENALTY_PHYSICS.ballRadius;
  const sq = speed * speed;
  const discriminant =
    sq * sq -
    PENALTY_PHYSICS.gravity *
      (PENALTY_PHYSICS.gravity * range * range + 2 * height * sq);
  const angle =
    discriminant > 0
      ? Math.atan(
          (sq - Math.sqrt(discriminant)) / (PENALTY_PHYSICS.gravity * range)
        )
      : Math.atan2(height + 3, range);
  const horizontal = speed * Math.cos(angle);
  return {
    x: (horizontal * target.x) / range,
    y: speed * Math.sin(angle),
    z: (horizontal * PENALTY_PHYSICS.goalZ) / range,
  };
}

function pointDistanceToSegment(p, a, b) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const den = dx * dx + dy * dy;
  const t = den
    ? clamp(((p.x - a.x) * dx + (p.y - a.y) * dy) / den, 0, 1)
    : 0;
  return Math.hypot(p.x - a.x - dx * t, p.y - a.y - dy * t);
}

function keeperSkeleton(k) {
  const up = { x: Math.sin(k.angle), y: Math.cos(k.angle) };
  const side = { x: Math.cos(k.angle), y: -Math.sin(k.angle) };
  const p = (along, across = 0) => ({
    x: k.x + up.x * along + side.x * across,
    y: k.y + up.y * along + side.y * across,
  });
  const hip = p(0);
  const shoulder = p(0.54);
  const head = p(0.8);
  const reach = k.diving ? 0.7 : 0.48;
  const lift = k.diving ? 0.18 : 0.05;
  const leftElbow = p(0.4 + lift, -0.34);
  const rightElbow = p(0.4 + lift, 0.34);
  const leftHand = p(0.47 + lift, -reach);
  const rightHand = p(0.47 + lift, reach);
  const leftKnee = p(-0.29, -0.16);
  const rightKnee = p(-0.27, 0.18);
  const leftFoot = p(-0.66, -0.25);
  const rightFoot = p(-0.64, 0.27);
  const segments = [
    { a: hip, b: shoulder, radius: 0.19, part: 'body' },
    { a: shoulder, b: leftElbow, radius: 0.085, part: 'arm' },
    { a: leftElbow, b: leftHand, radius: 0.085, part: 'arm' },
    { a: shoulder, b: rightElbow, radius: 0.085, part: 'arm' },
    { a: rightElbow, b: rightHand, radius: 0.085, part: 'arm' },
    { a: hip, b: leftKnee, radius: 0.11, part: 'leg' },
    { a: leftKnee, b: leftFoot, radius: 0.09, part: 'leg' },
    { a: hip, b: rightKnee, radius: 0.11, part: 'leg' },
    { a: rightKnee, b: rightFoot, radius: 0.09, part: 'leg' },
  ];
  return {
    hip,
    shoulder,
    head,
    leftElbow,
    rightElbow,
    leftHand,
    rightHand,
    leftKnee,
    rightKnee,
    leftFoot,
    rightFoot,
    segments,
  };
}

function keeperCollision(previous, ball, keeper) {
  const plane = PENALTY_PHYSICS.goalZ - 0.38;
  const t =
    ball.z === previous.z
      ? 1
      : clamp((plane - previous.z) / (ball.z - previous.z), 0, 1);
  const p = {
    x: lerp(previous.x, ball.x, t),
    y: lerp(previous.y, ball.y, t),
  };
  const depth = Math.abs(lerp(previous.z, ball.z, t) - plane);
  if (depth > 0.34) return null;
  const skeleton = keeperSkeleton(keeper);
  for (const s of skeleton.segments) {
    if (
      Math.hypot(pointDistanceToSegment(p, s.a, s.b), depth) <=
      s.radius + PENALTY_PHYSICS.ballRadius
    ) {
      return { point: { ...p, z: plane }, part: s.part };
    }
  }
  for (const [joint, radius, part] of [
    [skeleton.head, 0.145, 'head'],
    [skeleton.leftHand, 0.125, 'glove'],
    [skeleton.rightHand, 0.125, 'glove'],
  ]) {
    if (
      Math.hypot(p.x - joint.x, p.y - joint.y, depth) <=
      radius + PENALTY_PHYSICS.ballRadius
    ) {
      return { point: { ...p, z: plane }, part };
    }
  }
  return null;
}

function updateKeeper(k, ball, time, spin, c) {
  if (!c.enabled) return;
  if (!k.target && time >= k.reaction) {
    const t = clamp(
      (PENALTY_PHYSICS.goalZ - 0.38 - ball.z) / Math.max(1, ball.vz),
      0,
      1.8
    );
    // A delayed, imperfect extrapolation from the observed ball. No knowledge
    // of the shooter's target or the eventual categorical result is provided.
    k.target = {
      x: ball.x + ball.vx * t + 0.5 * spin * 5.5 * t * t + k.errorX,
      y:
        ball.y +
        ball.vy * t -
        0.5 * PENALTY_PHYSICS.gravity * t * t +
        k.errorY,
    };
    k.diveStart = time + 0.065;
  }
  if (!k.target) return;
  const direction = Math.sign(k.target.x) || 1;
  k.diving =
    time >= k.diveStart &&
    (Math.abs(k.target.x) > 0.62 || k.target.y > 1.55);
  const target = clamp(
    k.target.x - (k.diving ? direction * 0.44 : 0),
    -3.25,
    3.25
  );
  const maxSpeed = k.diving ? c.diveSpeed : c.moveSpeed;
  k.vx = clamp(
    k.vx +
      clamp(
        (target - k.x) * 28 - k.vx * 5,
        -c.acceleration,
        c.acceleration
      ) *
        DT,
    -maxSpeed,
    maxSpeed
  );
  k.x += k.vx * DT;
  if (k.diving && !k.jumped) {
    k.jumped = true;
    k.vy =
      k.target.y > 1.5
        ? c.jumpSpeed
        : k.target.y > 0.9
          ? c.jumpSpeed * 0.65
          : 0.5;
  }
  if (k.jumped) {
    k.vy -= 8 * DT;
    k.y = Math.max(0.53, k.y + k.vy * DT);
  }
  const angle = k.diving
    ? direction * Math.min(1.28, 0.6 + Math.abs(k.target.x) * 0.18)
    : 0;
  k.angle += (angle - k.angle) * Math.min(1, DT * 9);
}

// Sweep the ball against the three frame cylinders. The normal is evaluated at
// the first contact, so a glancing shot can keep travelling into the goal.
function goalFrameCollision(previous, ball) {
  const radius = PENALTY_PHYSICS.ballRadius + PENALTY_PHYSICS.postRadius;
  const motion = {
    x: ball.x - previous.x,
    y: ball.y - previous.y,
    z: ball.z - previous.z,
  };
  const contacts = [];
  const sweep = (axis, center, part) => {
    const offset = previous[axis] - center;
    const depth = previous.z - PENALTY_PHYSICS.goalZ;
    const a = motion[axis] ** 2 + motion.z ** 2;
    if (a < 1e-12) return;
    const b = 2 * (offset * motion[axis] + depth * motion.z);
    const c = offset ** 2 + depth ** 2 - radius ** 2;
    const discriminant = b * b - 4 * a * c;
    if (discriminant < 0) return;
    const t = (-b - Math.sqrt(discriminant)) / (2 * a);
    if (t < -1e-7 || t > 1) return;
    const point = {
      x: lerp(previous.x, ball.x, Math.max(0, t)),
      y: lerp(previous.y, ball.y, Math.max(0, t)),
      z: lerp(previous.z, ball.z, Math.max(0, t)),
    };
    if (
      axis === 'x'
        ? point.y < 0 || point.y > PENALTY_PHYSICS.goalHeight
        : Math.abs(point.x) > PENALTY_PHYSICS.goalWidth / 2
    ) {
      return;
    }
    const normal = {
      x: 0,
      y: 0,
      z: (point.z - PENALTY_PHYSICS.goalZ) / radius,
    };
    normal[axis] = (point[axis] - center) / radius;
    const approach =
      motion.x * normal.x + motion.y * normal.y + motion.z * normal.z;
    if (approach >= 0) return;
    contacts.push({ t: Math.max(0, t), point, normal, part });
  };
  sweep('x', -PENALTY_PHYSICS.goalWidth / 2, 'post');
  sweep('x', PENALTY_PHYSICS.goalWidth / 2, 'post');
  sweep('y', PENALTY_PHYSICS.goalHeight, 'crossbar');
  return contacts.sort((a, b) => a.t - b.t)[0] || null;
}

/** Deterministic 120 Hz 3D simulation, used directly by the browser game. */
export function simulatePenaltyShot(options = {}) {
  const mode = PENALTY_MODES[options.mode] || PENALTY_MODES.placement;
  const power = clamp(Number(options.power) || 0.65, 0.06, 1);
  const inputSpin = clamp(Number(options.spin) || 0, -1, 1);
  const spin = inputSpin * mode.spinFactor;
  const random = randomGenerator(Number(options.seed) || 1827);
  const aim = {
    x: clamp(Number(options.aim?.x) || 0, -5.2, 5.2),
    y: clamp(Number(options.aim?.y) || 1.25, 0.03, 4),
  };
  const dispersion =
    mode.dispersion * (0.4 + power * power * 1.2) * (options.accuracyScale ?? 1);
  const target = {
    x: aim.x + gaussian(random) * dispersion,
    y: aim.y + gaussian(random) * dispersion * 0.7,
  };
  const speed = mode.baseSpeed + power * mode.addedSpeed;
  const v = initialVelocity(target, speed);
  const ball = { x: 0, y: 0.11, z: 0, vx: v.x, vy: v.y, vz: v.z };
  const c = {
    enabled: true,
    reaction: 0.18,
    reactionVariation: 0.065,
    error: 0.22,
    moveSpeed: 3.6,
    diveSpeed: 6.8,
    acceleration: 28,
    jumpSpeed: 2.8,
    ...(options.keeper || {}),
  };
  const k = {
    x: 0,
    y: 0.86,
    vx: 0,
    vy: 0,
    angle: 0,
    target: null,
    diving: false,
    jumped: false,
    reaction: c.reaction + random() * c.reactionVariation,
    errorX: gaussian(random) * c.error,
    errorY: gaussian(random) * c.error * 0.65,
    diveStart: 100,
  };
  const frames = [];
  const frameContacts = [];
  let event = null;
  let crossing = null;
  let bounces = 0;
  let travelled = 0;
  let time = 0;
  const snapshot = () => ({
    time,
    ball: { x: ball.x, y: ball.y, z: ball.z },
    keeper: { x: k.x, y: k.y, angle: k.angle, diving: k.diving },
    rotation: travelled / 0.11,
    event: event?.type || null,
  });
  frames.push(snapshot());
  for (let step = 1; step <= 480; step++) {
    time = step * DT;
    const previous = { x: ball.x, y: ball.y, z: ball.z };
    const ballSpeed = Math.hypot(ball.vx, ball.vy, ball.vz);
    const drag = PENALTY_PHYSICS.drag * (options.dragScale ?? 1) * ballSpeed;
    ball.vx += (spin * 5.5 * Math.abs(ball.vz) / 24 - drag * ball.vx) * DT;
    ball.vy += (-9.81 - drag * ball.vy + Math.abs(spin) * 0.35) * DT;
    ball.vz -= drag * ball.vz * DT;
    ball.x += ball.vx * DT;
    ball.y += ball.vy * DT;
    ball.z += ball.vz * DT;
    travelled += Math.hypot(
      ball.x - previous.x,
      ball.y - previous.y,
      ball.z - previous.z
    );
    if (ball.y < 0.11) {
      ball.y = 0.11;
      ball.vy = Math.abs(ball.vy) > 0.8 ? Math.abs(ball.vy) * 0.37 : 0;
      ball.vx *= 0.93;
      ball.vz *= 0.94;
      bounces++;
    }
    updateKeeper(k, ball, time, spin, c);
    if (!event && c.enabled && ball.vz > 0) {
      const contact = keeperCollision(previous, ball, k);
      if (contact) {
        event = {
          type: 'save',
          time,
          point: contact.point,
          part: contact.part,
        };
        ball.z = contact.point.z - 0.15;
        ball.vz = -Math.abs(ball.vz) * 0.32;
        ball.vx += (ball.x - k.x) * 2.2;
        ball.vy = Math.max(0.8, Math.abs(ball.vy) * 0.45);
      }
    }
    if (!event) {
      const contact = goalFrameCollision(previous, ball);
      if (contact) {
        const { normal, point, t, part } = contact;
        const normalSpeed =
          ball.vx * normal.x + ball.vy * normal.y + ball.vz * normal.z;
        ball.vx = (ball.vx - 1.65 * normalSpeed * normal.x) * 0.99;
        ball.vy = (ball.vy - 1.65 * normalSpeed * normal.y) * 0.99;
        ball.vz = (ball.vz - 1.65 * normalSpeed * normal.z) * 0.99;
        const remaining = DT * (1 - t);
        ball.x = point.x + normal.x * 0.001 + ball.vx * remaining;
        ball.y = Math.max(0.11, point.y + normal.y * 0.001 + ball.vy * remaining);
        ball.z = point.z + normal.z * 0.001 + ball.vz * remaining;
        frameContacts.push({
          time: time - DT + DT * t,
          point: { ...point },
          part,
        });
      }
      const line = PENALTY_PHYSICS.goalZ + PENALTY_PHYSICS.ballRadius;
      if (previous.z < line && ball.z >= line && ball.vz > 0) {
        const t = clamp((line - previous.z) / (ball.z - previous.z), 0, 1);
        crossing = {
          x: lerp(previous.x, ball.x, t),
          y: lerp(previous.y, ball.y, t),
          z: line,
        };
        const inside = Math.abs(crossing.x) < 3.55 && crossing.y < 2.33;
        event = {
          type: inside
            ? 'goal'
            : frameContacts.length
              ? 'post'
              : crossing.y > 2.44
                ? 'high'
                : 'wide',
          time: time - DT + DT * t,
          point: crossing,
        };
      } else if (
        frameContacts.length &&
        ball.vz < 0 &&
        ball.z < PENALTY_PHYSICS.goalZ - 0.45
      ) {
        const last = frameContacts.at(-1);
        event = { type: 'post', time, point: last.point, part: last.part };
      }
    }
    if (event?.type === 'goal' && ball.z >= 12.35) {
      ball.z = 12.35;
      ball.vz = -Math.abs(ball.vz) * 0.12;
      ball.vx *= 0.5;
      ball.vy *= 0.4;
    }
    frames.push(snapshot());
    if (event && time >= event.time + 0.85) break;
    if (!event && (time >= 3 || (bounces > 4 && ballSpeed < 2))) {
      const last = frameContacts.at(-1);
      event = {
        type: last ? 'post' : 'short',
        time,
        point: last?.point || { x: ball.x, y: ball.y, z: ball.z },
        ...(last ? { part: last.part } : {}),
      };
      break;
    }
  }
  event ||= {
    type: 'short',
    time,
    point: { x: ball.x, y: ball.y, z: ball.z },
  };
  return {
    frames,
    event,
    frameContacts,
    outcome: event.type,
    input: {
      aim,
      power,
      mode: options.mode || 'placement',
      spin: inputSpin,
      seed: Number(options.seed) || 1827,
    },
    stats: {
      speedKmh: speed * 3.6,
      flightTime: event.time,
      point: event.point,
      bounces,
      reaction: k.reaction,
      spin,
    },
  };
}

/**
 * Computes a zero-dispersion trajectory preview curve from the penalty spot to the goal plane,
 * showing how gravity, speed, and spin curve the ball relative to the aimed point.
 */
export function previewPenaltyTrajectory(options = {}) {
  const mode = PENALTY_MODES[options.mode] || PENALTY_MODES.placement;
  const power = clamp(Number(options.power) || 0.65, 0.06, 1);
  const inputSpin = clamp(Number(options.spin) || 0, -1, 1);
  const spin = inputSpin * mode.spinFactor;
  const aim = {
    x: clamp(Number(options.aim?.x) || 0, -5.2, 5.2),
    y: clamp(Number(options.aim?.y) || 1.25, 0.03, 4),
  };
  const accuracyScale = options.accuracyScale ?? 1;
  const dispersion = mode.dispersion * (0.4 + power * power * 1.2) * accuracyScale;
  const speed = mode.baseSpeed + power * mode.addedSpeed;
  const v = initialVelocity(aim, speed);
  const ball = { x: 0, y: 0.11, z: 0, vx: v.x, vy: v.y, vz: v.z };
  const points = [{ x: 0, y: 0.11, z: 0 }];
  let landing = { x: aim.x, y: aim.y, z: PENALTY_PHYSICS.goalZ };

  for (let step = 1; step <= 260; step++) {
    const prev = { x: ball.x, y: ball.y, z: ball.z };
    const ballSpeed = Math.hypot(ball.vx, ball.vy, ball.vz);
    const drag = PENALTY_PHYSICS.drag * ballSpeed;
    ball.vx += (spin * 5.5 * Math.abs(ball.vz) / 24 - drag * ball.vx) * DT;
    ball.vy += (-9.81 - drag * ball.vy + Math.abs(spin) * 0.35) * DT;
    ball.vz -= drag * ball.vz * DT;
    ball.x += ball.vx * DT;
    ball.y = Math.max(0.11, ball.y + ball.vy * DT);
    ball.z += ball.vz * DT;

    if (step % 4 === 0) {
      points.push({ x: ball.x, y: ball.y, z: ball.z });
    }
    if (prev.z < PENALTY_PHYSICS.goalZ && ball.z >= PENALTY_PHYSICS.goalZ) {
      const t = clamp(
        (PENALTY_PHYSICS.goalZ - prev.z) / Math.max(0.0001, ball.z - prev.z),
        0,
        1
      );
      landing = {
        x: lerp(prev.x, ball.x, t),
        y: lerp(prev.y, ball.y, t),
        z: PENALTY_PHYSICS.goalZ,
      };
      points.push(landing);
      break;
    }
  }
  return {
    points,
    landing,
    aim,
    curveOffsetX: landing.x - aim.x,
    dispersionRadius: dispersion * 11,
  };
}

function createSoundEngine() {
  let ctx = null;
  let enabled = readSoundEnabled();

  function ensureContext() {
    if (!enabled) return null;
    try {
      if (!ctx) {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (!AudioCtx) return null;
        ctx = new AudioCtx();
      }
      if (ctx.state === 'suspended') {
        ctx.resume().catch(() => {});
      }
      return ctx;
    } catch {
      return null;
    }
  }

  function noiseBuffer(audio, duration = 0.35) {
    const length = Math.max(1, Math.floor(audio.sampleRate * duration));
    const buffer = audio.createBuffer(1, length, audio.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < length; i++) {
      data[i] = Math.random() * 2 - 1;
    }
    return buffer;
  }

  return {
    isEnabled: () => enabled,
    setEnabled(next) {
      enabled = Boolean(next);
      try {
        localStorage.setItem(SOUND_KEY, enabled ? '1' : '0');
      } catch {}
    },
    playKick(power = 0.75, mode = 'placement') {
      const audio = ensureContext();
      if (!audio) return;
      try {
        const now = audio.currentTime;
        const osc = audio.createOscillator();
        const gain = audio.createGain();
        osc.type = 'sine';
        const startFreq = mode === 'power' ? 168 : 138;
        osc.frequency.setValueAtTime(startFreq + power * 36, now);
        osc.frequency.exponentialRampToValueAtTime(34, now + 0.13);
        gain.gain.setValueAtTime(0.28 + power * 0.18, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.15);
        osc.connect(gain);
        gain.connect(audio.destination);
        osc.start(now);
        osc.stop(now + 0.16);

        // Leather snap transient
        const src = audio.createBufferSource();
        src.buffer = noiseBuffer(audio, 0.06);
        const filter = audio.createBiquadFilter();
        filter.type = 'bandpass';
        filter.frequency.value = mode === 'curl' ? 1100 : 820;
        const nGain = audio.createGain();
        nGain.gain.setValueAtTime(0.14 * power, now);
        nGain.gain.exponentialRampToValueAtTime(0.001, now + 0.055);
        src.connect(filter);
        filter.connect(nGain);
        nGain.connect(audio.destination);
        src.start(now);
      } catch {}
    },
    playPostHit() {
      const audio = ensureContext();
      if (!audio) return;
      try {
        const now = audio.currentTime;
        for (const [freq, amp, decay] of [
          [415, 0.22, 0.34],
          [685, 0.16, 0.26],
          [1240, 0.11, 0.18],
        ]) {
          const osc = audio.createOscillator();
          const gain = audio.createGain();
          osc.type = 'triangle';
          osc.frequency.setValueAtTime(freq, now);
          osc.frequency.exponentialRampToValueAtTime(freq * 0.94, now + decay);
          gain.gain.setValueAtTime(amp, now);
          gain.gain.exponentialRampToValueAtTime(0.001, now + decay);
          osc.connect(gain);
          gain.connect(audio.destination);
          osc.start(now);
          osc.stop(now + decay + 0.02);
        }
      } catch {}
    },
    playSaveHit() {
      const audio = ensureContext();
      if (!audio) return;
      try {
        const now = audio.currentTime;
        const src = audio.createBufferSource();
        src.buffer = noiseBuffer(audio, 0.11);
        const filter = audio.createBiquadFilter();
        filter.type = 'bandpass';
        filter.frequency.value = 640;
        filter.Q.value = 1.4;
        const gain = audio.createGain();
        gain.gain.setValueAtTime(0.24, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.1);
        src.connect(filter);
        filter.connect(gain);
        gain.connect(audio.destination);
        src.start(now);
      } catch {}
    },
    playGoal() {
      const audio = ensureContext();
      if (!audio) return;
      try {
        const now = audio.currentTime;
        // Net swish + stadium crowd swell
        const src = audio.createBufferSource();
        src.buffer = noiseBuffer(audio, 1.05);
        const filter = audio.createBiquadFilter();
        filter.type = 'bandpass';
        filter.frequency.setValueAtTime(950, now);
        filter.frequency.exponentialRampToValueAtTime(520, now + 0.95);
        filter.Q.value = 0.7;
        const gain = audio.createGain();
        gain.gain.setValueAtTime(0.01, now);
        gain.gain.linearRampToValueAtTime(0.18, now + 0.12);
        gain.gain.exponentialRampToValueAtTime(0.002, now + 1.02);
        src.connect(filter);
        filter.connect(gain);
        gain.connect(audio.destination);
        src.start(now);

        // Warm celebratory triad
        for (const [freq, delay] of [
          [261.63, 0.04],
          [329.63, 0.09],
          [392.0, 0.14],
          [523.25, 0.2],
        ]) {
          const osc = audio.createOscillator();
          const cGain = audio.createGain();
          osc.type = 'triangle';
          osc.frequency.setValueAtTime(freq, now + delay);
          cGain.gain.setValueAtTime(0.001, now + delay);
          cGain.gain.linearRampToValueAtTime(0.065, now + delay + 0.04);
          cGain.gain.exponentialRampToValueAtTime(0.001, now + delay + 0.55);
          osc.connect(cGain);
          cGain.connect(audio.destination);
          osc.start(now + delay);
          osc.stop(now + delay + 0.58);
        }
      } catch {}
    },
    playMiss() {
      const audio = ensureContext();
      if (!audio) return;
      try {
        const now = audio.currentTime;
        const osc = audio.createOscillator();
        const gain = audio.createGain();
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(195, now);
        osc.frequency.exponentialRampToValueAtTime(118, now + 0.36);
        const filter = audio.createBiquadFilter();
        filter.type = 'lowpass';
        filter.frequency.value = 360;
        gain.gain.setValueAtTime(0.08, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.38);
        osc.connect(filter);
        filter.connect(gain);
        gain.connect(audio.destination);
        osc.start(now);
        osc.stop(now + 0.4);
      } catch {}
    },
    playWhistle() {
      const audio = ensureContext();
      if (!audio) return;
      try {
        const now = audio.currentTime;
        const osc = audio.createOscillator();
        const lfo = audio.createOscillator();
        const lfoGain = audio.createGain();
        const gain = audio.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(2780, now);
        osc.frequency.linearRampToValueAtTime(2920, now + 0.08);
        osc.frequency.linearRampToValueAtTime(2740, now + 0.22);
        lfo.frequency.value = 42;
        lfoGain.gain.value = 110;
        lfo.connect(lfoGain);
        lfoGain.connect(osc.frequency);
        gain.gain.setValueAtTime(0.001, now);
        gain.gain.linearRampToValueAtTime(0.065, now + 0.025);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.24);
        osc.connect(gain);
        gain.connect(audio.destination);
        osc.start(now);
        lfo.start(now);
        osc.stop(now + 0.25);
        lfo.stop(now + 0.25);
      } catch {}
    },
  };
}

const arrow =
  '<svg width="16" height="16" viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M4 16 15 5M5 5h10v10" stroke="currentColor" stroke-width="1.5"/></svg>';
const replayIcon =
  '<svg width="16" height="16" viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M3 7h5V2M3.7 7a7 7 0 1 1-.2 6" stroke="currentColor" stroke-width="1.4"/><path d="m9 7 4 3-4 3Z" fill="currentColor"/></svg>';
const soundOnIcon =
  '<svg width="15" height="15" viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M3 8h3.5L11 4.5v11L6.5 12H3V8Z" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/><path d="M14 7.2a4 4 0 0 1 0 5.6M16.2 5a7.2 7.2 0 0 1 0 10" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>';
const soundOffIcon =
  '<svg width="15" height="15" viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M3 8h3.5L11 4.5v11L6.5 12H3V8Z" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/><path d="m14 7.5 4.5 5M18.5 7.5l-4.5 5" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>';

const outcomes = {
  goal: '进球！',
  save: '被扑出',
  post: '击中门框',
  high: '高出横梁',
  wide: '偏出球门',
  short: '未到门线',
};

const outcomeBadges = {
  goal: 'GOAL · 破门得分',
  save: 'SAVED · 门将神扑',
  post: 'WOODWORK · 门框拒绝',
  high: 'OVER · 高出横梁',
  wide: 'WIDE · 偏出底线',
  short: 'SHORT · 力量不足',
};

function describeZone(aim) {
  const absX = Math.abs(aim.x);
  const side = absX < 0.55 ? '中路' : aim.x < 0 ? '左路' : '右路';
  const heightLabel =
    aim.y > 2.35
      ? '偏高危险区'
      : aim.y > 1.55
        ? '上角'
        : aim.y > 0.75
          ? '半高'
          : '贴地';
  if (absX > 3.55 || aim.y > 2.42) return '门框外侧';
  if (absX > 2.4 && aim.y > 1.6) return `${aim.x < 0 ? '左上' : '右上'}死角`;
  if (absX > 2.4 && aim.y < 0.72) return `${aim.x < 0 ? '左下' : '右下'}死角`;
  return `${side}${heightLabel}`;
}

export function renderPenalty() {
  const best = readBest();
  const soundEnabled = readSoundEnabled();
  return `<section class="penalty-widget" aria-label="点球游戏">
    <header class="penalty-topline">
      <div class="penalty-game-label">
        <span>11 M</span>
        <i></i>
        <span>STRETFORD END · 点球决胜</span>
      </div>
      <div class="penalty-round" role="group" aria-label="本轮五次射门">
        ${Array.from(
          { length: ROUND_SIZE },
          (_, i) =>
            `<span class="penalty-shot" data-penalty-shot="${i}" aria-label="第 ${i + 1} 球，未射门"></span>`
        ).join('')}
        <span class="penalty-round-count" data-penalty-count>0 / 5</span>
      </div>
      <div class="penalty-topline-meta">
        <div class="penalty-streak" data-penalty-streak-wrap title="当前连续进球数">
          <span>连进</span>
          <strong data-penalty-streak>0</strong>
        </div>
        <div class="penalty-best" title="五球单轮最高进球纪录">
          <span>最佳</span>
          <strong data-penalty-best>${best}</strong>
          <span>/ 5</span>
        </div>
        <button type="button" class="penalty-sound-btn" data-penalty-sound aria-pressed="${soundEnabled}" title="切换球场音效">
          <span data-penalty-sound-icon>${soundEnabled ? soundOnIcon : soundOffIcon}</span>
          <span>音效</span>
        </button>
      </div>
    </header>

    <div class="penalty-stage">
      <canvas class="penalty-canvas" role="button" tabindex="0" aria-label="点球球场。鼠标移动或触屏拖动瞄准，按住蓄力后松开射门；方向键瞄准，空格蓄力，回车快速射门。" aria-describedby="penalty-help">浏览器暂不支持画布。</canvas>

      <div class="penalty-scene-label" aria-hidden="true">
        <span>STRETFORD END · THEATRE OF DREAMS</span>
        <span>7.32 × 2.44 M · 120 HZ PHYSICS</span>
      </div>

      <div class="penalty-result" data-penalty-result aria-hidden="true">
        <em class="penalty-result-badge" data-penalty-result-badge>GOAL</em>
        <strong data-penalty-result-title></strong>
        <span data-penalty-result-detail></span>
      </div>

      <div class="penalty-hud-bottom">
        <div class="penalty-power-panel">
          <div class="penalty-aim-readout" data-penalty-aim></div>
          <div class="penalty-power">
            <div class="penalty-power-header">
              <span data-penalty-power-label>蓄力槽</span>
              <em class="penalty-power-tag" data-penalty-power-tag>黄金区 72–86%</em>
              <strong data-penalty-power-value>76<span>%</span></strong>
            </div>
            <div class="penalty-power-track">
              <span class="penalty-power-sweet" title="黄金发力区间 72%–86%"></span>
              <span class="penalty-power-danger" title="过载区 >88%"></span>
              <span class="penalty-power-fill" data-penalty-power-fill style="transform:scaleX(.76)"></span>
              <i class="penalty-power-SweetLine"></i>
            </div>
            <small data-penalty-action-hint>按住蓄力（往返摆动），在黄金区间松开射门</small>
          </div>
        </div>

        <div class="penalty-quick-zones" role="group" aria-label="快速瞄准区域">
          <span class="penalty-quick-label">快速落点</span>
          <div class="penalty-quick-grid">
            ${Object.entries(QUICK_ZONES)
              .map(
                ([id, z]) =>
                  `<button type="button" class="penalty-zone-btn" data-penalty-zone="${id}">${z.label}</button>`
              )
              .join('')}
          </div>
        </div>
      </div>

      <div class="penalty-summary" data-penalty-summary hidden aria-live="polite">
        <div class="penalty-summary-card">
          <div class="penalty-summary-head">
            <span class="penalty-summary-eyebrow" data-penalty-summary-badge>本轮结算 · STRETFORD END</span>
            <h3 class="penalty-summary-title" data-penalty-summary-title>点球大战结束</h3>
            <p class="penalty-summary-sub" data-penalty-summary-sub></p>
          </div>
          <div class="penalty-summary-score">
            <div class="penalty-summary-big">
              <strong data-penalty-summary-goals>0</strong>
              <span>/ 5 进球</span>
            </div>
            <div class="penalty-summary-metrics">
              <div><span>命中率</span><strong data-penalty-summary-acc>0%</strong></div>
              <div><span>平均球速</span><strong data-penalty-summary-speed>—</strong></div>
              <div><span>最高连进</span><strong data-penalty-summary-streak>0</strong></div>
              <div><span>历史最佳</span><strong data-penalty-summary-best>${best} / 5</strong></div>
            </div>
          </div>
          <div class="penalty-summary-shots" data-penalty-summary-shots></div>
          <div class="penalty-summary-actions">
            <button type="button" class="penalty-summary-replay" data-penalty-summary-replay>${replayIcon} 慢速回放第 5 球</button>
            <button type="button" class="penalty-summary-restart" data-penalty-summary-restart>再来一轮 ${arrow}</button>
          </div>
        </div>
      </div>
    </div>

    <div class="penalty-controls">
      <div class="penalty-mode-control">
        <span class="penalty-control-label">射门脚法</span>
        <div class="penalty-modes" role="group" aria-label="射门方式">
          ${Object.entries(PENALTY_MODES)
            .map(
              ([id, m]) =>
                `<button type="button" data-penalty-mode="${id}" aria-pressed="${id === 'placement'}" title="${m.detail}">${m.name}</button>`
            )
            .join('')}
        </div>
        <span class="penalty-mode-detail" data-penalty-mode-detail>${PENALTY_MODES.placement.detail}</span>
      </div>

      <div class="penalty-difficulty-control">
        <span class="penalty-control-label">门将难度</span>
        <div class="penalty-difficulties" role="group" aria-label="守门员难度">
          ${Object.entries(PENALTY_DIFFICULTIES)
            .map(
              ([id, d]) =>
                `<button type="button" data-penalty-difficulty="${id}" aria-pressed="${id === 'standard'}" title="${d.detail}">${d.name}</button>`
            )
            .join('')}
        </div>
        <span class="penalty-mode-detail" data-penalty-difficulty-detail>${PENALTY_DIFFICULTIES.standard.detail}</span>
      </div>

      <div class="penalty-spin-control">
        <label class="penalty-control-label" for="penalty-spin">
          <span>弧线旋转</span>
          <output id="penalty-spin-value" for="penalty-spin">0.0（直球）</output>
        </label>
        <div class="penalty-spin-row">
          <span>左旋</span>
          <input id="penalty-spin" type="range" min="-1" max="1" step="0.1" value="0" aria-label="旋转方向与强度">
          <span>右旋</span>
        </div>
        <span class="penalty-mode-detail">拖动滑块预览三维弧线偏移轨迹</span>
      </div>

      <div class="penalty-actions">
        <button class="penalty-replay" type="button" data-penalty-replay disabled>${replayIcon} 慢速回放</button>
        <button class="penalty-shoot" type="button" data-penalty-shoot>按住蓄力射门 ${arrow}</button>
      </div>
    </div>

    <div class="penalty-statline">
      <div><span>球速</span><strong data-penalty-speed>—</strong><em>km/h</em></div>
      <div><span>飞行</span><strong data-penalty-flight>—</strong><em>s</em></div>
      <div><span>门将反应</span><strong data-penalty-reaction>—</strong><em>s</em></div>
      <div class="penalty-landing-stat">
        <span data-penalty-point-label>门线落点</span>
        <strong data-penalty-landing>等待第一脚射门</strong>
      </div>
      <div class="penalty-statline-actions">
        <button type="button" class="penalty-next" data-penalty-next hidden>下一球 →</button>
        <button type="button" class="penalty-reset" data-penalty-reset>重新开始</button>
      </div>
    </div>

    <p id="penalty-help" class="penalty-help">
      <span>桌面端：移动鼠标瞄准，按住画布或右侧按钮蓄力射门 · 触屏端：在球门画布拖动瞄准，按住「蓄力射门」按钮发射</span>
      <span>方向键微调落点（Shift 精细瞄准）· 空格键按住蓄力 · Enter 黄金力度快射</span>
    </p>
    <p class="penalty-sr-only" data-penalty-status role="status" aria-live="polite">五球一轮。瞄准球门，按住蓄力后松开射门。</p>
  </section>`;
}

export function initPenalty() {
  const root = document.querySelector('.penalty-widget');
  if (!root) return () => {};

  const $ = (selector) => root.querySelector(selector);
  const canvas = $('.penalty-canvas');
  const ctx = canvas?.getContext('2d');
  const shootButton = $('[data-penalty-shoot]');
  const replayButton = $('[data-penalty-replay]');
  const nextButton = $('[data-penalty-next]');
  const resetButton = $('[data-penalty-reset]');
  const soundButton = $('[data-penalty-sound]');
  const status = $('[data-penalty-status]');
  const result = $('[data-penalty-result]');
  const summaryEl = $('[data-penalty-summary]');
  const summaryRestartBtn = $('[data-penalty-summary-restart]');
  const summaryReplayBtn = $('[data-penalty-summary-replay]');
  const dots = [...root.querySelectorAll('[data-penalty-shot]')];
  const modeButtons = [...root.querySelectorAll('[data-penalty-mode]')];
  const diffButtons = [...root.querySelectorAll('[data-penalty-difficulty]')];
  const zoneButtons = [...root.querySelectorAll('[data-penalty-zone]')];
  const spinInput = $('#penalty-spin');
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const sound = createSoundEngine();
  const removers = [];

  let activePointerId = null;
  let touchAimPointerId = null;
  let disposed = false;
  let animation = 0;
  let width = 1100;
  let height = 620;
  let camera = null;

  let state = 'ready'; // 'ready' | 'charging' | 'flight' | 'replay' | 'result'
  let mode = 'placement';
  let difficulty = 'standard';
  let spin = 0;
  let aim = { x: 1.85, y: 1.52 };
  let power = 0.76;
  let best = readBest();
  let streak = 0;
  let bestStreak = readBestStreak();
  let goals = 0;
  let attempts = [];
  let shotHistory = [];
  let lastShot = null;
  let playbackStart = 0;
  let chargeStart = 0;
  let chargeSource = null;
  let resultStartTime = 0;
  let autoReadyTimer = 0;
  let summaryTimer = 0;
  let displayFrame = idleFrame();
  let pointerShotTime = -1000;
  let aimVisible = true;

  // Visual effects state
  let particles = [];
  let shockwaves = [];
  let cameraShake = 0;
  let goalFlash = 0;
  let processedContactCount = 0;
  let processedMainEvent = false;
  let lastBounceZ = -1;

  const listen = (target, event, handler, options) => {
    if (!target) return;
    target.addEventListener(event, handler, options);
    removers.push(() => target.removeEventListener(event, handler, options));
  };

  function idleFrame() {
    return {
      time: 0,
      ball: { x: 0, y: 0.11, z: 0 },
      keeper: { x: 0, y: 0.86, angle: 0, diving: false },
      rotation: 0,
      event: null,
    };
  }

  function clearTimers() {
    if (autoReadyTimer) {
      clearTimeout(autoReadyTimer);
      autoReadyTimer = 0;
    }
    if (summaryTimer) {
      clearTimeout(summaryTimer);
      summaryTimer = 0;
    }
  }

  function project(p) {
    const dy = p.y - camera.y;
    const dz = p.z - camera.z;
    const depth = dz * camera.cos - dy * camera.sin;
    const v = dy * camera.cos + dz * camera.sin;
    const scale = camera.f / Math.max(0.2, depth);
    return {
      x: width * 0.5 + scale * p.x,
      y: height * 0.28 - scale * v,
      scale,
      depth,
    };
  }

  function aimFromPointer(event) {
    if (!camera) return;
    const b = canvas.getBoundingClientRect();
    const sx = (event.clientX - b.left - width * 0.5) / camera.f;
    const sy = (height * 0.28 - (event.clientY - b.top)) / camera.f;
    const dy = sy * camera.cos - camera.sin;
    const dz = sy * camera.sin + camera.cos;
    const t = (PENALTY_PHYSICS.goalZ - camera.z) / Math.max(0.01, dz);
    aim = {
      x: clamp(sx * t, -5.2, 5.2),
      y: clamp(camera.y + dy * t, 0.03, 4),
    };
    aimVisible = true;
    zoneButtons.forEach((btn) => btn.classList.remove('is-active'));
    updateAim();
  }

  function path(points, stroke, lineWidth = 1, close = false, fill = null) {
    ctx.beginPath();
    points.forEach((p, i) => {
      const q = project(p);
      if (!i) ctx.moveTo(q.x, q.y);
      else ctx.lineTo(q.x, q.y);
    });
    if (close) ctx.closePath();
    if (fill) {
      ctx.fillStyle = fill;
      ctx.fill();
    }
    if (stroke) {
      ctx.strokeStyle = stroke;
      ctx.lineWidth = lineWidth;
      ctx.stroke();
    }
  }

  function computeOscillatingPower(elapsedMs) {
    // Oscillates smoothly between 0.08 and 1.0 every 1150ms
    const cycle = (elapsedMs % 2300) / 1150;
    const raw = cycle <= 1 ? cycle : 2 - cycle;
    // Slight ease near top so the sweet spot (72-86%) and peak feel responsive
    return clamp(0.08 + raw * 0.92, 0.08, 1);
  }

  function getPowerQuality(p) {
    if (p >= SWEET_MIN && p <= SWEET_MAX) {
      return {
        id: 'sweet',
        label: '黄金发力 · 落点精准',
        accuracyScale: 0.7,
      };
    }
    if (p > OVERCHARGE_MIN) {
      return {
        id: 'over',
        label: '过载危险 · 易偏离',
        accuracyScale: 1.38,
      };
    }
    if (p < 0.3) {
      return {
        id: 'low',
        label: '力度偏轻 · 易被扑',
        accuracyScale: 0.9,
      };
    }
    return {
      id: 'normal',
      label: '黄金区 72–86%',
      accuracyScale: 1,
    };
  }

  function spawnKickParticles() {
    for (let i = 0; i < 16; i++) {
      const angle = (Math.random() - 0.5) * 1.8;
      const spd = 1.2 + Math.random() * 2.8;
      particles.push({
        x: (Math.random() - 0.5) * 0.16,
        y: 0.06 + Math.random() * 0.06,
        z: (Math.random() - 0.3) * 0.16,
        vx: Math.sin(angle) * spd * 0.65,
        vy: 0.8 + Math.random() * 1.9,
        vz: Math.cos(angle) * spd,
        size: 0.022 + Math.random() * 0.025,
        color: i % 3 === 0 ? '#e6ead8' : '#567d59',
        alpha: 0.85,
        life: 0,
        maxLife: 0.38 + Math.random() * 0.25,
        gravity: 6.5,
      });
    }
  }

  function spawnBounceParticles(pos) {
    for (let i = 0; i < 8; i++) {
      particles.push({
        x: pos.x + (Math.random() - 0.5) * 0.12,
        y: 0.06,
        z: pos.z + (Math.random() - 0.5) * 0.12,
        vx: (Math.random() - 0.5) * 1.4,
        vy: 0.5 + Math.random() * 1.1,
        vz: (Math.random() - 0.5) * 1.2,
        size: 0.02 + Math.random() * 0.02,
        color: i % 2 ? '#d7decb' : '#4a6b4d',
        alpha: 0.7,
        life: 0,
        maxLife: 0.3 + Math.random() * 0.18,
        gravity: 5.5,
      });
    }
  }

  function spawnPostImpact(point) {
    cameraShake = Math.max(cameraShake, 0.45);
    shockwaves.push({
      x: point.x,
      y: point.y,
      z: point.z,
      radius: 0.08,
      maxRadius: 0.85,
      color: '#ffd166',
      life: 0,
      maxLife: 0.42,
    });
    for (let i = 0; i < 26; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = (Math.random() - 0.35) * Math.PI;
      const spd = 2 + Math.random() * 4.8;
      particles.push({
        x: point.x,
        y: point.y,
        z: point.z,
        vx: Math.cos(theta) * Math.cos(phi) * spd,
        vy: Math.sin(phi) * spd + 1.1,
        vz: -Math.abs(Math.sin(theta) * spd * 0.7),
        size: 0.025 + Math.random() * 0.03,
        color: i % 2 === 0 ? '#ffe082' : '#ff7b54',
        alpha: 1,
        life: 0,
        maxLife: 0.35 + Math.random() * 0.28,
        gravity: 7.5,
      });
    }
  }

  function spawnSaveImpact(point) {
    cameraShake = Math.max(cameraShake, 0.22);
    shockwaves.push({
      x: point.x,
      y: point.y,
      z: point.z,
      radius: 0.1,
      maxRadius: 0.72,
      color: '#e8f1f5',
      life: 0,
      maxLife: 0.36,
    });
    for (let i = 0; i < 18; i++) {
      const a = Math.random() * Math.PI * 2;
      const spd = 1.4 + Math.random() * 3.2;
      particles.push({
        x: point.x,
        y: point.y,
        z: point.z,
        vx: Math.cos(a) * spd,
        vy: Math.sin(a) * spd + 0.6,
        vz: -0.8 - Math.random() * 2,
        size: 0.024 + Math.random() * 0.024,
        color: i % 2 ? '#ffffff' : '#a8dadc',
        alpha: 0.9,
        life: 0,
        maxLife: 0.32 + Math.random() * 0.2,
        gravity: 5,
      });
    }
  }

  function spawnGoalCelebration(point) {
    goalFlash = 1;
    cameraShake = Math.max(cameraShake, 0.18);
    shockwaves.push({
      x: point.x,
      y: point.y,
      z: point.z,
      radius: 0.14,
      maxRadius: 1.45,
      color: '#f4d35e',
      life: 0,
      maxLife: 0.65,
    });
    const palette = ['#ef5b4d', '#f4d35e', '#f8f9fa', '#4ade80'];
    for (let i = 0; i < 52; i++) {
      const spreadX = point.x + (Math.random() - 0.5) * 4.8;
      const spreadY = clamp(point.y + (Math.random() - 0.3) * 1.4, 0.3, 2.4);
      particles.push({
        x: spreadX,
        y: spreadY,
        z: PENALTY_PHYSICS.goalZ + Math.random() * 0.8,
        vx: (Math.random() - 0.5) * 3.8,
        vy: 1.6 + Math.random() * 3.8,
        vz: (Math.random() - 0.5) * 2.5,
        size: 0.035 + Math.random() * 0.04,
        color: palette[i % palette.length],
        alpha: 0.95,
        life: 0,
        maxLife: 0.7 + Math.random() * 0.55,
        gravity: 4.2,
      });
    }
  }

  function stepEffects(dt) {
    cameraShake = Math.max(0, cameraShake - dt * 1.8);
    goalFlash = Math.max(0, goalFlash - dt * 1.5);

    for (let i = particles.length - 1; i >= 0; i--) {
      const p = particles[i];
      p.life += dt;
      if (p.life >= p.maxLife) {
        particles.splice(i, 1);
        continue;
      }
      p.vy -= (p.gravity ?? 6) * dt;
      p.x += p.vx * dt;
      p.y = Math.max(0.02, p.y + p.vy * dt);
      p.z += p.vz * dt;
    }

    for (let i = shockwaves.length - 1; i >= 0; i--) {
      const sw = shockwaves[i];
      sw.life += dt;
      if (sw.life >= sw.maxLife) {
        shockwaves.splice(i, 1);
        continue;
      }
      const progress = sw.life / sw.maxLife;
      sw.radius = lerp(0.08, sw.maxRadius, 1 - (1 - progress) ** 2);
    }
  }

  function triggerFlightEvents(frameTime, frameBall) {
    if (!lastShot) return;

    // Ground bounce dust
    if (
      frameTime > 0.06 &&
      frameBall.y <= 0.125 &&
      Math.abs(frameBall.z - lastBounceZ) > 0.8
    ) {
      lastBounceZ = frameBall.z;
      spawnBounceParticles(frameBall);
    }

    // Post / Crossbar contacts
    while (
      processedContactCount < lastShot.frameContacts.length &&
      frameTime >= lastShot.frameContacts[processedContactCount].time
    ) {
      const contact = lastShot.frameContacts[processedContactCount];
      processedContactCount++;
      spawnPostImpact(contact.point);
      if (state === 'flight') sound.playPostHit();
    }

    // Main outcome event (save / goal / miss)
    if (!processedMainEvent && lastShot.event && frameTime >= lastShot.event.time) {
      processedMainEvent = true;
      const ev = lastShot.event;
      if (ev.type === 'save') {
        spawnSaveImpact(ev.point);
        if (state === 'flight') {
          sound.playSaveHit();
          sound.playMiss();
        }
      } else if (ev.type === 'goal') {
        spawnGoalCelebration(ev.point);
        if (state === 'flight') sound.playGoal();
      } else if (state === 'flight') {
        if (ev.type !== 'post') sound.playMiss();
      }
    }
  }

  function drawStadium(now) {
    const t = now * 0.001;
    const isGoal =
      (state === 'result' || state === 'flight' || state === 'replay') &&
      processedMainEvent &&
      lastShot?.outcome === 'goal';

    // Night sky gradient
    const sky = ctx.createLinearGradient(0, 0, 0, height);
    sky.addColorStop(0, '#070d15');
    sky.addColorStop(0.45, '#18242c');
    sky.addColorStop(1, '#10211b');
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, width, height);

    // Atmospheric Stretford End crimson glow
    const glowIntensity = isGoal ? 0.34 + Math.sin(t * 10) * 0.06 : 0.16;
    const haze = ctx.createRadialGradient(
      width * 0.5,
      height * 0.16,
      0,
      width * 0.5,
      height * 0.16,
      width * 0.68
    );
    haze.addColorStop(0, `rgba(208, 53, 43, ${glowIntensity})`);
    haze.addColorStop(0.5, 'rgba(163, 69, 57, 0.07)');
    haze.addColorStop(1, 'rgba(118, 47, 50, 0)');
    ctx.fillStyle = haze;
    ctx.fillRect(0, 0, width, height * 0.72);

    // Roof silhouette & gantry trusses
    ctx.fillStyle = '#060b10';
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(width, 0);
    ctx.lineTo(width, height * 0.115);
    ctx.lineTo(width * 0.5, height * 0.048);
    ctx.lineTo(0, height * 0.115);
    ctx.fill();

    ctx.strokeStyle = 'rgba(200, 212, 205, 0.08)';
    ctx.lineWidth = 1;
    for (let i = 1; i < 12; i++) {
      const rx = (width * i) / 12;
      ctx.beginPath();
      ctx.moveTo(rx, 0);
      ctx.lineTo(rx, height * 0.09);
      ctx.stroke();
    }

    // Stadium tiers & animated crowd
    for (let tier = 0; tier < 4; tier++) {
      const y = height * (0.12 + tier * 0.054);
      ctx.strokeStyle = tier === 3 ? '#d3c9b836' : '#a3a9a31c';
      ctx.lineWidth = tier === 3 ? 2 : 1;
      ctx.beginPath();
      ctx.moveTo(0, y + height * 0.04);
      ctx.quadraticCurveTo(width * 0.5, y - height * 0.025, width, y + height * 0.04);
      ctx.stroke();

      for (let row = 0; row < 3; row++) {
        for (let seat = 0; seat < 112; seat++) {
          const x = (width * seat) / 111;
          const curve = (x / width - 0.5) ** 2 * height * 0.11;
          const seedVal = tier * 17.3 + row * 8.5 + seat * 31.1;
          const n = Math.abs(Math.sin(seedVal));
          const wave = isGoal
            ? Math.sin(t * 14 + seat * 0.35 + row) * height * 0.0045
            : Math.sin(t * 2.2 + seedVal) * height * 0.0008;
          ctx.fillStyle =
            n > 0.82
              ? '#e4d6c478'
              : n > 0.46
                ? isGoal
                  ? '#ef5b4d90'
                  : '#b6434352'
                : '#8f999c28';
          ctx.fillRect(
            x,
            y + row * height * 0.013 + curve + wave,
            Math.max(1.2, width * 0.002),
            Math.max(1.4, height * 0.003)
          );
        }
      }
    }

    // Camera flashes in the stands on goal or shot flight
    if (isGoal || state === 'flight') {
      for (let f = 0; f < 8; f++) {
        const flashPhase = Math.sin(t * 19 + f * 7.7);
        if (flashPhase > 0.92) {
          const fx = width * (0.08 + ((f * 0.117 + Math.sin(f * 3)) % 0.84));
          const fy = height * (0.13 + (f % 4) * 0.045);
          const fg = ctx.createRadialGradient(fx, fy, 0, fx, fy, width * 0.022);
          fg.addColorStop(0, 'rgba(255, 253, 235, 0.85)');
          fg.addColorStop(1, 'rgba(255, 253, 235, 0)');
          ctx.fillStyle = fg;
          ctx.beginPath();
          ctx.arc(fx, fy, width * 0.022, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    // Pitch ground gradient
    const horizonY = project({ x: 0, y: 0, z: 32 }).y;
    const ground = ctx.createLinearGradient(0, horizonY, 0, height);
    ground.addColorStop(0, '#2f5042');
    ground.addColorStop(0.45, '#233d2e');
    ground.addColorStop(1, '#15271f');
    ctx.fillStyle = ground;
    ctx.fillRect(0, horizonY, width, height - horizonY);

    // Mowed grass stripes (z-bands + x-checkerboard sheen)
    for (let z = -4; z <= 34; z += 2) {
      path(
        [
          { x: -34, y: 0, z },
          { x: 34, y: 0, z },
          { x: 34, y: 0, z: z + 2 },
          { x: -34, y: 0, z: z + 2 },
        ],
        null,
        0,
        true,
        (z / 2) % 2 ? 'rgba(195, 218, 165, 0.032)' : 'rgba(0, 0, 0, 0.025)'
      );
    }
    for (let x = -18; x < 18; x += 3) {
      path(
        [
          { x, y: 0.001, z: -4 },
          { x: x + 1.5, y: 0.001, z: -4 },
          { x: x + 1.5, y: 0.001, z: 16 },
          { x, y: 0.001, z: 16 },
        ],
        null,
        0,
        true,
        'rgba(255, 255, 255, 0.01)'
      );
    }

    // Perimeter LED advertising boards behind goal (z = 14.4)
    const adZ = 14.4;
    const adH = 0.72;
    path(
      [
        { x: -16, y: 0, z: adZ },
        { x: -16, y: adH, z: adZ },
        { x: 16, y: adH, z: adZ },
        { x: 16, y: 0, z: adZ },
      ],
      '#d8dfd026',
      1,
      true,
      isGoal ? '#381619' : '#10181d'
    );
    // Accent top ribbon on LED board
    path(
      [
        { x: -16, y: adH, z: adZ },
        { x: 16, y: adH, z: adZ },
      ],
      isGoal ? '#f4d35e99' : '#d0352b88',
      2
    );

    // Pitch spotlight cone
    const pitchLight = ctx.createRadialGradient(
      width * 0.5,
      height * 0.62,
      0,
      width * 0.5,
      height * 0.62,
      width * 0.65
    );
    pitchLight.addColorStop(0, 'rgba(198, 212, 166, 0.1)');
    pitchLight.addColorStop(1, 'rgba(198, 212, 166, 0)');
    ctx.fillStyle = pitchLight;
    ctx.fillRect(0, 0, width, height);

    // Pitch chalk markings: Goal line, 6-yard box, 18-yard box, and penalty arc
    const lineWidth = Math.max(1, width / 880);
    const chalk = '#eef0e45e';
    path(
      [
        { x: -28, y: 0.006, z: 11 },
        { x: 28, y: 0.006, z: 11 },
      ],
      chalk,
      lineWidth * 1.15
    );
    // 6-yard box
    path(
      [
        { x: -9.16, y: 0.006, z: 11 },
        { x: -9.16, y: 0.006, z: 5.5 },
        { x: 9.16, y: 0.006, z: 5.5 },
        { x: 9.16, y: 0.006, z: 11 },
      ],
      chalk,
      lineWidth
    );
    // 18-yard box
    path(
      [
        { x: -20.16, y: 0.006, z: 11 },
        { x: -20.16, y: 0.006, z: -5.5 },
        { x: 20.16, y: 0.006, z: -5.5 },
        { x: 20.16, y: 0.006, z: 11 },
      ],
      '#e8ead93d',
      lineWidth
    );

    // Penalty spot (z = 0)
    const spot = project({ x: 0, y: 0.004, z: 0 });
    ctx.fillStyle = '#f2f3e8a8';
    ctx.beginPath();
    ctx.ellipse(
      spot.x,
      spot.y,
      spot.scale * 0.1,
      spot.scale * 0.038,
      0,
      0,
      Math.PI * 2
    );
    ctx.fill();

    // Floodlight banks
    for (const side of [-1, 1]) {
      const x = width * (side < 0 ? 0.08 : 0.92);
      const y = height * 0.095;
      const shimmer = 0.13 + Math.sin(t * 3.5 + side) * 0.015;
      const lamp = ctx.createRadialGradient(x, y, 0, x, y, width * 0.3);
      lamp.addColorStop(0, `rgba(240, 247, 232, ${shimmer})`);
      lamp.addColorStop(0.18, 'rgba(240, 247, 232, 0.045)');
      lamp.addColorStop(1, 'rgba(240, 247, 232, 0)');
      ctx.fillStyle = lamp;
      ctx.fillRect(0, 0, width, height);

      ctx.fillStyle = '#edf2e1c4';
      ctx.shadowColor = '#f4ffe3';
      ctx.shadowBlur = width * 0.016;
      for (let i = 0; i < 6; i++) {
        ctx.fillRect(
          x + i * width * 0.006 - width * 0.016,
          y,
          width * 0.0032,
          height * 0.0062
        );
      }
      ctx.shadowBlur = 0;
    }

    // Goal flash overlay
    if (goalFlash > 0.01) {
      ctx.fillStyle = `rgba(244, 211, 94, ${goalFlash * 0.14})`;
      ctx.fillRect(0, 0, width, height);
    }

    // Vignette
    const shade = ctx.createRadialGradient(
      width * 0.5,
      height * 0.5,
      width * 0.18,
      width * 0.5,
      height * 0.5,
      width * 0.82
    );
    shade.addColorStop(0, 'rgba(3, 11, 8, 0)');
    shade.addColorStop(1, 'rgba(3, 11, 8, 0.55)');
    ctx.fillStyle = shade;
    ctx.fillRect(0, 0, width, height);
  }

  function netPoint(x, y, z, time) {
    if (!lastShot || lastShot.outcome !== 'goal' || time < lastShot.event.time) {
      return { x, y, z };
    }
    const age = time - lastShot.event.time - 0.035;
    if (age <= 0) return { x, y, z };
    const d = Math.hypot(
      x - lastShot.event.point.x,
      y - lastShot.event.point.y
    );
    const wave =
      0.32 *
      Math.sin(Math.min(1, age / 0.11) * Math.PI * 0.82) *
      Math.exp(-age * 3.2 - d * 1.45);
    return {
      x: x + (x - lastShot.event.point.x) * wave * 0.08,
      y: y + (y - lastShot.event.point.y) * wave * 0.06,
      z: z + wave,
    };
  }

  function drawNet(time) {
    const w = 3.66;
    const h = 2.44;
    const z = 11;
    const back = 12.45;

    // Back stanchions (external support posts)
    for (const side of [-w, w]) {
      path(
        [
          { x: side, y: h, z },
          { x: side * 1.02, y: h * 0.92, z: back },
          { x: side * 1.02, y: 0, z: back },
        ],
        '#6f7d7380',
        1.8
      );
    }

    path(
      [
        { x: -w, y: 0, z: back },
        { x: -w, y: h, z: back },
        { x: w, y: h, z: back },
        { x: w, y: 0, z: back },
      ],
      '#e9eddf2e',
      0.9,
      false,
      '#07100b1a'
    );
    for (let x = -w; x <= w + 0.001; x += w / 16) {
      path(
        [netPoint(x, 0, back, time), netPoint(x, h, back, time)],
        '#dde4d640',
        0.68
      );
      path([{ x, y: h, z }, netPoint(x, h, back, time)], '#dde4d630', 0.62);
    }
    for (let y = 0; y <= h + 0.001; y += h / 11) {
      const points = [];
      for (let x = -w; x <= w + 0.001; x += w / 16) {
        points.push(netPoint(x, y, back, time));
      }
      path(points, '#e2e9da42', 0.68);
      for (const side of [-w, w]) {
        path(
          [{ x: side, y, z }, netPoint(side, y, back, time)],
          '#dde4d630',
          0.62
        );
      }
    }
    for (let d = z; d <= back + 0.001; d += 1.45 / 6) {
      for (const side of [-w, w]) {
        path(
          [
            { x: side, y: 0, z: d },
            { x: side, y: h, z: d },
          ],
          '#dde4d630',
          0.62
        );
      }
    }
  }

  function drawFrame() {
    const w = 3.66;
    const h = 2.44;
    const z = 11;
    const line = Math.max(2.6, project({ x: 0, y: 1, z }).scale * 0.06 * 1.85);
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.shadowColor = '#000a';
    ctx.shadowBlur = 12;
    ctx.shadowOffsetY = 5;
    path(
      [
        { x: -w, y: 0, z },
        { x: -w, y: h, z },
        { x: w, y: h, z },
        { x: w, y: 0, z },
      ],
      '#9da79c',
      line + 1.6
    );
    ctx.shadowBlur = 0;
    ctx.shadowOffsetY = 0;
    path(
      [
        { x: -w, y: 0, z },
        { x: -w, y: h, z },
        { x: w, y: h, z },
        { x: w, y: 0, z },
      ],
      '#eff3e8',
      line * 0.6
    );
  }

  function computeVisualKeeperPose(frame, now) {
    const k = frame.keeper;
    const t = now * 0.001;
    const isIdle = state === 'ready' || state === 'charging';
    const isResult = state === 'result';

    // Pre-shot rhythmic ready bounce & line sway
    const bounce = isIdle ? Math.abs(Math.sin(t * 5.2)) * 0.045 : 0;
    const sway = isIdle ? Math.sin(t * 2.1) * 0.14 : 0;
    const chargeCrouch = state === 'charging' ? 0.055 : 0;

    // Post-shot reaction
    let postLift = 0;
    let armCelebrate = 0;
    if (isResult && lastShot) {
      if (lastShot.outcome === 'save') {
        armCelebrate = 0.22 + Math.sin(t * 9) * 0.08;
        postLift = 0.04;
      } else if (lastShot.outcome === 'goal') {
        postLift = -0.07;
      }
    }

    const kx = k.x + sway;
    const ky = Math.max(0.48, k.y + bounce - chargeCrouch + postLift);
    const angle = k.angle + (isIdle ? Math.sin(t * 2.1) * 0.035 : 0);

    const up = { x: Math.sin(angle), y: Math.cos(angle) };
    const side = { x: Math.cos(angle), y: -Math.sin(angle) };
    const p = (along, across = 0) => ({
      x: kx + up.x * along + side.x * across,
      y: ky + up.y * along + side.y * across,
    });

    const diveFactor = clamp(
      (k.diving ? 0.85 : 0) + Math.abs(angle) * 0.45,
      0,
      1
    );
    const reach = lerp(0.46, 0.74, diveFactor) + armCelebrate * 0.4;
    const lift = lerp(0.06, 0.22, diveFactor) + armCelebrate;
    const kneeSpread = isIdle ? 0.19 + chargeCrouch * 0.5 : 0.16;

    const hip = p(0);
    const waist = p(0.22);
    const shoulder = p(0.55);
    const neck = p(0.65);
    const head = p(0.81);

    const leftShoulder = p(0.52, -0.2);
    const rightShoulder = p(0.52, 0.2);
    const leftElbow = p(0.38 + lift * 0.7, -0.35);
    const rightElbow = p(0.38 + lift * 0.7, 0.35);
    const leftHand = p(0.48 + lift, -reach);
    const rightHand = p(0.48 + lift, reach);

    const leftHip = p(-0.02, -0.11);
    const rightHip = p(-0.02, 0.11);
    const leftKnee = p(-0.29 + bounce * 0.4, -kneeSpread);
    const rightKnee = p(-0.27 + bounce * 0.4, kneeSpread);
    const leftFoot = p(-0.66, -0.25);
    const rightFoot = p(-0.64, 0.26);

    return {
      kx,
      ky,
      angle,
      hip,
      waist,
      shoulder,
      neck,
      head,
      leftShoulder,
      rightShoulder,
      leftElbow,
      rightElbow,
      leftHand,
      rightHand,
      leftHip,
      rightHip,
      leftKnee,
      rightKnee,
      leftFoot,
      rightFoot,
    };
  }

  function drawKeeper(frame, now) {
    const s = computeVisualKeeperPose(frame, now);
    const z = 10.62;

    // Ground shadow
    const shadow = project({ x: s.kx, y: 0, z });
    ctx.fillStyle = 'rgba(2, 8, 5, 0.42)';
    ctx.beginPath();
    ctx.ellipse(
      shadow.x,
      shadow.y,
      shadow.scale * 0.5,
      shadow.scale * 0.095,
      0,
      0,
      Math.PI * 2
    );
    ctx.fill();

    const seg = (a, b, r, color) => {
      const p = project({ ...a, z });
      const q = project({ ...b, z });
      ctx.lineWidth = Math.max(1.8, p.scale * r * 2);
      ctx.lineCap = 'round';
      ctx.strokeStyle = color;
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(q.x, q.y);
      ctx.stroke();
    };

    // Socks & calves
    seg(s.leftKnee, s.leftFoot, 0.072, '#d8dfd0');
    seg(s.rightKnee, s.rightFoot, 0.072, '#d8dfd0');
    // Sock hoop stripes
    const lMid = {
      x: lerp(s.leftKnee.x, s.leftFoot.x, 0.35),
      y: lerp(s.leftKnee.y, s.leftFoot.y, 0.35),
    };
    const rMid = {
      x: lerp(s.rightKnee.x, s.rightFoot.x, 0.35),
      y: lerp(s.rightKnee.y, s.rightFoot.y, 0.35),
    };
    seg(s.leftKnee, lMid, 0.075, '#1d5c42');
    seg(s.rightKnee, rMid, 0.075, '#1d5c42');

    // Boots
    for (const foot of [s.leftFoot, s.rightFoot]) {
      const fp = project({ ...foot, z });
      ctx.fillStyle = '#14181b';
      ctx.beginPath();
      ctx.ellipse(
        fp.x,
        fp.y,
        Math.max(2.5, fp.scale * 0.095),
        Math.max(1.8, fp.scale * 0.055),
        -s.angle * 0.5,
        0,
        Math.PI * 2
      );
      ctx.fill();
    }

    // Shorts & thighs
    seg(s.leftHip, s.leftKnee, 0.105, '#1b2421');
    seg(s.rightHip, s.rightKnee, 0.105, '#1b2421');

    // Athletic tapered torso (Jersey)
    const ls = project({ ...s.leftShoulder, z });
    const rs = project({ ...s.rightShoulder, z });
    const lh = project({ ...s.leftHip, z });
    const rh = project({ ...s.rightHip, z });
    ctx.fillStyle = '#1f6e4d';
    ctx.beginPath();
    ctx.moveTo(ls.x, ls.y);
    ctx.lineTo(rs.x, rs.y);
    ctx.lineTo(rh.x, rh.y);
    ctx.lineTo(lh.x, lh.y);
    ctx.closePath();
    ctx.fill();
    ctx.strokeStyle = '#e0b04a88';
    ctx.lineWidth = Math.max(1, ls.scale * 0.02);
    ctx.stroke();

    // Kit number "1" on chest
    const chest = project({
      x: lerp(s.shoulder.x, s.hip.x, 0.42),
      y: lerp(s.shoulder.y, s.hip.y, 0.42),
      z,
    });
    ctx.save();
    ctx.translate(chest.x, chest.y);
    ctx.rotate(-s.angle);
    ctx.fillStyle = '#f2efe2c8';
    ctx.font = `700 ${Math.max(8, Math.round(chest.scale * 0.2))}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('1', 0, 0);
    ctx.restore();

    // Arms & jersey sleeves
    seg(s.leftShoulder, s.leftElbow, 0.078, '#1a5c40');
    seg(s.rightShoulder, s.rightElbow, 0.078, '#1a5c40');
    seg(s.leftElbow, s.leftHand, 0.068, '#287a57');
    seg(s.rightElbow, s.rightHand, 0.068, '#287a57');

    // Padded goalkeeper gloves with red wrist strap
    for (const [elbow, hand] of [
      [s.leftElbow, s.leftHand],
      [s.rightElbow, s.rightHand],
    ]) {
      const p = project({ ...hand, z });
      const e = project({ ...elbow, z });
      const wristX = lerp(e.x, p.x, 0.82);
      const wristY = lerp(e.y, p.y, 0.82);
      ctx.fillStyle = '#d0352b';
      ctx.beginPath();
      ctx.arc(wristX, wristY, Math.max(2, p.scale * 0.065), 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = '#f3f5ea';
      ctx.strokeStyle = '#26332c88';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(p.x, p.y, Math.max(2.6, p.scale * 0.112), 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }

    // Neck & Head
    seg(s.shoulder, s.neck, 0.065, '#c8a98a');
    const head = project({ ...s.head, z });
    ctx.save();
    ctx.translate(head.x, head.y);
    ctx.rotate(-s.angle);
    // Face oval
    ctx.fillStyle = '#d2b494';
    ctx.beginPath();
    ctx.ellipse(0, 0, head.scale * 0.115, head.scale * 0.142, 0, 0, Math.PI * 2);
    ctx.fill();
    // Hair cap
    ctx.fillStyle = '#251e1a';
    ctx.beginPath();
    ctx.ellipse(
      0,
      -head.scale * 0.04,
      head.scale * 0.118,
      head.scale * 0.1,
      0,
      Math.PI,
      Math.PI * 2
    );
    ctx.fill();
    ctx.restore();
  }

  function drawEffects() {
    // 3D Shockwave rings
    for (const sw of shockwaves) {
      const p = project(sw);
      const progress = sw.life / sw.maxLife;
      ctx.save();
      ctx.strokeStyle = sw.color;
      ctx.globalAlpha = Math.max(0, (1 - progress) * 0.85);
      ctx.lineWidth = Math.max(1.5, (1 - progress) * 4);
      ctx.beginPath();
      ctx.arc(p.x, p.y, Math.max(2, p.scale * sw.radius), 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }

    // 3D Particles (sparks, turf, confetti)
    for (const pt of particles) {
      const p = project(pt);
      const progress = pt.life / pt.maxLife;
      ctx.save();
      ctx.fillStyle = pt.color;
      ctx.globalAlpha = Math.max(0, (1 - progress) * pt.alpha);
      const r = Math.max(1.4, p.scale * pt.size);
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  }

  function drawBall(frame) {
    const ball = frame.ball;
    const p = project(ball);
    const shadow = project({ x: ball.x, y: 0.005, z: ball.z });
    const r = Math.max(2.2, p.scale * 0.11);

    // Ball ground shadow
    ctx.fillStyle = `rgba(0, 8, 3, ${clamp(0.46 - ball.y * 0.07, 0.12, 0.42)})`;
    ctx.beginPath();
    ctx.ellipse(
      shadow.x + r * 0.25,
      shadow.y,
      r * (1.28 + ball.y * 0.18),
      r * 0.38,
      0,
      0,
      Math.PI * 2
    );
    ctx.fill();

    // Ribbon trail during flight / replay
    if (lastShot && (state === 'flight' || state === 'replay')) {
      const index = Math.round(frame.time / DT);
      const trailColor =
        lastShot.input.mode === 'power'
          ? [239, 91, 77]
          : lastShot.input.mode === 'curl'
            ? [114, 214, 236]
            : [232, 224, 175];
      for (let i = 9; i >= 1; i--) {
        const previous = lastShot.frames[Math.max(0, index - i * 2)];
        if (!previous || previous.time === frame.time) continue;
        const pp = project(previous.ball);
        const alpha = 0.16 * (1 - i / 10);
        ctx.fillStyle = `rgba(${trailColor[0]}, ${trailColor[1]}, ${trailColor[2]}, ${alpha})`;
        ctx.beginPath();
        ctx.arc(
          pp.x,
          pp.y,
          Math.max(1.2, pp.scale * 0.1 * (1 - i * 0.06)),
          0,
          Math.PI * 2
        );
        ctx.fill();
      }
    }

    // 3D Football sphere shading
    const sphere = ctx.createRadialGradient(
      p.x - r * 0.35,
      p.y - r * 0.4,
      r * 0.1,
      p.x,
      p.y,
      r
    );
    sphere.addColorStop(0, '#ffffff');
    sphere.addColorStop(0.62, '#e2e6da');
    sphere.addColorStop(1, '#6c776b');
    ctx.fillStyle = sphere;
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fill();

    // Rotating pentagon panels
    const rotation = frame.rotation * 0.35;
    ctx.save();
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.clip();
    for (let panel = 0; panel < 6; panel++) {
      const longitude = (panel * Math.PI) / 3 + rotation;
      const px = p.x + Math.sin(longitude) * r * 0.78;
      const py = p.y + Math.cos(longitude + rotation * 0.25) * r * 0.35;
      ctx.fillStyle = panel % 2 ? '#1e3329' : '#b8322b';
      ctx.beginPath();
      for (let corner = 0; corner < 5; corner++) {
        const a = corner * Math.PI * 0.4 + rotation;
        const x = px + Math.cos(a) * r * 0.28;
        const y = py + Math.sin(a) * r * 0.28;
        if (!corner) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.fill();
    }
    ctx.restore();
  }

  function drawAim(now) {
    if (!aimVisible || !['ready', 'charging', 'result'].includes(state)) return;
    if (attempts.length >= ROUND_SIZE) return;

    const quality = getPowerQuality(power);
    const preview = previewPenaltyTrajectory({
      aim,
      power,
      mode,
      spin,
      accuracyScale: quality.accuracyScale,
    });

    const aimProj = project({ ...aim, z: PENALTY_PHYSICS.goalZ });
    const landProj = project(preview.landing);
    const inside =
      Math.abs(preview.landing.x) < 3.52 && preview.landing.y < 2.33;

    // 1. 3D Curved Flight Trajectory Arc from Spot to Landing
    ctx.save();
    const curveColor =
      mode === 'curl' && Math.abs(spin) > 0.05
        ? 'rgba(126, 218, 242, 0.55)'
        : quality.id === 'sweet'
          ? 'rgba(232, 206, 114, 0.52)'
          : quality.id === 'over'
            ? 'rgba(239, 91, 77, 0.52)'
            : 'rgba(235, 232, 210, 0.36)';
    ctx.strokeStyle = curveColor;
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 6]);
    ctx.lineDashOffset = -(now * 0.025);
    ctx.beginPath();
    preview.points.forEach((pt, idx) => {
      const q = project(pt);
      if (idx === 0) ctx.moveTo(q.x, q.y);
      else ctx.lineTo(q.x, q.y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();

    // 2. If spin causes lateral bend, show ghost line between aim crosshair and curved landing
    if (Math.abs(preview.curveOffsetX) > 0.08) {
      ctx.save();
      ctx.strokeStyle = 'rgba(126, 218, 242, 0.42)';
      ctx.lineWidth = 1.2;
      ctx.setLineDash([2, 4]);
      ctx.beginPath();
      ctx.moveTo(aimProj.x, aimProj.y);
      ctx.lineTo(landProj.x, landProj.y);
      ctx.stroke();
      ctx.setLineDash([]);

      // Small predicted landing ring
      const dispPx = clamp(
        landProj.scale * preview.dispersionRadius * 0.085,
        6,
        36
      );
      ctx.strokeStyle = inside
        ? 'rgba(126, 218, 242, 0.78)'
        : 'rgba(239, 91, 77, 0.78)';
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.arc(landProj.x, landProj.y, dispPx, 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillStyle = inside ? '#7edaf2' : '#ef5b4d';
      ctx.beginPath();
      ctx.arc(landProj.x, landProj.y, 3, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    // 3. Main Aiming Reticle with Dispersion Halo
    const r = clamp(width * 0.01, 8, 13);
    const pulse = 1 + Math.sin(now * 0.006) * 0.06;
    const strokeCol =
      quality.id === 'sweet' && state === 'charging'
        ? '#f4d35e'
        : inside
          ? '#f3edd2e0'
          : '#ef7674e0';

    ctx.save();
    ctx.strokeStyle = strokeCol;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(aimProj.x, aimProj.y, r * pulse, 0, Math.PI * 2);
    ctx.stroke();

    // Crosshair ticks
    ctx.beginPath();
    ctx.moveTo(aimProj.x - r - 6, aimProj.y);
    ctx.lineTo(aimProj.x - r + 2, aimProj.y);
    ctx.moveTo(aimProj.x + r - 2, aimProj.y);
    ctx.lineTo(aimProj.x + r + 6, aimProj.y);
    ctx.moveTo(aimProj.x, aimProj.y - r - 6);
    ctx.lineTo(aimProj.x, aimProj.y - r + 2);
    ctx.moveTo(aimProj.x, aimProj.y + r - 2);
    ctx.lineTo(aimProj.x, aimProj.y + r + 6);
    ctx.stroke();

    // Center dot
    ctx.fillStyle = strokeCol;
    ctx.beginPath();
    ctx.arc(aimProj.x, aimProj.y, 2.2, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  function draw(now = performance.now()) {
    if (!ctx || disposed || !camera || width < 1 || height < 1) return;
    const ratio = Math.min(devicePixelRatio || 1, 2);
    if (
      canvas.width !== Math.round(width * ratio) ||
      canvas.height !== Math.round(height * ratio)
    ) {
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
    }
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);

    // Apply subtle camera shake if active
    if (cameraShake > 0.005 && !reducedMotion.matches) {
      const sx = (Math.random() - 0.5) * cameraShake * 16;
      const sy = (Math.random() - 0.5) * cameraShake * 12;
      ctx.translate(sx, sy);
    }

    ctx.clearRect(-20, -20, width + 40, height + 40);
    drawStadium(now);
    drawNet(displayFrame.time);
    drawKeeper(displayFrame, now);
    drawEffects();
    drawBall(displayFrame);
    drawFrame();
    drawAim(now);
  }

  function resize() {
    width = canvas.clientWidth;
    height = canvas.clientHeight;
    camera = {
      y: 3.3,
      z: -8,
      f: Math.min(width * 1.6, height * 3),
      sin: Math.sin(0.13),
      cos: Math.cos(0.13),
    };
    draw();
  }

  function updateAim() {
    const zoneName = describeZone(aim);
    const xText =
      Math.abs(aim.x) < 0.05
        ? '中路'
        : `${aim.x < 0 ? '左' : '右'} ${Math.abs(aim.x).toFixed(1)}m`;
    const preview = previewPenaltyTrajectory({ aim, power, mode, spin });
    const curveNote =
      Math.abs(preview.curveOffsetX) > 0.1
        ? ` · 弧线落点 ${preview.landing.x < 0 ? '左' : '右'} ${Math.abs(preview.landing.x).toFixed(1)}m`
        : '';
    $('[data-penalty-aim]').textContent =
      `瞄准：${zoneName}（${xText} / 高 ${aim.y.toFixed(1)}m）${curveNote}`;
  }

  function updatePower() {
    const pct = Math.round(power * 100);
    const quality = getPowerQuality(power);
    $('[data-penalty-power-value]').innerHTML = `${pct}<span>%</span>`;
    const fill = $('[data-penalty-power-fill]');
    fill.style.transform = `scaleX(${power})`;
    fill.dataset.zone = quality.id;
    fill.dataset.high = String(power > OVERCHARGE_MIN);

    const tag = $('[data-penalty-power-tag]');
    if (tag) {
      tag.textContent = quality.label;
      tag.dataset.zone = quality.id;
    }

    $('[data-penalty-power-label]').textContent =
      state === 'charging' ? '蓄力摆动中' : '蓄力槽';
    $('[data-penalty-action-hint]').textContent =
      state === 'charging'
        ? quality.id === 'sweet'
          ? '最佳时机！立即松开射门'
          : '松开射门（蓄力条往返摆动）'
        : state === 'replay'
          ? '0.35 × 慢速回放中'
          : '按住蓄力（或轻点用 76% 黄金力度快射）';
  }

  function updateControls() {
    const locked = ['flight', 'replay', 'charging'].includes(state);
    const over = attempts.length >= ROUND_SIZE;
    root.dataset.state = state;
    $('[data-penalty-count]').textContent = `${attempts.length} / ${ROUND_SIZE}`;
    $('[data-penalty-streak]').textContent = String(streak);
    const streakWrap = $('[data-penalty-streak-wrap]');
    if (streakWrap) streakWrap.dataset.hot = String(streak >= 3);

    dots.forEach((dot, i) => {
      dot.dataset.result = attempts[i] || '';
      dot.setAttribute(
        'aria-label',
        `第 ${i + 1} 球，${outcomes[attempts[i]] || '未射门'}`
      );
    });

    modeButtons.forEach((button) => {
      button.disabled = locked;
      button.setAttribute(
        'aria-pressed',
        String(button.dataset.penaltyMode === mode)
      );
    });

    diffButtons.forEach((button) => {
      button.disabled = locked;
      button.setAttribute(
        'aria-pressed',
        String(button.dataset.penaltyDifficulty === difficulty)
      );
    });

    zoneButtons.forEach((button) => {
      button.disabled = locked || over;
    });

    spinInput.disabled = locked;
    replayButton.disabled = !lastShot || locked;
    resetButton.disabled = locked;
    if (nextButton) {
      nextButton.hidden = !(state === 'result' && !over);
    }

    shootButton.disabled = state === 'flight' || state === 'replay';
    shootButton.innerHTML = over
      ? `再来一轮 ${arrow}`
      : state === 'charging'
        ? `松开射门 ${arrow}`
        : state === 'flight'
          ? '射门飞行中…'
          : state === 'replay'
            ? '慢动作回放中…'
            : `按住蓄力射门 ${arrow}`;

    canvas.setAttribute(
      'aria-disabled',
      String(state === 'flight' || state === 'replay')
    );
    updatePower();
  }

  function showStats(shot) {
    $('[data-penalty-speed]').textContent = Math.round(shot.stats.speedKmh);
    $('[data-penalty-flight]').textContent = shot.stats.flightTime.toFixed(2);
    const reactionEl = $('[data-penalty-reaction]');
    if (reactionEl) {
      reactionEl.textContent = shot.stats.reaction.toFixed(2);
    }
    $('[data-penalty-point-label]').textContent =
      shot.outcome === 'save'
        ? '扑救接触点'
        : shot.outcome === 'post'
          ? '门框接触点'
          : '门线落点';
    const p = shot.stats.point;
    $('[data-penalty-landing]').textContent =
      `${Math.abs(p.x) < 0.05 ? '中路' : `${p.x < 0 ? '左' : '右'} ${Math.abs(p.x).toFixed(2)} m`} / 高 ${p.y.toFixed(2)} m`;
  }

  function showResult(shot, replay = false) {
    const badge = $('[data-penalty-result-badge]');
    if (badge) {
      badge.textContent = replay
        ? 'SLOW MOTION · 0.35×'
        : outcomeBadges[shot.outcome] || 'RESULT';
    }
    $('[data-penalty-result-title]').textContent = replay
      ? '慢速回放'
      : outcomes[shot.outcome];
    $('[data-penalty-result-detail]').textContent = replay
      ? `${PENALTY_MODES[shot.input.mode].name} · ${Math.round(shot.stats.speedKmh)} km/h`
      : `${Math.round(shot.stats.speedKmh)} km/h · 飞行 ${shot.stats.flightTime.toFixed(2)}s${streak >= 2 && shot.outcome === 'goal' ? ` · ${streak} 连进！` : ''}`;
    result.classList.add('is-visible');
    result.dataset.outcome = shot.outcome;
  }

  function openRoundSummary(isNewBest = false) {
    if (!summaryEl) return;
    const acc = Math.round((goals / ROUND_SIZE) * 100);
    const avgSpeed = shotHistory.length
      ? Math.round(
          shotHistory.reduce((sum, s) => sum + s.stats.speedKmh, 0) /
            shotHistory.length
        )
      : 0;
    const saves = attempts.filter((a) => a === 'save').length;
    const posts = attempts.filter((a) => a === 'post').length;
    const misses = attempts.filter((a) =>
      ['high', 'wide', 'short'].includes(a)
    ).length;

    const ratingTitle =
      goals === 5
        ? '传奇射手 · 完美五连破门！'
        : goals === 4
          ? '冷静终结 · 梦剧场神射手'
          : goals === 3
            ? '稳定发挥 · 合格点球手'
            : goals >= 1
              ? '仍需磨炼 · 再接再厉'
              : '门神封锁 · 重整旗鼓';

    $('[data-penalty-summary-badge]').textContent = isNewBest
      ? '★ 新纪录达成 · STRETFORD END'
      : `本轮结算 · ${PENALTY_DIFFICULTIES[difficulty].name}难度`;
    $('[data-penalty-summary-title]').textContent = ratingTitle;
    $('[data-penalty-summary-sub]').textContent =
      `进球 ${goals} · 被扑 ${saves} · 门框 ${posts} · 偏出 ${misses}`;
    $('[data-penalty-summary-goals]').textContent = String(goals);
    $('[data-penalty-summary-acc]').textContent = `${acc}%`;
    $('[data-penalty-summary-speed]').textContent = `${avgSpeed} km/h`;
    $('[data-penalty-summary-streak]').textContent = `${bestStreak} 球`;
    $('[data-penalty-summary-best]').textContent = `${best} / 5`;

    const shotsContainer = $('[data-penalty-summary-shots]');
    if (shotsContainer) {
      shotsContainer.innerHTML = shotHistory
        .map(
          (s, idx) =>
            `<div class="penalty-summary-shot" data-outcome="${s.outcome}">
              <span>第 ${idx + 1} 球</span>
              <strong>${outcomes[s.outcome]}</strong>
              <em>${PENALTY_MODES[s.input.mode].name} · ${Math.round(s.stats.speedKmh)} km/h</em>
            </div>`
        )
        .join('');
    }
    summaryEl.hidden = false;
  }

  function returnToSpot() {
    if (disposed || attempts.length >= ROUND_SIZE) return;
    if (state === 'result') {
      state = 'ready';
      displayFrame = idleFrame();
      aimVisible = true;
      updateControls();
    }
  }

  function finishPlayback(isReplay) {
    if (disposed || !lastShot) return;
    displayFrame = lastShot.frames.at(-1);
    triggerFlightEvents(displayFrame.time + 0.01, displayFrame.ball);

    let isNewBest = false;
    if (!isReplay) {
      attempts.push(lastShot.outcome);
      shotHistory.push(lastShot);
      if (lastShot.outcome === 'goal') {
        goals++;
        streak++;
        if (streak > bestStreak) {
          bestStreak = streak;
          try {
            localStorage.setItem(STREAK_KEY, String(bestStreak));
          } catch {}
        }
      } else {
        streak = 0;
      }
      if (goals > best) {
        best = goals;
        isNewBest = true;
        $('[data-penalty-best]').textContent = best;
        try {
          localStorage.setItem(BEST_KEY, String(best));
        } catch {}
      }
      showStats(lastShot);
      status.textContent = `${outcomes[lastShot.outcome]}。球速 ${Math.round(lastShot.stats.speedKmh)} 公里每小时，飞行 ${lastShot.stats.flightTime.toFixed(2)} 秒。${attempts.length === ROUND_SIZE ? `本轮五球进 ${goals} 球。` : `已完成 ${attempts.length} 次射门。`}`;
    }

    state = 'result';
    resultStartTime = performance.now();
    showResult(lastShot, false);
    updateControls();
    draw();

    clearTimers();
    if (attempts.length >= ROUND_SIZE) {
      aimVisible = false;
      const delay = reducedMotion.matches ? 60 : 850;
      summaryTimer = window.setTimeout(() => {
        if (!disposed && attempts.length >= ROUND_SIZE && state === 'result') {
          openRoundSummary(isNewBest);
        }
      }, delay);
    } else {
      // Automatically return the ball to the penalty spot after a short beat so the user can aim the next shot seamlessly
      const delay = reducedMotion.matches ? 250 : 1350;
      autoReadyTimer = window.setTimeout(() => {
        if (!disposed && state === 'result' && attempts.length < ROUND_SIZE) {
          returnToSpot();
        }
      }, delay);
    }
  }

  let lastTickTime = performance.now();
  function animate(now) {
    if (disposed) return;
    const dt = clamp((now - lastTickTime) / 1000, 0.001, 0.05);
    lastTickTime = now;

    stepEffects(dt);

    if (state === 'charging') {
      power = computeOscillatingPower(now - chargeStart);
      updatePower();
      draw(now);
    } else if (state === 'flight' || state === 'replay') {
      const isReplay = state === 'replay';
      const time = ((now - playbackStart) / 1000) * (isReplay ? 0.35 : 1);
      const i = clamp(
        Math.floor(time / DT),
        0,
        lastShot.frames.length - 1
      );
      displayFrame = lastShot.frames[i];
      triggerFlightEvents(displayFrame.time, displayFrame.ball);
      draw(now);
      if (time >= lastShot.frames.at(-1).time) {
        finishPlayback(isReplay);
      }
    } else {
      // Ready or result idle loop (keeper bounce, floodlight shimmer, particles decay)
      draw(now);
    }

    if (!reducedMotion.matches && !document.hidden) {
      animation = requestAnimationFrame(animate);
    } else {
      animation = 0;
    }
  }

  function ensureLoop() {
    if (
      !disposed &&
      !animation &&
      !reducedMotion.matches &&
      !document.hidden
    ) {
      lastTickTime = performance.now();
      animation = requestAnimationFrame(animate);
    }
  }

  function releaseShot(releasePower = power) {
    if (
      disposed ||
      attempts.length >= ROUND_SIZE ||
      ['flight', 'replay'].includes(state)
    ) {
      return;
    }
    clearTimers();
    power = clamp(releasePower, 0.06, 1);
    chargeSource = null;
    activePointerId = null;

    const quality = getPowerQuality(power);
    const keeperPreset =
      PENALTY_DIFFICULTIES[difficulty] || PENALTY_DIFFICULTIES.standard;

    lastShot = simulatePenaltyShot({
      aim,
      power,
      mode,
      spin,
      accuracyScale: quality.accuracyScale,
      keeper: keeperPreset,
      seed: freshSeed(),
    });

    processedContactCount = 0;
    processedMainEvent = false;
    lastBounceZ = -1;
    particles = [];
    shockwaves = [];
    spawnKickParticles();
    sound.playKick(power, mode);

    displayFrame = lastShot.frames[0];
    state = 'flight';
    aimVisible = false;
    result.classList.remove('is-visible');
    if (summaryEl) summaryEl.hidden = true;
    status.textContent = `${PENALTY_MODES[mode].name}，力度 ${Math.round(power * 100)}%。射门。`;
    updateControls();

    if (reducedMotion.matches || document.hidden) {
      finishPlayback(false);
      return;
    }
    playbackStart = performance.now();
    ensureLoop();
  }

  function startCharge(source) {
    if (disposed || ['flight', 'replay', 'charging'].includes(state)) return;
    if (attempts.length >= ROUND_SIZE) {
      reset();
      return;
    }
    clearTimers();
    state = 'charging';
    chargeSource = source;
    chargeStart = performance.now();
    power = 0.08;
    displayFrame = idleFrame();
    aimVisible = true;
    result.classList.remove('is-visible');
    if (summaryEl) summaryEl.hidden = true;
    updateControls();
    draw();
    ensureLoop();
  }

  function cancelCharge() {
    if (state !== 'charging') return;
    state = 'ready';
    chargeSource = null;
    activePointerId = null;
    power = 0.76;
    updateControls();
    draw();
  }

  function reset() {
    clearTimers();
    attempts = [];
    shotHistory = [];
    goals = 0;
    lastShot = null;
    state = 'ready';
    chargeSource = null;
    activePointerId = null;
    touchAimPointerId = null;
    power = 0.76;
    aim = { x: 1.85, y: 1.52 };
    particles = [];
    shockwaves = [];
    cameraShake = 0;
    goalFlash = 0;
    displayFrame = idleFrame();
    aimVisible = true;
    result.classList.remove('is-visible');
    if (summaryEl) summaryEl.hidden = true;
    $('[data-penalty-speed]').textContent = '—';
    $('[data-penalty-flight]').textContent = '—';
    const reactionEl = $('[data-penalty-reaction]');
    if (reactionEl) reactionEl.textContent = '—';
    $('[data-penalty-landing]').textContent = '等待第一脚射门';
    status.textContent = '新一轮。五次射门。';
    sound.playWhistle();
    updateAim();
    updateControls();
    draw();
    ensureLoop();
  }

  function replay() {
    if (!lastShot || ['flight', 'replay', 'charging'].includes(state)) return;
    clearTimers();
    if (summaryEl) summaryEl.hidden = true;
    if (reducedMotion.matches) {
      showResult(lastShot);
      status.textContent = '已显示上一球结果。';
      return;
    }
    processedContactCount = 0;
    processedMainEvent = false;
    lastBounceZ = -1;
    particles = [];
    shockwaves = [];
    spawnKickParticles();
    state = 'replay';
    displayFrame = lastShot.frames[0];
    aimVisible = false;
    showResult(lastShot, true);
    updateControls();
    playbackStart = performance.now();
    ensureLoop();
  }

  if (!ctx) {
    shootButton.disabled = true;
    status.textContent = '浏览器暂不支持画布。';
    return () => {};
  }

  // Mode buttons
  for (const button of modeButtons) {
    listen(button, 'click', () => {
      if (['flight', 'replay', 'charging'].includes(state)) return;
      mode = button.dataset.penaltyMode;
      if (mode === 'curl' && Math.abs(spin) < 0.1) {
        spin = 0.6;
        spinInput.value = '0.6';
        $('#penalty-spin-value').textContent = '+0.6（右旋）';
      }
      $('[data-penalty-mode-detail]').textContent = PENALTY_MODES[mode].detail;
      returnToSpot();
      updateAim();
      updateControls();
      draw();
    });
  }

  // Difficulty buttons
  for (const button of diffButtons) {
    listen(button, 'click', () => {
      if (['flight', 'replay', 'charging'].includes(state)) return;
      difficulty = button.dataset.penaltyDifficulty;
      const detailEl = $('[data-penalty-difficulty-detail]');
      if (detailEl) {
        detailEl.textContent = PENALTY_DIFFICULTIES[difficulty].detail;
      }
      returnToSpot();
      updateControls();
      draw();
    });
  }

  // Quick target zone buttons
  for (const button of zoneButtons) {
    listen(button, 'click', () => {
      if (
        ['flight', 'replay', 'charging'].includes(state) ||
        attempts.length >= ROUND_SIZE
      ) {
        return;
      }
      const zone = QUICK_ZONES[button.dataset.penaltyZone];
      if (!zone) return;
      zoneButtons.forEach((b) =>
        b.classList.toggle('is-active', b === button)
      );
      returnToSpot();
      // Compensate for spin curve so the ball curls into the selected corner
      const spinFactor = (PENALTY_MODES[mode] || PENALTY_MODES.placement).spinFactor;
      const estCurveOffset = spin * spinFactor * 1.15;
      aim = {
        x: clamp(zone.x - estCurveOffset, -5.2, 5.2),
        y: zone.y,
      };
      aimVisible = true;
      result.classList.remove('is-visible');
      updateAim();
      draw();
    });
  }

  // Spin slider
  listen(spinInput, 'input', () => {
    spin = Number(spinInput.value);
    const dirLabel =
      Math.abs(spin) < 0.05 ? '直球' : spin > 0 ? '右旋' : '左旋';
    $('#penalty-spin-value').textContent =
      `${spin > 0 ? '+' : ''}${spin.toFixed(1)}（${dirLabel}）`;
    returnToSpot();
    updateAim();
    draw();
  });

  // Sound toggle button
  if (soundButton) {
    listen(soundButton, 'click', () => {
      const next = !sound.isEnabled();
      sound.setEnabled(next);
      soundButton.setAttribute('aria-pressed', String(next));
      const iconEl = $('[data-penalty-sound-icon]');
      if (iconEl) iconEl.innerHTML = next ? soundOnIcon : soundOffIcon;
      if (next) sound.playWhistle();
    });
  }

  // Canvas pointer handling:
  // - Touch on canvas: tap or drag to aim smoothly without accidentally firing; use the Shoot button to charge & shoot.
  // - Mouse on canvas: move to aim; press & hold (or click) to charge & shoot.
  listen(canvas, 'pointermove', (event) => {
    if (event.isPrimary === false) return;
    if (
      ['flight', 'replay'].includes(state) ||
      attempts.length >= ROUND_SIZE
    ) {
      return;
    }
    if (event.pointerType === 'touch') {
      if (touchAimPointerId !== event.pointerId && state !== 'charging') return;
      returnToSpot();
      aimFromPointer(event);
      draw();
      return;
    }
    if (activePointerId !== null && event.pointerId !== activePointerId) return;
    if (state === 'result') {
      returnToSpot();
    }
    aimFromPointer(event);
    draw();
  });

  listen(canvas, 'pointerdown', (event) => {
    if (event.isPrimary === false || event.button !== 0) return;
    if (['flight', 'replay'].includes(state)) return;
    if (attempts.length >= ROUND_SIZE) {
      reset();
      return;
    }
    event.preventDefault();
    canvas.focus({ preventScroll: true });
    canvas.setPointerCapture?.(event.pointerId);

    if (event.pointerType === 'touch') {
      touchAimPointerId = event.pointerId;
      returnToSpot();
      aimFromPointer(event);
      result.classList.remove('is-visible');
      $('[data-penalty-action-hint]').textContent =
        '已锁定瞄准位置 · 按住下方「蓄力射门」按钮发射';
      draw();
      return;
    }

    returnToSpot();
    aimFromPointer(event);
    startCharge('canvas');
    if (state === 'charging' && chargeSource === 'canvas') {
      activePointerId = event.pointerId;
    }
  });

  listen(canvas, 'pointerup', (event) => {
    if (event.pointerType === 'touch' && event.pointerId === touchAimPointerId) {
      aimFromPointer(event);
      touchAimPointerId = null;
      draw();
      return;
    }
    if (
      event.isPrimary === false ||
      event.pointerId !== activePointerId ||
      state !== 'charging' ||
      chargeSource !== 'canvas'
    ) {
      return;
    }
    aimFromPointer(event);
    const elapsed = performance.now() - chargeStart;
    // Quick click (<135ms) uses optimal 0.76 power; hold uses oscillating gauge
    power = elapsed < 135 ? 0.76 : computeOscillatingPower(elapsed);
    pointerShotTime = performance.now();
    releaseShot(power);
  });

  listen(canvas, 'pointercancel', (event) => {
    if (event.pointerId === touchAimPointerId) touchAimPointerId = null;
    if (event.pointerId === activePointerId) cancelCharge();
  });

  // Shoot button pointer handling (works identically on mouse and touch)
  listen(shootButton, 'pointerdown', (event) => {
    if (
      event.isPrimary === false ||
      event.button !== 0 ||
      shootButton.disabled
    ) {
      return;
    }
    if (attempts.length >= ROUND_SIZE) {
      event.preventDefault();
      pointerShotTime = performance.now();
      reset();
      return;
    }
    event.preventDefault();
    shootButton.setPointerCapture?.(event.pointerId);
    returnToSpot();
    startCharge('button');
    if (state === 'charging' && chargeSource === 'button') {
      activePointerId = event.pointerId;
    }
  });

  listen(shootButton, 'pointerup', (event) => {
    if (
      event.isPrimary === false ||
      event.pointerId !== activePointerId ||
      state !== 'charging' ||
      chargeSource !== 'button'
    ) {
      return;
    }
    const elapsed = performance.now() - chargeStart;
    power = elapsed < 135 ? 0.76 : computeOscillatingPower(elapsed);
    pointerShotTime = performance.now();
    releaseShot(power);
  });

  listen(shootButton, 'pointercancel', (event) => {
    if (event.pointerId === activePointerId) cancelCharge();
  });

  listen(shootButton, 'click', (event) => {
    if (performance.now() - pointerShotTime < 400) return;
    if (attempts.length >= ROUND_SIZE) {
      reset();
      return;
    }
    if (event.detail === 0 && state !== 'charging') {
      releaseShot(0.76);
    }
  });

  // Keyboard controls
  const onKeyDown = (event) => {
    if (
      ![
        'ArrowLeft',
        'ArrowRight',
        'ArrowUp',
        'ArrowDown',
        'Enter',
        ' ',
        'Escape',
      ].includes(event.key)
    ) {
      return;
    }
    event.preventDefault();
    if (event.key === 'Escape') {
      cancelCharge();
      return;
    }
    if (attempts.length >= ROUND_SIZE) {
      if ((event.key === 'Enter' || event.key === ' ') && !event.repeat) {
        reset();
      }
      return;
    }
    if (['flight', 'replay'].includes(state)) return;
    if (event.key === ' ') {
      if (!event.repeat) {
        returnToSpot();
        startCharge('keyboard');
      }
      return;
    }
    if (event.key === 'Enter') {
      if (!event.repeat) {
        returnToSpot();
        releaseShot(0.76);
      }
      return;
    }

    returnToSpot();
    const step = event.shiftKey ? 0.07 : 0.25;
    if (event.key === 'ArrowLeft') aim.x = clamp(aim.x - step, -5.2, 5.2);
    if (event.key === 'ArrowRight') aim.x = clamp(aim.x + step, -5.2, 5.2);
    if (event.key === 'ArrowUp') aim.y = clamp(aim.y + step, 0.03, 4);
    if (event.key === 'ArrowDown') aim.y = clamp(aim.y - step, 0.03, 4);
    aimVisible = true;
    updateAim();
    if (state !== 'charging') {
      displayFrame = idleFrame();
      result.classList.remove('is-visible');
      draw();
    }
    status.textContent = $('[data-penalty-aim]').textContent;
  };

  listen(canvas, 'keydown', onKeyDown);
  listen(shootButton, 'keydown', onKeyDown);

  for (const target of [canvas, shootButton]) {
    listen(target, 'keyup', (event) => {
      if (
        event.key === ' ' &&
        state === 'charging' &&
        chargeSource === 'keyboard'
      ) {
        event.preventDefault();
        const elapsed = performance.now() - chargeStart;
        power = elapsed < 135 ? 0.76 : computeOscillatingPower(elapsed);
        releaseShot(power);
      }
    });
  }

  listen(canvas, 'click', (event) => {
    if (
      event.detail === 0 &&
      performance.now() - pointerShotTime > 400 &&
      state !== 'charging'
    ) {
      releaseShot(0.76);
    }
  });

  listen(canvas, 'blur', () => {
    if (chargeSource === 'keyboard') cancelCharge();
  });
  listen(shootButton, 'blur', () => {
    if (chargeSource === 'keyboard') cancelCharge();
  });

  listen(replayButton, 'click', replay);
  listen(resetButton, 'click', reset);
  if (nextButton) {
    listen(nextButton, 'click', () => {
      clearTimers();
      returnToSpot();
      result.classList.remove('is-visible');
      draw();
    });
  }
  if (summaryRestartBtn) {
    listen(summaryRestartBtn, 'click', reset);
  }
  if (summaryReplayBtn) {
    listen(summaryReplayBtn, 'click', replay);
  }

  listen(document, 'visibilitychange', () => {
    if (document.hidden) {
      if (state === 'charging') cancelCharge();
      else if (state === 'flight' || state === 'replay') {
        const wasReplay = state === 'replay';
        cancelAnimationFrame(animation);
        animation = 0;
        finishPlayback(wasReplay);
      } else {
        cancelAnimationFrame(animation);
        animation = 0;
      }
    } else {
      ensureLoop();
    }
  });

  listen(reducedMotion, 'change', () => {
    if (reducedMotion.matches && (state === 'flight' || state === 'replay')) {
      const wasReplay = state === 'replay';
      cancelAnimationFrame(animation);
      animation = 0;
      finishPlayback(wasReplay);
    } else if (!reducedMotion.matches) {
      ensureLoop();
    }
  });

  listen(window, 'storage', (event) => {
    if (event.key === BEST_KEY || event.key === null) {
      best = readBest();
      $('[data-penalty-best]').textContent = best;
    }
  });

  let observer;
  if (typeof ResizeObserver === 'function') {
    observer = new ResizeObserver(resize);
    observer.observe(canvas);
  } else {
    listen(window, 'resize', resize);
  }

  updateAim();
  updateControls();
  resize();
  ensureLoop();

  return () => {
    disposed = true;
    clearTimers();
    cancelAnimationFrame(animation);
    observer?.disconnect();
    removers.forEach((remove) => remove());
  };
}
