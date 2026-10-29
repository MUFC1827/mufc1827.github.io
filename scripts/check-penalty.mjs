import assert from 'node:assert/strict';
import {
  simulatePenaltyShot,
  previewPenaltyTrajectory,
  PENALTY_PHYSICS,
  PENALTY_MODES,
  PENALTY_DIFFICULTIES,
  renderPenalty,
} from '../src/penalty.js';

const base = { mode: 'placement', power: 0.75, accuracyScale: 0, seed: 1827, keeper: { enabled: false } };
const fixtures = [
  ['正中扑救', { aim: { x: 0, y: 1.3 }, seed: 1 }, 'save'],
  ['角落进球', { ...base, aim: { x: 2.8, y: 1.8 }, power: 0.9 }, 'goal'],
  ['擦柱入网', { ...base, aim: { x: 3.482, y: 1 } }, 'goal'],
  ['击柱未进', { ...base, aim: { x: 3.57, y: 1 } }, 'post'],
  ['高出横梁', { ...base, aim: { x: 0, y: 4 } }, 'high'],
  ['偏出球门', { ...base, aim: { x: 5.2, y: 1 } }, 'wide'],
];
const results = new Map();
for (const [label, input, expected] of fixtures) {
  const result = simulatePenaltyShot(input);
  assert.equal(result.outcome, expected, label);
  assert(
    result.frames.every((frame) =>
      [frame.time, frame.ball.x, frame.ball.y, frame.ball.z, frame.keeper.x, frame.keeper.y].every(Number.isFinite)
    ),
    `${label}：轨迹出现无效数值`
  );
  results.set(label, result);
}
const glance = results.get('擦柱入网');
assert(glance.frameContacts.length > 0, '擦柱入网必须实际碰到门框');
assert(glance.event.time > glance.frameContacts[0].time, '碰框后必须继续计算入网');
assert(results.get('击柱未进').frameContacts.length > 0, '击柱结果必须实际碰到门框');

const spinInput = { ...base, mode: 'curl', aim: { x: 0, y: 1.3 } };
const left = simulatePenaltyShot({ ...spinInput, spin: -1 });
const right = simulatePenaltyShot({ ...spinInput, spin: 1 });
assert(left.stats.point.x < 0 && right.stats.point.x > 0, '旋转方向必须实际改变球路');
assert.deepEqual(simulatePenaltyShot(fixtures[2][1]), glance, '相同参数与种子必须能重现轨迹');

// Verify trajectory preview and difficulty presets
const previewLeft = previewPenaltyTrajectory({ mode: 'curl', aim: { x: 0, y: 1.4 }, power: 0.78, spin: -0.8 });
const previewRight = previewPenaltyTrajectory({ mode: 'curl', aim: { x: 0, y: 1.4 }, power: 0.78, spin: 0.8 });
assert(previewLeft.points.length > 5, '预览轨迹应包含多个三维采样点');
assert(previewLeft.curveOffsetX < -0.2 && previewRight.curveOffsetX > 0.2, '预览轨迹必须反映旋转弧线偏移');
assert(PENALTY_PHYSICS.goalZ === 11 && PENALTY_MODES.curl && PENALTY_DIFFICULTIES.legend, '物理与模式难度常量导出完整');

const markup = renderPenalty();
assert(markup.includes('data-penalty-summary') && markup.includes('data-penalty-difficulty') && markup.includes('data-penalty-zone'), 'UI 结构包含结算面板、难度切换与快速落点');

console.log('点球回归检查通过：扑救、入网、门框反弹、偏出、旋转、三维轨迹预览、难度预设与可复现轨迹。');
