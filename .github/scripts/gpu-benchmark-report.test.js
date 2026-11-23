'use strict';

// Run with `node --test .github/scripts/*.test.js`; no dependencies beyond Node itself.

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { test } = require('node:test');

const {
  buildReport,
  clip,
  describeLimitations,
  icon,
  parsePresetResults,
  readRecords,
  report,
  COMMENT_HARD_LIMIT,
} = require('./gpu-benchmark-report.js');

const GITHUB_COMMENT_LIMIT = 65536;

/** Write benchmark outputs into a fresh directory; `null` leaves a file out. */
function outputs(files) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'gpu-benchmark-report-'));
  for (const [name, content] of Object.entries(files)) {
    if (content !== null) fs.writeFileSync(path.join(dir, name), content);
  }
  return dir;
}

const jsonl = records => records.map(record => JSON.stringify(record)).join('\n');
const failure = detail => ({ status: 'worker_failed', backend: 'triton', shape: { batch: 1, workers: 2, tasks: 3 }, detail });

function build(overrides) {
  return buildReport({ presets: ['rfdetr', 'wide'], jobResult: 'success', gpu: 'L4', runUrl: 'https://run', ...overrides });
}

test('icon is a green check only for success', () => {
  assert.equal(icon('success'), '✅');
  for (const result of ['failure', 'cancelled', 'skipped', undefined]) assert.equal(icon(result), '❌');
});

test('clip keeps short text and marks cut text', () => {
  assert.equal(clip('abc', 3), 'abc');
  assert.equal(clip('abcd', 3), 'abc...');
});

test('readRecords returns nothing for a missing file and flags an unparsable line', () => {
  const dir = outputs({ 'ok.jsonl': '{"status":"measured"}\nnot json\n\n' });
  assert.deepEqual(readRecords(path.join(dir, 'missing.jsonl')), []);
  assert.deepEqual(readRecords(path.join(dir, 'ok.jsonl')), [{ status: 'measured' }, { status: 'unparsable_record' }]);
});

test('describeLimitations lists only records that were not measured', () => {
  assert.equal(describeLimitations([{ status: 'measured' }]), '');
  const text = describeLimitations([
    { status: 'measured' },
    { status: 'gpu_ineligible', details: ['T4 has compute capability 7.5', 'skipped'] },
    failure('boom'),
  ]);
  assert.match(text, /^\*\*Limitations \(not measured\):\*\*/);
  assert.match(text, /- `gpu_ineligible`: T4 has compute capability 7.5; skipped\n/);
  assert.match(text, /- `worker_failed` triton 1 x 2 x 3: boom\n/);
  assert.doesNotMatch(text, /measured`/);
});

test('describeLimitations clips long diagnostics and caps the list', () => {
  const records = Array.from({ length: 25 }, () => failure('x'.repeat(4000)));
  const text = describeLimitations(records);
  assert.equal(text.match(/^- `worker_failed`/gm).length, 20);
  assert.match(text, /- 5 additional records omitted/);
  assert.ok(text.length < 20 * 450);
});

test('parsePresetResults reads the benchmark matrix job names and ignores other jobs', () => {
  const jobs = [
    { name: 'gpu-benchmarks-label / gpu-benchmarks-resolve', conclusion: 'success' },
    { name: 'gpu-benchmarks-label / gpu-benchmarks-run (rfdetr)', conclusion: 'success' },
    { name: 'gpu-benchmarks-label / gpu-benchmarks-run (wide)', conclusion: 'failure' },
    { name: 'gpu-benchmarks-label / gpu-benchmarks-report', conclusion: null },
    { name: 'gpu-tests-label / gpu-tests-run', conclusion: 'success' },
    { name: 'gpu-tests-label / gpu-tests-report', conclusion: null },
  ];
  assert.deepEqual(parsePresetResults(jobs), { rfdetr: 'success', wide: 'failure' });
});

test('buildReport marks the run and each preset with its own outcome', () => {
  const dir = outputs({ 'benchmark-rfdetr.txt': 'table', 'benchmark-wide.txt': 'table' });
  const { body } = build({ jobResult: 'failure', presetResults: { rfdetr: 'success', wide: 'failure' }, outputDir: dir });
  assert.match(body, /^## ❌ GPU benchmark results/);
  assert.match(body, /\*\*Job result:\*\* ❌ `failure` · \*\*GPU:\*\* L4/);
  assert.match(body, /### ✅ Preset `rfdetr`/);
  assert.match(body, /### ❌ Preset `wide`/);
  assert.match(body, /\[Workflow run\]\(https:\/\/run\)$/);
});

test('buildReport puts the tested commit line under the job result, and nothing when there is none', () => {
  const line = '**Tested commit:** [`aaaaaaa`](https://c) · 2 commits behind the pull request head';
  const outputDir = outputs({});
  const { body, comment } = build({ commitLine: line, outputDir });
  for (const text of [body, comment]) assert.match(text, /GPU:\*\* L4\n\n\*\*Tested commit:\*\* \[`aaaaaaa`\]/);
  assert.doesNotMatch(build({ outputDir }).body, /Tested commit/);
});

test('buildReport falls back to the overall result for a preset without a job result', () => {
  const dir = outputs({ 'benchmark-rfdetr.txt': 'table', 'benchmark-wide.txt': 'table' });
  const { body } = build({ outputDir: dir });
  assert.equal(body.match(/### ✅ Preset/g).length, 2);
});

test('buildReport says so when a preset produced no table and lists limitations when it did', () => {
  const dir = outputs({
    'benchmark-rfdetr.txt': 'table',
    'benchmark-rfdetr.jsonl': jsonl([{ status: 'gpu_ineligible', details: ['no GPU'] }]),
    'benchmark-wide.txt': null,
  });
  const { body } = build({ presetResults: { rfdetr: 'failure', wide: 'failure' }, outputDir: dir });
  assert.match(body, /- `gpu_ineligible`: no GPU/);
  assert.match(body, /### ❌ Preset `wide`[^]*No results were produced/);
});

test('buildReport truncates a table that does not fit and says where the full one is', () => {
  const dir = outputs({ 'benchmark-rfdetr.txt': 'row\n'.repeat(60000), 'benchmark-wide.txt': 'short' });
  const { body } = build({ outputDir: dir });
  assert.match(body, /\.\.\. truncated, see the workflow artifact/);
  assert.match(body, /short\n```\n<\/details>/);
});

test('buildReport keeps the comment under the GitHub limit with many long failures and big tables', () => {
  const failures = jsonl(Array.from({ length: 30 }, () => failure('x'.repeat(4000))));
  const dir = outputs({
    'benchmark-rfdetr.txt': 'row\n'.repeat(60000),
    'benchmark-rfdetr.jsonl': failures,
    'benchmark-wide.txt': 'row\n'.repeat(60000),
    'benchmark-wide.jsonl': failures,
  });
  const { comment } = build({ jobResult: 'failure', outputDir: dir });
  assert.ok(comment.length < GITHUB_COMMENT_LIMIT, `comment is ${comment.length} characters`);
});

test('buildReport posts a pointer when the headings and limitations alone are too long', () => {
  const presets = Array.from({ length: 1000 }, (_, index) => `preset-${index}`);
  const { body, comment } = build({ presets, outputDir: outputs({}) });
  assert.ok(body.length > COMMENT_HARD_LIMIT);
  assert.match(comment, /too long for a comment/);
  assert.ok(comment.length < GITHUB_COMMENT_LIMIT);
});

/** Stand-ins for the objects `actions/github-script` passes in. */
function harness({ pullRequest = { number: 7 }, jobs, postFails = false } = {}) {
  const calls = { comments: [], warnings: [], summary: [] };
  const github = {
    rest: {
      actions: {
        listJobsForWorkflowRun: async () => {
          if (!jobs) throw new Error('no access');
          return { data: { jobs } };
        },
      },
      issues: {
        createComment: async args => {
          if (postFails) throw new Error('rejected');
          calls.comments.push(args);
        },
      },
    },
  };
  const core = {
    summary: { addRaw: text => ({ write: async () => calls.summary.push(text) }) },
    warning: message => calls.warnings.push(message),
  };
  const context = {
    repo: { owner: 'owner', repo: 'repo' },
    runId: 42,
    payload: { repository: { html_url: 'https://github.com/owner/repo' }, pull_request: pullRequest },
  };
  return { github, core, context, calls };
}

const env = { PRESETS: '["rfdetr"]', JOB_RESULT: 'success', GPU: 'L4' };

test('report writes the summary and comments on a pull request', async () => {
  const dir = outputs({ 'benchmark-rfdetr.txt': 'table' });
  const { calls, ...rest } = harness({ jobs: [{ name: 'gpu-benchmarks-label / gpu-benchmarks-run (rfdetr)', conclusion: 'success' }] });
  await report({ ...rest, env, outputDir: dir });
  assert.equal(calls.summary.length, 1);
  assert.equal(calls.comments.length, 1);
  assert.equal(calls.comments[0].issue_number, 7);
  assert.match(calls.comments[0].body, /### ✅ Preset `rfdetr`/);
  assert.match(calls.comments[0].body, /actions\/runs\/42\)$/);
});

test('report writes only the summary outside a pull request', async () => {
  const { calls, ...rest } = harness({ pullRequest: null, jobs: [] });
  await report({ ...rest, env, outputDir: outputs({}) });
  assert.equal(calls.summary.length, 1);
  assert.deepEqual(calls.comments, []);
});

test('report warns and carries on when job results or the comment are unavailable', async () => {
  const { calls, ...rest } = harness({ postFails: true });
  await report({ ...rest, env, outputDir: outputs({ 'benchmark-rfdetr.txt': 'table' }) });
  assert.equal(calls.summary.length, 1);
  assert.equal(calls.warnings.length, 2);
  assert.match(calls.warnings[0], /per-preset job results/);
  assert.match(calls.warnings[1], /pull request comment/);
});
