'use strict';

// Run with `node --test .github/scripts/*.test.js`; no dependencies beyond Node itself.

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { test } = require('node:test');

const { buildReport, legTitle, logPath, parseLegResults, report, summaryLine, COMMENT_HARD_LIMIT } = require('./gpu-test-report.js');

const GITHUB_COMMENT_LIMIT = 65536;

/** Write one log per device count into a fresh directory; `null` leaves a leg's log out. */
function logs(files) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'gpu-test-report-'));
  for (const [devices, content] of Object.entries(files)) {
    if (content === null) continue;
    const file = logPath(dir, Number(devices));
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, content);
  }
  return dir;
}

function build(overrides) {
  return buildReport({ legs: [1, 2], jobResult: 'success', runUrl: 'https://run', ...overrides });
}

test('legTitle pluralises the device count', () => {
  assert.equal(legTitle(1), '1 GPU');
  assert.equal(legTitle(2), '2 GPUs');
});

test('parseLegResults reads the leg job names and ignores other jobs', () => {
  const jobs = [
    { name: 'gpu-tests-label / gpu-tests-run (1 GPU)', conclusion: 'success' },
    { name: 'gpu-tests-label / gpu-tests-run (2 GPU)', conclusion: 'failure' },
    { name: 'gpu-tests-label / gpu-tests-report', conclusion: null },
    { name: 'gpu-benchmarks-label / gpu-benchmarks-run (rfdetr)', conclusion: 'success' },
  ];
  assert.deepEqual(parseLegResults(jobs), { 1: 'success', 2: 'failure' });
});

test('summaryLine keeps the last line that counts tests, and nothing when there is none', () => {
  assert.equal(summaryLine('a\n== 3 passed in 1s ==\n== 1 failed, 3 passed ==\n\n'), '== 1 failed, 3 passed ==');
  assert.equal(summaryLine('collected nothing\n'), '');
});

test('buildReport gives each leg its own heading, outcome and summary', () => {
  const dir = logs({ 1: '== 40 passed, 2 skipped ==\n', 2: '== 2 failed ==\n' });
  const body = build({ jobResult: 'failure', legResults: { 1: 'success', 2: 'failure' }, outputDir: dir });
  assert.match(body, /^## ❌ GPU test results/);
  assert.match(body, /\*\*Job result:\*\* ❌ `failure`/);
  assert.match(body, /### ✅ 1 GPU\n\n\*\*Summary:\*\* == 40 passed, 2 skipped ==/);
  assert.match(body, /### ❌ 2 GPUs\n\n\*\*Summary:\*\* == 2 failed ==/);
  assert.match(body, /\[Workflow run\]\(https:\/\/run\)$/);
});

test('buildReport strips terminal colour codes from the log', () => {
  const body = build({ legs: [1], outputDir: logs({ 1: '\x1b[32m== 1 passed ==\x1b[0m\n' }) });
  assert.doesNotMatch(body, /\x1b/);
  assert.match(body, /== 1 passed ==/);
});

test('buildReport puts the tested commit line under the job result, and nothing when there is none', () => {
  const line = '**Tested commit:** [`aaaaaaa`](https://c) · the current pull request head';
  const outputDir = logs({});
  assert.match(build({ commitLine: line, outputDir }), /`success`\n\n\*\*Tested commit:\*\* \[`aaaaaaa`\]/);
  assert.doesNotMatch(build({ outputDir }), /Tested commit/);
});

test('buildReport falls back to the overall result for a leg without a job result', () => {
  const body = build({ outputDir: logs({}) });
  assert.equal(body.match(/### ✅/g).length, 2);
});

test('buildReport says so when a leg produced no log and still shows the other leg', () => {
  const body = build({ jobResult: 'failure', legResults: { 1: 'success', 2: 'failure' }, outputDir: logs({ 1: '== 1 passed ==\n', 2: null }) });
  assert.match(body, /### ❌ 2 GPUs\n\nNo pytest log was produced/);
  assert.match(body, /### ✅ 1 GPU[^]*== 1 passed ==/);
});

test('buildReport truncates a log that does not fit and says where the full one is', () => {
  const body = build({ outputDir: logs({ 1: 'row\n'.repeat(60000), 2: 'short' }) });
  assert.match(body, /\.\.\. truncated, see the workflow artifact/);
  assert.match(body, /short\n```\n<\/details>/);
});

test('buildReport keeps the comment under the GitHub limit with huge logs on every leg', () => {
  const legs = [1, 2, 4, 8];
  const files = Object.fromEntries(legs.map(devices => [devices, 'row\n'.repeat(60000)]));
  const body = build({ legs, outputDir: logs(files) });
  assert.ok(body.length < GITHUB_COMMENT_LIMIT, `comment is ${body.length} characters`);
});

test('buildReport posts a pointer when the headings alone are too long', () => {
  const legs = Array.from({ length: 3000 }, (_, index) => index + 1);
  const body = build({ legs, outputDir: logs({}) });
  assert.match(body, /too long for a comment/);
  assert.ok(body.length < COMMENT_HARD_LIMIT);
});

/** Stand-ins for the objects `actions/github-script` passes in. */
function harness({ jobs, postFails = false } = {}) {
  const calls = { comments: [], warnings: [] };
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
  const core = { warning: message => calls.warnings.push(message) };
  const context = {
    repo: { owner: 'owner', repo: 'repo' },
    runId: 42,
    payload: { repository: { html_url: 'https://github.com/owner/repo' }, pull_request: { number: 7 } },
  };
  return { github, core, context, calls };
}

const env = { LEGS: '[1,2]', GPU_JOB_RESULT: 'success' };

test('report comments each leg on the pull request', async () => {
  const jobs = [
    { name: 'gpu-tests-label / gpu-tests-run (1 GPU)', conclusion: 'success' },
    { name: 'gpu-tests-label / gpu-tests-run (2 GPU)', conclusion: 'failure' },
  ];
  const { calls, ...rest } = harness({ jobs });
  await report({ ...rest, env, outputDir: logs({ 1: '== 1 passed ==\n', 2: '== 1 failed ==\n' }) });
  assert.equal(calls.comments.length, 1);
  assert.equal(calls.comments[0].issue_number, 7);
  assert.match(calls.comments[0].body, /### ✅ 1 GPU/);
  assert.match(calls.comments[0].body, /### ❌ 2 GPUs/);
  assert.match(calls.comments[0].body, /actions\/runs\/42\)$/);
});

test('report warns and carries on when job results or the comment are unavailable', async () => {
  const { calls, ...rest } = harness({ postFails: true });
  await report({ ...rest, env, outputDir: logs({ 1: '== 1 passed ==\n' }) });
  assert.equal(calls.warnings.length, 2);
  assert.match(calls.warnings[0], /per-leg job results/);
  assert.match(calls.warnings[1], /pull request comment/);
});
