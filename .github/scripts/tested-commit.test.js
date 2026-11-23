'use strict';

// Run with `node --test .github/scripts/*.test.js`; no dependencies beyond Node itself.

const assert = require('node:assert/strict');
const { test } = require('node:test');

const { commitLine, testedCommitLine } = require('./tested-commit.js');

const REPO_URL = 'https://github.com/o/r';
const TESTED = 'a'.repeat(40);
const HEAD = 'b'.repeat(40);

test('commitLine names the commit with a short, linked sha', () => {
  assert.equal(commitLine({ sha: TESTED, repoUrl: REPO_URL }), `**Tested commit:** [\`aaaaaaa\`](${REPO_URL}/commit/${TESTED})`);
});

test('commitLine says when no commit is known', () => {
  assert.equal(commitLine({ repoUrl: REPO_URL }), '**Tested commit:** unknown');
});

test('commitLine says the commit is the current head', () => {
  assert.match(commitLine({ sha: TESTED, repoUrl: REPO_URL, headSha: TESTED }), /the current pull request head$/);
});

test('commitLine counts the commits the head is ahead, in the singular and the plural', () => {
  const one = commitLine({ sha: TESTED, repoUrl: REPO_URL, headSha: HEAD, commitsBehind: 1 });
  const many = commitLine({ sha: TESTED, repoUrl: REPO_URL, headSha: HEAD, commitsBehind: 3 });
  assert.match(one, / 1 commit behind the pull request head \[`bbbbbbb`\]/);
  assert.match(many, / 3 commits behind the pull request head \[`bbbbbbb`\]/);
});

test('commitLine says so when the head is not a descendant of the tested commit', () => {
  assert.match(commitLine({ sha: TESTED, repoUrl: REPO_URL, headSha: HEAD }), /the pull request head is now \[`bbbbbbb`\].*not a descendant/);
});

function harness({ pullRequest = { number: 7 }, headSha = HEAD, comparison = { status: 'ahead', ahead_by: 2 }, pullsError, compareError } = {}) {
  const calls = { compare: [] };
  const github = {
    rest: {
      pulls: {
        get: async () => {
          if (pullsError) throw pullsError;
          return { data: { head: { sha: headSha } } };
        },
      },
      repos: {
        compareCommitsWithBasehead: async options => {
          calls.compare.push(options.basehead);
          if (compareError) throw compareError;
          return { data: comparison };
        },
      },
    },
  };
  const context = { repo: { owner: 'o', repo: 'r' }, payload: { repository: { html_url: REPO_URL }, pull_request: pullRequest } };
  return { github, context, calls };
}

test('testedCommitLine compares the tested commit with the live head', async () => {
  const { github, context, calls } = harness();
  const line = await testedCommitLine({ github, context, sha: TESTED });
  assert.match(line, /2 commits behind/);
  assert.deepEqual(calls.compare, [`${TESTED}...${HEAD}`]);
});

test('testedCommitLine skips the comparison when the head is the tested commit', async () => {
  const { github, context, calls } = harness({ headSha: TESTED });
  assert.match(await testedCommitLine({ github, context, sha: TESTED }), /the current pull request head$/);
  assert.deepEqual(calls.compare, []);
});

test('testedCommitLine reports a diverged head after a force push', async () => {
  const { github, context } = harness({ comparison: { status: 'diverged', ahead_by: 1 } });
  assert.match(await testedCommitLine({ github, context, sha: TESTED }), /not a descendant/);
});

test('testedCommitLine gives only the commit outside a pull request', async () => {
  const { github, context, calls } = harness({ pullRequest: null });
  const line = await testedCommitLine({ github, context, sha: TESTED });
  assert.equal(line, commitLine({ sha: TESTED, repoUrl: REPO_URL }));
  assert.deepEqual(calls.compare, []);
});

test('testedCommitLine falls back to the commit when the head cannot be read', async () => {
  const { github, context } = harness({ pullsError: new Error('boom') });
  assert.equal(await testedCommitLine({ github, context, sha: TESTED }), commitLine({ sha: TESTED, repoUrl: REPO_URL }));
});

test('testedCommitLine gives only the commit when the comparison fails, never a divergence claim', async () => {
  const { github, context } = harness({ compareError: new Error('unavailable') });
  const line = await testedCommitLine({ github, context, sha: TESTED });
  assert.equal(line, commitLine({ sha: TESTED, repoUrl: REPO_URL }));
  assert.doesNotMatch(line, /descendant/);
});
