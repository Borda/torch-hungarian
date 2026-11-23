'use strict';

// Builds the pull request comment for the GPU test run: one section per device-count leg, each with
// its own result, pytest summary line and log. `_modal-gpu-tests.yml` calls `report`; it lives here
// rather than inline in the workflow so `gpu-test-report.test.js` can run it without GitHub.

const fs = require('node:fs');
const path = require('node:path');

const { clip, icon } = require('./gpu-benchmark-report.js');
const { testedCommitLine } = require('./tested-commit.js');

// GitHub rejects a comment over 65536 characters. The logs share what the headings leave of the
// budget, and the hard limit is only reached when the headings alone are too long.
const COMMENT_BUDGET = 60000;
const COMMENT_HARD_LIMIT = 65000;
// Room for the `<details>` wrapper and the truncation note around each log.
const LOG_WRAPPER_CHARS = 250;
// A leg's job is named `gpu-tests-run (<devices> GPU)`; the caller prefix in front of it is ignored.
const LEG_JOB_NAME = /gpu-tests-run \((\d+) GPU\)$/;
const SUMMARY_CHARS = 300;

/** The artifact directory a leg's log is downloaded into. */
function logPath(outputDir, devices) {
  return path.join(outputDir, `gpu-pytest-output-${devices}gpu`, 'pytest-output.log');
}

function legTitle(devices) {
  return devices === 1 ? '1 GPU' : `${devices} GPUs`;
}

/** Map each leg's device count to its job conclusion, from the run's job list. */
function parseLegResults(jobs) {
  const results = {};
  for (const job of jobs) {
    const match = LEG_JOB_NAME.exec(job.name);
    if (match) results[Number(match[1])] = job.conclusion;
  }
  return results;
}

/** Read one leg's log without terminal colour codes, or null when the leg produced none. */
function readLog(file) {
  if (!fs.existsSync(file)) return null;
  // eslint-disable-next-line no-control-regex
  return fs.readFileSync(file, 'utf8').replace(/\x1b\[[0-9;]*m/g, '');
}

/** The last pytest line that counts passed or failed tests, or '' when the log has none. */
function summaryLine(output) {
  const line = output.split('\n').filter(text => text.includes('passed') || text.includes('failed')).pop();
  return line ? clip(line.trim(), SUMMARY_CHARS) : '';
}

/** One leg's heading and summary, and its log text when the leg produced one. */
function buildSection({ devices, result, outputDir }) {
  const title = legTitle(devices);
  const heading = result ? `${icon(result)} ${title}` : `❔ ${title} — result unavailable`;
  let body = `### ${heading}\n\n`;
  const log = readLog(logPath(outputDir, devices));
  if (log === null) {
    return {
      body: `${body}Pytest log unavailable. See the workflow run for details.\n\n`,
      log: null,
    };
  }
  const summary = summaryLine(log);
  if (summary) body += `**Summary:** ${summary}\n\n`;
  return { body, log };
}

/**
 * Build the comment text for the legs of one run.
 *
 * The logs share what is left of the budget after the headings and wrappers, so no mix of long
 * logs can push the comment over GitHub's limit. Individual headings use only per-leg conclusions.
 */
function buildReport({ legs, jobResult, legResults = {}, outputDir, runUrl, commitLine = '' }) {
  const commit = commitLine ? `${commitLine}\n\n` : '';
  const head = `## ${icon(jobResult)} GPU test results\n\n**Job result:** ${icon(jobResult)} \`${jobResult}\`\n\n${commit}`;
  const link = `[Workflow run](${runUrl})`;
  const sections = legs.map(devices => buildSection({ devices, result: legResults[devices], outputDir }));

  const withLogs = sections.filter(section => section.log !== null);
  const fixed = head.length + link.length + sections.reduce((sum, section) => sum + section.body.length, 0);
  const limit = Math.max(0, Math.floor((COMMENT_BUDGET - fixed - withLogs.length * LOG_WRAPPER_CHARS) / (withLogs.length || 1)));

  let body = head;
  for (const section of sections) {
    body += section.body;
    if (section.log === null) continue;
    body += '<details>\n<summary>Full test output</summary>\n\n```\n';
    body += section.log.substring(0, limit);
    if (section.log.length > limit) {
      body += '\n... truncated, see the workflow artifact for the full log\n';
    } else if (!section.log.endsWith('\n')) {
      // The closing fence must start its own line, or the log swallows it.
      body += '\n';
    }
    body += '```\n</details>\n\n';
  }
  body += link;

  return body.length <= COMMENT_HARD_LIMIT
    ? body
    : `${head}The report is too long for a comment; see the workflow artifacts.\n\n${link}`;
}

async function readLegResults({ github, context, core }) {
  try {
    const { data } = await github.rest.actions.listJobsForWorkflowRun({
      owner: context.repo.owner,
      repo: context.repo.repo,
      run_id: context.runId,
      per_page: 100,
    });
    return parseLegResults(data.jobs);
  } catch (error) {
    core.warning(`Could not read the per-leg job results: ${error.message}`);
    return {};
  }
}

/** Entry point for `actions/github-script`: comment the result of every leg on the pull request. */
async function report({ github, context, core, env = process.env, outputDir = 'test-outputs' }) {
  const body = buildReport({
    legs: JSON.parse(env.LEGS),
    jobResult: env.GPU_JOB_RESULT,
    legResults: await readLegResults({ github, context, core }),
    commitLine: await testedCommitLine({ github, context, sha: env.TESTED_SHA }),
    outputDir,
    runUrl: `${context.payload.repository.html_url}/actions/runs/${context.runId}`,
  });

  // Let a rejected write fail the workflow step so label removal waits for delivery.
  await github.rest.issues.createComment({
    owner: context.repo.owner,
    repo: context.repo.repo,
    issue_number: context.payload.pull_request.number,
    body,
  });
}

module.exports = {
  buildReport,
  legTitle,
  parseLegResults,
  report,
  summaryLine,
  COMMENT_HARD_LIMIT,
};
