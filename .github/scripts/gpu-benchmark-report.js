'use strict';

// Builds the GPU benchmark report that `_modal-dispatch_gpu-benchmarks.yml` writes to the run summary and,
// for the `run-gpu-benchmarks` label, comments on the pull request. It lives here rather than
// inline in the workflow so `gpu-benchmark-report.test.js` can run it without GitHub.

const fs = require('node:fs');
const path = require('node:path');

const { testedCommitLine } = require('./tested-commit.js');

// GitHub rejects a comment over 65536 characters. The tables are sized to fit the budget, and
// the hard limit is only reached when the headings and limitations alone are too long.
const COMMENT_BUDGET = 60000;
const COMMENT_HARD_LIMIT = 65000;
// Room for the `<details>` wrapper and the truncation note around each table.
const TABLE_WRAPPER_CHARS = 250;
const MAX_LISTED_LIMITATIONS = 20;
// Worker failures can carry long diagnostics; the full text stays in the artifact.
const MAX_MESSAGE_CHARS = 300;

/** A green check for a passing result, a red cross for anything else, so the outcome reads at a glance. */
function icon(result) {
  return result === 'success' ? '✅' : '❌';
}

function clip(text, size) {
  return text.length > size ? `${text.slice(0, size)}...` : text;
}

/** Read one JSONL file; a missing file is no records and an unparsable line is a record that says so. */
function readRecords(file) {
  if (!fs.existsSync(file)) return [];
  return fs
    .readFileSync(file, 'utf8')
    .split('\n')
    .filter(Boolean)
    .map(line => {
      try {
        return JSON.parse(line);
      } catch {
        return { status: 'unparsable_record' };
      }
    });
}

/** List every record that is not a measurement, such as a GPU below the supported compute capability. */
function describeLimitations(records) {
  const problems = records.filter(record => record.status !== 'measured');
  if (!problems.length) return '';

  let text = '**Limitations (not measured):**\n\n';
  for (const record of problems.slice(0, MAX_LISTED_LIMITATIONS)) {
    const dims = record.shape || {};
    const shape = dims.batch != null ? ` ${dims.batch} x ${dims.workers} x ${dims.tasks}` : '';
    const message = clip(record.detail || (Array.isArray(record.details) ? record.details.join('; ') : ''), MAX_MESSAGE_CHARS);
    text += `- \`${record.status}\`${record.backend ? ` ${record.backend}` : ''}${shape}${message ? `: ${message}` : ''}\n`;
  }
  if (problems.length > MAX_LISTED_LIMITATIONS) {
    text += `- ${problems.length - MAX_LISTED_LIMITATIONS} additional records omitted; see the workflow artifact.\n`;
  }
  return `${text}\n`;
}

/** Map each preset to the conclusion of its own matrix job, which `needs` only reports as one result. */
function parsePresetResults(jobs) {
  const results = {};
  for (const job of jobs) {
    // A caller job prefixes the reusable workflow's job names, as in `gpu-benchmarks-label / gpu-benchmarks-run (wide)`.
    const match = /\/ gpu-benchmarks-run \((.+)\)$/.exec(job.name);
    if (match) results[match[1]] = job.conclusion;
  }
  return results;
}

async function readPresetResults({ github, context, core }) {
  try {
    const { data } = await github.rest.actions.listJobsForWorkflowRun({
      owner: context.repo.owner,
      repo: context.repo.repo,
      run_id: context.runId,
      per_page: 100,
    });
    return parsePresetResults(data.jobs);
  } catch (error) {
    core.warning(`Could not read the per-preset job results: ${error.message}`);
    return {};
  }
}

/** One section per preset: its heading and limitations, and the table text when the run produced one. */
function buildSection({ preset, result, outputDir }) {
  let body = `### ${icon(result)} Preset \`${preset}\`\n\n`;
  const tablePath = path.join(outputDir, `benchmark-${preset}.txt`);
  if (!fs.existsSync(tablePath)) {
    return {
      body: `${body}No results were produced; the install step or the Modal run failed, see the workflow logs.\n\n`,
      table: null,
    };
  }
  body += describeLimitations(readRecords(path.join(outputDir, `benchmark-${preset}.jsonl`)));
  return { body, table: fs.readFileSync(tablePath, 'utf8') };
}

/**
 * Build the full report and the comment text.
 *
 * The tables share what is left of the budget after the headings, limitations and wrappers, so
 * no mix of failures and tables can push the comment over GitHub's limit. The run summary takes
 * the full text; the comment falls back to a pointer only if it is still too long.
 */
function buildReport({ presets, jobResult, gpu, presetResults = {}, outputDir, runUrl, commitLine = '' }) {
  const commit = commitLine ? `${commitLine}\n\n` : '';
  const head = `## ${icon(jobResult)} GPU benchmark results\n\n**Job result:** ${icon(jobResult)} \`${jobResult}\` · **GPU:** ${gpu}\n\n${commit}`;
  const link = `[Workflow run](${runUrl})`;
  const sections = presets.map(preset => buildSection({ preset, result: presetResults[preset] ?? jobResult, outputDir }));

  const withTables = sections.filter(section => section.table !== null);
  const fixed = head.length + link.length + sections.reduce((sum, section) => sum + section.body.length, 0);
  const limit = Math.max(0, Math.floor((COMMENT_BUDGET - fixed - withTables.length * TABLE_WRAPPER_CHARS) / (withTables.length || 1)));

  let body = head;
  for (const section of sections) {
    body += section.body;
    if (section.table === null) continue;
    body += '<details>\n<summary>Results table</summary>\n\n```\n';
    body += section.table.substring(0, limit);
    if (section.table.length > limit) {
      body += '\n... truncated, see the workflow artifact for the full table and JSONL records\n';
    } else if (!section.table.endsWith('\n')) {
      // The closing fence must start its own line, or the table swallows it.
      body += '\n';
    }
    body += '```\n</details>\n\n';
  }
  body += link;

  const comment =
    body.length <= COMMENT_HARD_LIMIT
      ? body
      : `${head}The report is too long for a comment; see the run summary and the workflow artifacts.\n\n${link}`;
  return { body, comment };
}

/** Entry point for `actions/github-script`: write the run summary and, on a pull request, the comment. */
async function report({ github, context, core, env = process.env, outputDir = 'test-outputs' }) {
  const { body, comment } = buildReport({
    presets: JSON.parse(env.PRESETS),
    jobResult: env.JOB_RESULT,
    gpu: env.GPU,
    presetResults: await readPresetResults({ github, context, core }),
    commitLine: await testedCommitLine({ github, context, sha: env.TESTED_SHA }),
    outputDir,
    runUrl: `${context.payload.repository.html_url}/actions/runs/${context.runId}`,
  });

  await core.summary.addRaw(body).write();
  if (!context.payload.pull_request) return;
  try {
    await github.rest.issues.createComment({
      owner: context.repo.owner,
      repo: context.repo.repo,
      issue_number: context.payload.pull_request.number,
      body: comment,
    });
  } catch (error) {
    core.warning(`Could not post the pull request comment: ${error.message}`);
  }
}

module.exports = {
  buildReport,
  clip,
  describeLimitations,
  icon,
  parsePresetResults,
  readRecords,
  report,
  COMMENT_HARD_LIMIT,
};
