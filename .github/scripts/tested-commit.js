'use strict';

// The line a GPU run adds to its pull request comment and run summary to say which commit it
// tested. A run can start long after the label was applied and the pull request keeps moving,
// so the comment also says how far the pull request head is from that commit.

const SHORT_SHA_LENGTH = 7;

function link(sha, repoUrl) {
  return `[\`${sha.slice(0, SHORT_SHA_LENGTH)}\`](${repoUrl}/commit/${sha})`;
}

/**
 * Describe the tested commit against the pull request head.
 *
 * `headSha` is the current head, or undefined outside a pull request or when it could not be read.
 * `commitsBehind` is how many commits the head has on top of the tested one, or undefined when
 * the two share no such history, such as after a force push.
 */
function commitLine({ sha, repoUrl, headSha, commitsBehind }) {
  if (!sha) return '**Tested commit:** unknown';
  const tested = `**Tested commit:** ${link(sha, repoUrl)}`;
  if (!headSha) return tested;
  if (headSha === sha) return `${tested} · the current pull request head`;
  if (commitsBehind === undefined) return `${tested} · the pull request head is now ${link(headSha, repoUrl)}, not a descendant of it`;
  const noun = commitsBehind === 1 ? 'commit' : 'commits';
  return `${tested} · ${commitsBehind} ${noun} behind the pull request head ${link(headSha, repoUrl)}`;
}

/** Entry point for `actions/github-script`: read the live head and build the line; never throws. */
async function testedCommitLine({ github, context, sha }) {
  const repoUrl = context.payload.repository.html_url;
  const pullRequest = context.payload.pull_request;
  if (!pullRequest || !sha) return commitLine({ sha, repoUrl });

  const { owner, repo } = context.repo;
  try {
    const { data } = await github.rest.pulls.get({ owner, repo, pull_number: pullRequest.number });
    const headSha = data.head.sha;
    if (headSha === sha) return commitLine({ sha, repoUrl, headSha });
    const comparison = await github.rest.repos.compareCommitsWithBasehead({ owner, repo, basehead: `${sha}...${headSha}` });
    // Only a comparison that answered can say the head is not a descendant.
    const commitsBehind = comparison.data.status === 'ahead' ? comparison.data.ahead_by : undefined;
    return commitLine({ sha, repoUrl, headSha, commitsBehind });
  } catch (error) {
    // The commit alone still answers what ran; the distance is a convenience.
    return commitLine({ sha, repoUrl });
  }
}

module.exports = { commitLine, testedCommitLine };
