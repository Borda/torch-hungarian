"""Tests for the README relative-path rewrite run before packaging."""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / ".github" / "scripts" / "readme_absolute_urls.py"
RAW = "https://raw.githubusercontent.com/org/repo/abc123"
BLOB = "https://github.com/org/repo/blob/abc123"


@pytest.fixture(scope="module")
def rewriter() -> ModuleType:
    """Load the README rewrite script by path; it is a build step, not an importable package module."""
    spec = importlib.util.spec_from_file_location("readme_absolute_urls", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """Build a repository root holding one chart and a LICENSE."""
    (tmp_path / "docs" / "assets").mkdir(parents=True)
    (tmp_path / "docs" / "assets" / "chart.svg").write_text("<svg/>")
    (tmp_path / "LICENSE").write_text("")
    return tmp_path


def run_cli(readme: Path) -> subprocess.CompletedProcess[str]:
    """Run the rewrite script on a README the way the publish workflow does."""
    command = [sys.executable, str(SCRIPT), str(readme), "--repo", "org/repo", "--ref", "abc123"]
    return subprocess.run(command, text=True, capture_output=True, check=False)


class TestAbsolutize:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ('<img src="docs/assets/chart.svg" />', f'<img src="{RAW}/docs/assets/chart.svg" />'),
            ("![chart](./docs/assets/chart.svg)", f"![chart]({RAW}/docs/assets/chart.svg)"),
            ("[License](LICENSE)", f"[License]({BLOB}/LICENSE)"),
            ('<a href="LICENSE">x</a>', f'<a href="{BLOB}/LICENSE">x</a>'),
            ("[top](#license)", "[top](#license)"),
            ('<img src="https://example.com/a.png" />', '<img src="https://example.com/a.png" />'),
            ("```\n![chart](missing.svg)\n```\n", "```\n![chart](missing.svg)\n```\n"),
        ],
    )
    def test_rewrites(self, rewriter: ModuleType, checkout: Path, text: str, expected: str) -> None:
        """Images pin to the commit, other files link to the blob, and anchors, URLs and code stay."""
        assert rewriter.absolutize(text, "org/repo", "abc123", checkout)[0] == expected

    def test_missing_target_raises(self, rewriter: ModuleType, checkout: Path) -> None:
        """A path that does not exist aborts instead of publishing a broken link."""
        with pytest.raises(FileNotFoundError, match="gone.svg"):
            rewriter.absolutize("![x](docs/assets/gone.svg)", "org/repo", "abc123", checkout)


@pytest.fixture
def readme_mirror(rewriter: ModuleType, tmp_path: Path) -> tuple[str, Path]:
    """Return the real README text and a root holding an empty file for each relative path it names.

    The rewrite is checked against the real README text, not against whichever files the test host ships:
    the Modal GPU image leaves out `.modal`, so a link into it would fail there for a reason unrelated to the rewrite.
    """
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for html_target, markdown_target in re.findall(r'(?:src|href)="([^"]+)"|\]\(([^)\s]+)\)', text):
        target = html_target or markdown_target
        if rewriter.is_relative(target):
            stub = tmp_path / target.removeprefix("./").split("#", 1)[0].split("?", 1)[0]
            stub.parent.mkdir(parents=True, exist_ok=True)
            stub.touch()
    return text, tmp_path


class TestRepositoryReadme:
    @pytest.mark.skipif(
        not (REPO_ROOT / ".modal").is_dir(), reason="a stripped copy of the checkout, such as the Modal GPU image"
    )
    def test_every_relative_path_exists(self, rewriter: ModuleType) -> None:
        """Every relative image and link in the real README names a file in the checkout."""
        text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        assert rewriter.absolutize(text, "Borda/torch-hungarian", "abc123", REPO_ROOT)[1]

    def test_no_relative_path_survives(self, rewriter: ModuleType, readme_mirror: tuple[str, Path]) -> None:
        """Every relative image and link in the real README is rewritten."""
        text, root = readme_mirror
        rewritten, replaced = rewriter.absolutize(text, "Borda/torch-hungarian", "abc123", root)
        remaining = [t for t in re.findall(r'(?:src|href)="([^"]+)"', rewritten) if rewriter.is_relative(t)]
        assert remaining == []
        assert replaced

    @pytest.mark.parametrize("name", ["linear-assignment", "hungarian-steps", "cpu-vs-triton", "backend-selection"])
    def test_illustrations_resolve_to_the_commit(
        self, rewriter: ModuleType, readme_mirror: tuple[str, Path], name: str
    ) -> None:
        """Each README illustration is pinned to the built commit on raw.githubusercontent.com."""
        text, root = readme_mirror
        rewritten, _ = rewriter.absolutize(text, "Borda/torch-hungarian", "abc123", root)
        url = f"https://raw.githubusercontent.com/Borda/torch-hungarian/abc123/docs/assets/{name}.svg"
        assert f'src="{url}"' in rewritten


class TestCommandLine:
    def test_rewrites_in_place(self, checkout: Path) -> None:
        """The README is rewritten in place against the commit."""
        readme = checkout / "README.md"
        readme.write_text("![chart](docs/assets/chart.svg)\n", encoding="utf-8")
        run_cli(readme).check_returncode()
        assert readme.read_text(encoding="utf-8") == f"![chart]({RAW}/docs/assets/chart.svg)\n"

    def test_missing_target_exits_nonzero(self, checkout: Path) -> None:
        """A dangling relative path fails the run and leaves the README untouched."""
        readme = checkout / "README.md"
        readme.write_text("![chart](docs/assets/gone.svg)\n", encoding="utf-8")
        result = run_cli(readme)
        assert (result.returncode, readme.read_text(encoding="utf-8")) == (1, "![chart](docs/assets/gone.svg)\n")
