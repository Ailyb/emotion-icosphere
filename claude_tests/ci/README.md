# CI

`tests.yml.example` runs the 17 model-free tests on every push. GitHub rejected it as a workflow file because the token used by Claude Code lacks the `workflow` scope. To enable: `gh auth refresh -s workflow`, then `mkdir -p .github/workflows && git mv claude_tests/ci/tests.yml.example .github/workflows/tests.yml` and push.
