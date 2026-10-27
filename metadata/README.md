# Runtime provenance

The executable scripts record the dataset SHA-256, Git commit, dirty-worktree flag,
Python version and platform directly in their JSON outputs. MLflow runs record the
dataset hash and Git provenance as parameters.

Do not maintain a hand-written `current_commit.txt`: it becomes stale as soon as the
repository changes. Generate final release artifacts from a clean tagged checkout so
that `git_dirty` is `false` and `git_commit` names the source revision used for the run.
