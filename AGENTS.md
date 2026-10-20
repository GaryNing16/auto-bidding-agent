# Repository working agreement

- Read README.md and docs/rules.md before editing environment behavior.
- `src/autobid` is the competition implementation; `third_party/auctionnet` contains pinned, unmodified references.
- Preserve sparse participation: absent (request_id, player_id) means no participation.
- Agents receive only immutable own-history observations. Do not expose current/future requests, full opportunity counts, opponent trajectories or debug traces.
- Preserve one alpha per player per time slice and total won pvalue as reward.
- Budget, tie-breaking, time origin and alpha bounds must remain explicit configuration; do not label local assumptions as official rules.
- Run `python -m unittest discover -s tests -v` and an affected CLI smoke command before a PR.
- Split datasets by whole source day/period; never randomly split rows. Keep private data and trained weights out of git.
- Do not modify vendored source without retaining license and documenting modifications/provenance.
- Keep all generated run outputs in ignored `runs/`, data in `data/`, models in `models/`.
