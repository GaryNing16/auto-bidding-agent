# Pinned upstream reference

These files are unmodified copies from `alimama-tech/AuctionNet` commit
`c20de631a40e24669d1d6a8e2acefd8f492f172d`, distributed under its Apache-2.0
license (included in `LICENSE.txt`). `PROVENANCE.json` lists original paths and
Git blob SHA-1 values. No upstream NOTICE file was present in the inspected tree.

The request generator is executed by `scripts/generate_auctionnet.py`. The two
strategy files are retained for interface comparison only: their current-slot
value access and bids do not satisfy this competition's observation/action
contract. Competition-specific code lives in `src/autobid`, not in this directory.

See `docs/auctionnet.md` for reproduction status, differences, data provenance,
and the optional full upstream checkout.
