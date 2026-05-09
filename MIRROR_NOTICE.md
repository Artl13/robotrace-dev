# Read-only mirror

This repository is a one-way sync of the **RoboTrace Python SDK**
from our internal monorepo. The source of truth is private, but
this mirror is updated automatically on every change to the SDK
source — including every PyPI release.

## What this is

- The full source for [`robotrace-dev`](https://pypi.org/project/robotrace-dev/) on PyPI
- Identical code to what gets shipped into the wheel
- The same `tests/` and `examples/` we run internally

## Where to send things

| Question / artifact   | Where it goes                                                                                       |
| --------------------- | --------------------------------------------------------------------------------------------------- |
| Bug report            | [Open an issue](https://github.com/Artl13/robotrace-dev/issues/new)                                 |
| Feature request       | Same — open an issue and tag it `enhancement`                                                       |
| Pull request          | Welcome — we'll review here, then re-apply the patch in the private monorepo                        |
| Security disclosure   | Email `security@robotrace.dev`. Don't open a public issue.                                          |
| General contact       | [hello@robotrace.dev](mailto:hello@robotrace.dev) or [robotrace.dev](https://robotrace.dev)         |

## What's *not* here

- The web app, ingest API, portal, and admin code (private)
- Internal design docs, roadmap items, or invitee data
- The marketing site source

We may open-source pieces of the platform later — for now the SDK
is the only public surface.

## License

MIT — see [LICENSE](./LICENSE).
