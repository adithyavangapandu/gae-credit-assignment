# Website deployment and updates

The public research website is available at:

<https://gae-credit-assignment.adi05.chatgpt.site>

It is a public static deployment of the Quarto project in `research_site/`.
Website source, demonstration data, generated figures, and rendered pages remain
excluded from this GitHub repository, as required. The hosting project identity
is stored in `/.openai/hosting.json`; it is deployment metadata, not website
source.

## Stable URL contract

The hosting project ID and slug identify the existing site. Every update must:

1. read `/.openai/hosting.json`;
2. reuse its exact `project_id`;
3. save a new version to that project; and
4. deploy that saved version without creating another site or changing the slug.

Following those rules replaces the production content while preserving
`https://gae-credit-assignment.adi05.chatgpt.site`. The GitHub README therefore
does not need a new link after normal website updates. The link changes only if
the hosting project is deleted, its slug is deliberately changed, or a custom
domain is added and used instead.

## Prepare an update

Make the intended edits under `research_site/`, then validate and render them:

```bash
cd research_site
../.venv/bin/python -m unittest discover -s tests -v
quarto render
```

The render must finish with `Output created: _site/index.html`. Inspect the
rendered site locally if needed:

```bash
quarto preview
```

Do not add `research_site/`, `_site/`, data, or figures to the main Git
repository. They are intentionally ignored, and the repository hygiene check
will reject them if they become tracked.

## Publish at the existing URL

Ask Codex to deploy the current `research_site/` and explicitly say to reuse the
project in `/.openai/hosting.json`. A suitable request is:

> Deploy the current `research_site/` to the existing Sites project in
> `.openai/hosting.json`. Keep the current public URL and access settings. Do not
> modify the website source.

The deployment flow should copy the website into an isolated staging directory,
run its tests, render it with Quarto, and map Quarto's `_site/` output to the
host-supported `dist/` deployment directory. It then pushes the staged source to
the existing hosting source repository, saves a new version, deploys that exact
version, and waits for a successful production status. Temporary write
credentials must never be saved to disk or committed.

After deployment, verify at least these public routes:

```bash
curl -fsSIL https://gae-credit-assignment.adi05.chatgpt.site/
curl -fsSIL https://gae-credit-assignment.adi05.chatgpt.site/methods
curl -fsSIL https://gae-credit-assignment.adi05.chatgpt.site/interactive
```

Each route must ultimately return HTTP 200. Also rerun the main repository
hygiene check before pushing documentation changes:

```bash
cd ..
uv run python scripts/check_repository_hygiene.py
git status --short
```

## Rollback

The host retains saved versions. To roll back, select the last known-good saved
version for the same project and deploy it again. Do not create a replacement
project: redeploying an earlier version preserves the public URL.
