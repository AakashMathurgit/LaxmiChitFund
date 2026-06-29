# Deploying LCF to Azure Container Apps

LCF runs as an **always-on background worker** (not a website). The container's
start command is `python lcf.py schedule`, which drives all 5 flows on their
cadences. Secrets are supplied at runtime as **environment variables** — never
baked into the image or committed to git.

## What runs in the container
`lcf.py schedule` loops forever and, per cadence:
- `us-intraday` — every 4 min, US market hours only
- `us-daily`, `in-daily` — daily
- `us-predict` — weekly
- `advise` — monthly

Every BUY/SELL fans out to all available brokers (Vested paper + Alpaca paper)
and a Pushover alert. An hourly **heartbeat** ping confirms it's alive.

## Environment variables (set these as Container App secrets)

| Variable | Purpose | Required |
|---|---|---|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI base URL (`https://….openai.azure.com/openai/v1`) | yes |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI key | yes |
| `ALPACA_ENDPOINT` | `https://paper-api.alpaca.markets/v2` | yes (for trading) |
| `ALPACA_KEY_ID` | Alpaca paper key id | yes (for trading) |
| `ALPACA_SECRET_KEY` | Alpaca paper secret | yes (for trading) |
| `PUSHOVER_TOKEN` | Pushover app token | optional (enables alerts + heartbeat) |
| `PUSHOVER_USER` | Pushover user key | optional |

Locally these fall back to `credentials.yaml` (git-ignored). In the cloud, set
them as Container App **secrets** and reference them as env vars.

## Azure Portal steps

1. **Resource group** → Create → `lcf-rg`, region East US.
2. **Container Apps** → Create:
   - App name `lcf`, resource group `lcf-rg`.
   - **Container Apps Environment** → Create new (accept defaults).
3. **Deployment source = GitHub**:
   - Authorize GitHub, repo `mathuraakash_microsoft/LCF`, branch `main`.
   - **Dockerfile path:** `Dockerfile` (it's at the repo root), **build context:** `/` (repo root).
   - Azure generates a GitHub Action that builds + deploys on every push.
4. **Ingress = Disabled** (background worker, no web port).
5. **Secrets / Environment variables** → add the variables in the table above
   (store keys as *secrets*, reference them from env vars).
6. **Scaling** → Min replicas = **1**, Max replicas = **1** (always-on, single).
7. **Review + Create** → watch the build in the repo's **Actions** tab.

## Verify it's running

- **Logs:** Container App → Log stream → you'll see the scheduler ticking and
  flow output, same as local.
- **Heartbeat:** if Pushover keys are set, an hourly "LCF alive" push arrives.
- **Trades:** orders show up in your **Alpaca paper dashboard**; BUY/SELL alerts
  arrive via Pushover.
- **Status:** Container App → Revisions/Replicas shows the replica "Running."

## Updating later
`git push` to `lcf-cloud-deploy` → the GitHub Action rebuilds the image and
Container Apps rolls out the new revision automatically.

## Run locally (no container)
```bash
python lcf.py list                          # list flows
python lcf.py run us-intraday --top 8        # one flow
python lcf.py run all                        # one sweep of everything
python lcf.py schedule --max-ticks 1         # bounded scheduler (testing)
python lcf.py schedule                       # always-on (what the container runs)
```

## Cost note
A single always-on Container App replica is roughly **$15–40/mo**. Azure OpenAI
and Alpaca paper cost nothing extra to call. Scale to 0 / delete the app to stop
billing.
