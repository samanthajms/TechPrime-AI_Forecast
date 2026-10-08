# Render security configuration guide

Covers the two Render services of TechPrime AI and the pieces around them (GitHub, Supabase):

| Service | Render type | Repo | What it does |
|---|---|---|---|
| **techprime-ai** | Web Service (Docker: PHP 8.2 + Apache) | `TechPrime-AI` | The app: storefront, staff pages, `forecast_api.php` proxy |
| **techprime-forecast** | Web Service (Python, gunicorn) | `TechPrime-AI_Forecast` | XGBoost forecast API |

Steps marked **[you]** are done in a dashboard (Render, GitHub, Supabase) - nothing here can be applied from the repo alone. Render's UI changes over time; where a menu name may have moved, the intent is described too.

---

## 1. How the pieces trust each other

```
Browser ──(session cookie)──▶ techprime-ai (PHP)  ──(X-API-Key)──▶ techprime-forecast (Flask)
                                   │                                    ▲
                                   └──▶ Supabase Postgres               │ model files come from git
GitHub Actions (monthly) ──(read-only DB user)──▶ Supabase ──▶ PR ──merge──▶ Render redeploy
```

- The **browser never talks to the forecast service** and never sees its key. It calls `forecast_api.php`, which checks the login session, the role and the inputs, then calls Flask with the key.
- The forecast service trusts only holders of `FORECAST_API_KEY`. The maintenance endpoint `/api/reload` needs a second key, `FORECAST_ADMIN_KEY`.
- The monthly retrain runs in **GitHub Actions**, not on Render. Render's disk is ephemeral, so anything written there is lost on redeploy; models therefore travel through git (PR → merge → redeploy).

---

## 2. Secrets and environment variables

### 2.1 Generate strong keys (once, on your own machine)
```
python -c "import secrets; print(secrets.token_urlsafe(48))"
```
Run it **twice** - one value for `FORECAST_API_KEY`, a different one for `FORECAST_ADMIN_KEY`. Never reuse a key, never paste one into chat, a ticket or a commit.

### 2.2 Set them in Render **[you]**
Dashboard → each service → **Environment**.

| Variable | techprime-forecast | techprime-ai | Notes |
|---|:-:|:-:|---|
| `FORECAST_API_KEY` | ✅ | ✅ | **Same value on both.** `render.yaml` marks it `sync: false` so it is never in git. |
| `FORECAST_ADMIN_KEY` | ✅ | - | Only needed to call `/api/reload`. Leave unset to disable reload. |
| `FORECAST_URL` | - | ✅ | `https://techprime-forecast.onrender.com` (public) or the private URL in §3. |
| `DB_HOST` `DB_PORT` `DB_NAME` `DB_USER` `DB_PASS` | - | ✅ | The app's DB user. The forecast service needs **no** database access. |
| `RATE_LIMIT_PER_MIN` | optional (120) | - | Requests per minute per client IP. |
| `AUTH_FAIL_LIMIT` | optional (10) | - | Bad keys allowed per 5 minutes before the IP gets 429. |
| `TRUSTED_PROXY_HOPS` | optional (1) | - | See §4.3. |
| `HISTORY_CUTOFF` | demo only | - | Leave **unset** in production. |

Tip: put `FORECAST_API_KEY` in a Render **Environment Group** linked to both services so the two copies cannot drift apart.

### 2.3 Rules
- The service **refuses to start** without `FORECAST_API_KEY` (fail closed).
- `.env` files are for local use only and are git-ignored; on Render the real environment variables are used.
- The app's database password is the most sensitive secret: the forecast service and the retrain job must **not** use it (see §6).

### 2.4 Rotating a key **[you]**
Rotate every 90 days, immediately if a key may have leaked, and when someone with dashboard access leaves.
1. Generate a new value.
2. Update it on **techprime-forecast** and **techprime-ai** (Environment Group = one edit) and save - Render restarts both.
3. Forecast pages show "Forecast unavailable" for the minute both services restart; that is expected.
4. Check §10.

---

## 3. Network exposure

### 3.1 Recommended: make the forecast service private **[you]**
Only the PHP app needs to reach Flask, so Flask should not be on the public internet.
1. Render → **New → Private Service**, same repo/branch, same region and **same workspace** as the PHP service, with the commands from `render.yaml`.
2. In **techprime-ai** set `FORECAST_URL=http://<private-service-name>:10000` (Render shows the exact internal address on the service page).
3. `forecast_api.php` accepts plain `http://` **only** for single-label internal hosts and localhost; any public hostname must be `https://`.
4. Delete the old public web service once §10 passes.

Private services need a paid instance type. Do not move Flask to a private service until the PHP app is in the same region and workspace, or it won't resolve.

### 3.2 If it must stay public (free plan)
Defence is then the key + the limits already in the code:
- 32+ random bytes in `FORECAST_API_KEY` (§2.1) - not guessable.
- 10 bad keys / 5 min → HTTP 429 lockout per client IP; 120 requests/min per IP; `/api/reload` 5/min.
- `/health` is open on purpose (Render health checks) and returns only `{"status":"ok"}`.
- Check whether your Render plan offers **inbound IP rules** for the service; if it does, allow only the PHP service's outbound IPs.
- Do **not** add CORS. The service is server-to-server; browsers must not call it.

### 3.3 TLS **[you]**
Render terminates TLS and redirects HTTP → HTTPS on `onrender.com` and custom domains. Use a custom domain with a Render-managed certificate for the app. The forecast service additionally sends `Strict-Transport-Security`.

---

## 4. techprime-forecast (Flask) settings

### 4.1 What is already enforced in code
| Control | Where |
|---|---|
| API key on every route but `/health`, constant-time compare | `forecast_service.py` |
| Separate admin key for `/api/reload`, disabled when unset | `forecast_service.py` |
| Lockout after repeated bad keys, per-IP rate limit, stricter reload limit | `forecast_service.py` |
| `horizon` must be 1, 2 or 3 (HTTP 400 otherwise); `category` ≤ 80 chars, `top` clamped | service + `inventory_forecasting.check_horizon` |
| Generic error bodies (details only in server logs), JSON 404/405/413 | `forecast_service.py` |
| No request bodies accepted (1 KB cap) | `MAX_CONTENT_LENGTH` |
| Response headers: `no-store`, `nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, CSP `default-src 'none'`, HSTS | `forecast_service.py` |
| Models are XGBoost **JSON** (no pickle → loading a model cannot execute code) | `inventory_forecasting.py` |
| Runtime-only dependencies (no DB driver / Excel reader on the web service) | `requirements.txt` |
| gunicorn request-line / header size limits | `render.yaml` |

### 4.2 Render settings **[you]**
- **Auto-Deploy: On**, branch = the branch you merge retrain PRs into (`render.yaml` has `autoDeploy: true`).
- **Health Check Path:** `/health`.
- **Instance type:** the free plan sleeps after inactivity (first request ~50 s; the PHP proxy allows 75 s) and shares limited CPU. Use **Starter or higher** for production.
- **Workers:** keep `--workers 1 --threads 4`. The rate limiter and `/api/reload` are per process; with several workers a lockout or reload would apply to only one of them. Move limits to Redis before adding workers.
- Do not enable any "shell"/SSH access you don't need.

### 4.3 Client IP and `TRUSTED_PROXY_HOPS`
Limits are keyed on the client IP. Behind Render's load balancer that comes from `X-Forwarded-For`; the service trusts exactly `TRUSTED_PROXY_HOPS` proxy hop(s) (default 1) so callers cannot spoof the header. If you later put another proxy (e.g. Cloudflare) in front, raise it to 2. On a private service the only caller is your PHP app, so all traffic shares one bucket - raise `RATE_LIMIT_PER_MIN` accordingly (e.g. 600).

---

## 5. techprime-ai (PHP + Apache) settings

### 5.1 Already in code
- `forecast_api.php`: login session + **15-minute idle timeout** (`checkSessionTimeout()`), role check per forecast type, `horizon` 1-3 only, `category` whitelisted pattern, HTTPS required (except private hosts), no redirects followed, upstream auth/500 errors never relayed to the browser, `Cache-Control: no-store`.
- Roles: demand → admin, inventory custodian, retail officer; revenue → admin, retail officer; categories/metrics → those three. Cashiers and clients get 403.
- Forecast cards render server data with `textContent` (no HTML injection from product names).

### 5.2 Recommended additions (not applied - they change the live app, so review first)
**Session cookies** - `Dockerfile`, after the `COPY`:
```dockerfile
RUN { echo 'session.cookie_secure=1'; echo 'session.cookie_httponly=1'; \
      echo 'session.cookie_samesite=Lax'; echo 'session.use_strict_mode=1'; \
      echo 'expose_php=0'; echo 'display_errors=0'; echo 'log_errors=1'; } \
    > /usr/local/etc/php/conf.d/zz-security.ini
```
**Apache headers** (`mod_headers` is already enabled) - `Dockerfile`:
```dockerfile
RUN { echo 'ServerTokens Prod'; echo 'ServerSignature Off'; \
      echo 'Header always set X-Content-Type-Options "nosniff"'; \
      echo 'Header always set X-Frame-Options "SAMEORIGIN"'; \
      echo 'Header always set Referrer-Policy "strict-origin-when-cross-origin"'; \
      echo 'Header always set Strict-Transport-Security "max-age=31536000"'; } \
    > /etc/apache2/conf-available/security-extra.conf && a2enconf security-extra
```
**Block non-web folders.** The image copies the whole `TechPrime-AI/` folder into the web root, so anything not blocked is downloadable. At minimum deny `database/`, `scripts/`, `ml/`, `vendor/` and any `*.sql`, `*.md`, `*.env*`, `composer.*` files, e.g. in `.htaccess`:
```apache
<FilesMatch "\.(sql|md|env|lock|json|csv|py|bat)$">
    Require all denied
</FilesMatch>
RedirectMatch 404 ^/(database|scripts|ml|vendor|\.git)(/|$)
```
Test after deploying: `curl -I https://<app>/database/migration_cashier_pos.sql` must not return 200.

**Environment:** `DB_*`, `PAYMONGO_*` and `FORECAST_*` only in Render's Environment, never in the repo. Rotate them if they ever appeared in git history.

---

## 6. Database (Supabase) - read-only user for the retrain job **[you]**

The monthly job only needs to read sales. Give it its own user so a leaked CI secret cannot change or read anything else. Run in the Supabase SQL editor (this is a schema/permission change on the shared database - review it first):

```sql
-- 1) The role (use a long random password; store it ONLY as the GitHub secret DB_PASS)
CREATE ROLE forecast_ro LOGIN PASSWORD '<generated>';
ALTER ROLE forecast_ro SET default_transaction_read_only = on;
ALTER ROLE forecast_ro CONNECTION LIMIT 2;

-- 2) Table access: only what the job reads
GRANT USAGE ON SCHEMA public TO forecast_ro;
GRANT SELECT ON orders, order_items, products, pos_sales, pos_sale_items TO forecast_ro;

-- 3) These tables have row-level security ON with no policies, so a non-owner sees 0 rows.
--    Allow this one role to read them (policies are per table):
CREATE POLICY forecast_ro_read ON orders          FOR SELECT TO forecast_ro USING (true);
CREATE POLICY forecast_ro_read ON order_items     FOR SELECT TO forecast_ro USING (true);
CREATE POLICY forecast_ro_read ON products        FOR SELECT TO forecast_ro USING (true);
CREATE POLICY forecast_ro_read ON pos_sales       FOR SELECT TO forecast_ro USING (true);
CREATE POLICY forecast_ro_read ON pos_sale_items  FOR SELECT TO forecast_ro USING (true);
```
Notes
- Skip step 3 for a table that does not have RLS enabled. Verify with `SELECT count(*) FROM orders;` while connected as `forecast_ro`.
- The job selects only `product`, `quantity`, `price`, `status`, `created_at` columns - no customer names, emails or addresses. If you want to enforce that, grant column-level `SELECT` instead of whole-table `SELECT`.
- Use the **pooler** host and keep `sslmode=require` (already in the scripts). Scripts open the connection `readonly`, so even a mistaken query cannot write.
- Rotate `forecast_ro`'s password every 90 days (`ALTER ROLE forecast_ro PASSWORD '...'` + update the GitHub secret).

---

## 7. GitHub **[you]**

| Setting | Why |
|---|---|
| Both repos **private** | Sales history, product data and model files are in the repo. |
| Branch protection on the deploy branch: require a PR + 1 review, block force pushes | Whoever can push the model/CSV files controls the forecast, and the deploy branch controls production. |
| Settings → Code security: enable **secret scanning + push protection**, **Dependabot alerts/updates** | Catches committed keys and vulnerable packages (Flask, gunicorn, xgboost, pandas). |
| Settings → Secrets and variables → Actions: `DB_HOST DB_PORT DB_NAME DB_USER DB_PASS` | Only the read-only `forecast_ro` credentials from §6. |
| Settings → Actions → General: workflow permissions "Read and write", tick **Allow GitHub Actions to create and approve pull requests**; restrict allowed actions to GitHub-owned | The monthly workflow pushes a `retrain/*` branch and opens the PR. |
| Optional: pin `actions/checkout` and `actions/setup-python` to commit SHAs | Protects against a tag being re-pointed. |

The workflow never writes to the deploy branch directly: a person reviews the gate report in the PR and merges. A failed regression gate fails the run (GitHub emails the repo admins) and no PR is created.

---

## 8. Monitoring **[you]**

- Render → service → **Logs**. The forecast service logs:
  - `rejected request: bad API key from <ip>` - repeated lines = someone guessing; the lockout (429) will follow.
  - `rejected reload: bad admin key` - investigate immediately.
  - `models reloaded by <ip>` - should only follow your own deploys.
  - `forecast error` tracebacks - bugs, not attacks.
- Render → **Notifications**: turn on deploy-failed and health-check-failed alerts for both services (email/Slack).
- Keep the app's own Activity Logs (Admin → Activity Logs) as the audit trail for who viewed forecasts (`view_forecast`).
- After every retrain PR, read `GET /api/metrics` (via the app) - `data.last_month` should be the month you expect.

---

## 9. Responding to an incident

| Situation | Do this |
|---|---|
| `FORECAST_API_KEY` leaked | Rotate (§2.4). Check logs for 200-status requests from unknown IPs since the leak. |
| `FORECAST_ADMIN_KEY` leaked | Rotate it, or delete the variable to disable `/api/reload` entirely. Reload only re-reads files already in the deploy, so impact is limited. |
| DB credential leaked | Rotate in Supabase first, then the Render / GitHub secret. For `forecast_ro` also check its connection log. |
| Bad forecast after a retrain | Revert the retrain PR's merge commit → Render redeploys the previous models (git history is the backup). |
| Suspicious orders skewing data | Cancel/void them; only `to_ship / to_receive / delivered / completed` online orders and completed POS sales count, and the next retrain's regression gate blocks a model that got worse. |

---

## 10. Verification checklist (run after any change)

Replace `$F` with the forecast URL and `$K` with the API key (do not paste the key into shared channels).

```bash
# a) health is open, everything else is closed without the key
curl -s  $F/health                                   # {"status":"ok"}
curl -si $F/api/categories | head -1                 # HTTP/2 401
# b) 1-3 month window
curl -si -H "X-API-Key: $K" "$F/api/forecast/revenue?horizon=3" | head -1   # 200
curl -si -H "X-API-Key: $K" "$F/api/forecast/revenue?horizon=4" | head -1   # 400
# c) reload needs the admin key
curl -si -X POST -H "X-API-Key: $K" $F/api/reload | head -1                 # 401 (or 403 if disabled)
# d) security headers present
curl -sI -H "X-API-Key: $K" $F/api/categories | grep -iE "nosniff|no-store|strict-transport|frame-options"
# e) lockout (use a throwaway network; you will be blocked for 5 minutes)
for i in $(seq 1 12); do curl -s -o /dev/null -w "%{http_code} " -H "X-API-Key: wrong" $F/api/categories; done   # 401 ... then 429
```
In the app (log in as each role):
- Admin / Retail Officer → Forecast shows the three-option window (1/2/3 months) and loads products + chart.
- Inventory Custodian → `forecast_api.php?type=revenue` returns 403; cashier and client get 403 on every type.
- `forecast_api.php?type=revenue&horizon=4` → 400 `horizon must be 1, 2 or 3 months`.
- Idle for 15+ minutes → the next forecast request gets 401 `session_expired`.
- If you applied §5.2: `curl -I https://<app>/database/migration_cashier_pos.sql` is not 200.

Local automated checks (no network, no database): `python -m unittest discover -s tests -v`.

---

## 11. Known limits (be upfront about these in the defense)

- Rate limiting is in-memory per process (fine for one worker); use Redis for multi-worker.
- The free Render plan sleeps and restarts, so lockout counters reset on every restart.
- `diagnose_db.py` and `rebuild_features_from_db.py` need DB access; run them only with the read-only user.
- `.htaccess` in the forecast repo is an Apache file and does nothing under gunicorn; it is harmless but gives no protection (Flask serves no static files). It was left in place rather than deleted.
- The 80/20 test window moves every month, so the regression gate compares accuracy on different months; treat it as a safety net, not a benchmark.
