Status: active
Created: 2026-11-16
Updated: 2026-11-16
Parent: 2026-11-16-dependency-queue.md

# Coherent docs dependency refresh
Scope: inspect #430 #432 #434 #435 #437, current docs manifests/lock and upstream releases. Refresh as coherent compatible update in isolated worktree; preserve Dependabot attribution. Do not modify roadmap or Python files. No release/deployment.
Acceptance: npm clean install, build, dependency audit comparison, inspected lock diff; draft PR only until independent review and exact-head CI. Return evidence, exact commits, blockers and recommended original PR dispositions. Parent owns merge/closure.
References: AGENTS.md, docs/package.json, .github/workflows, parent ticket.
Blockers: none for investigation.

## Journal and evidence
- 2026-11-16: Started isolated maint/docs-dependency-queue worktree at 7279fec9. Inspected all five live patches. Individual Docusaurus updates introduce mixed versions; replace them with one aligned 3.10.2 tree. #437's body-parser and shell-quote update is included through current compatible releases (1.20.9 and 1.12.0).
- Primary sources: https://github.com/meta/docusaurus/releases/tag/v3.10.2 (patch fixes including BaseUrlIssueBanner security issue and gray-matter replacement); https://github.com/expressjs/body-parser/releases/tag/1.20.6 (CVE-2026-12590); https://github.com/ljharb/shell-quote/blob/main/CHANGELOG.md (linear-time parse fix and subsequent compatible fixes).
- Baseline npm audit on unchanged main lock: 42 reported dependency findings (7 low, 4 moderate, 14 high, 17 critical). These include propagated findings, not 42 independent vulnerabilities. Raw local evidence: /tmp/osmosis-docs-audit-before.json.
- Coherent patch clean installation and production build passed on Node 26.7.0/npm 11.19.0. Baseline npm install refreshed compatible transitive dependencies of Docusaurus; targeted updates subsequently clear obsolete vulnerable brace-expansion, browserslist, compression, fast-uri, follow-redirects, launch-editor and proxy-addr resolutions. No audit force/downgrade used.
- Existing qs and serialize-javascript overrides are themselves stale: update to 6.17.1 and 7.1.4, metadata supports Node 20. Retain uuid 11.1.1 and webpack-dev-server 5.2.6 to avoid expanding the Node floor.
- Tinypool is constrained by Docusaurus core to ^1.0.2; no fixed 1.x release exists. Primary v2.0.0 release notes identify dropping Node18 and availableParallelism as changes; v2.1.2 fixes worker filename prototype pollution. Registry metadata requires ^20.0.0 || >=22.0.0. Docusaurus uses constructor worker options, run and destroy; validating an exact 2.1.2 override with forced two-worker SSG to exercise the real API instead of the single-thread shortcut.
- Sources: https://github.com/tinylibs/tinypool/releases/tag/v2.0.0 and https://github.com/tinylibs/tinypool/releases/tag/v2.1.2.

## Review
Pending independent review and exact-head CI. Original PRs must remain open until replacement merge.

## Retrospective
A stale security override can become the vulnerability source. Re-audit actual current locks, retain explicit Node floor, and force the SSG pool when validating tinypool: small default builds can bypass it entirely. npm audit counts include vulnerable ancestor propagation and must not be described as unique advisories.
- Final clean npm ci passed; forced two-worker production build passed on Node26.7.0. Local dev server compiled and returned HTTP200 at /dbt-osmosis/. Final audit has 5 low findings, all propagated from deepmerge; zero moderate/high/critical. No fixed deepmerge version exists; follow-up .10x/tickets/2026-11-16-docs-deepmerge-advisory.md owns it. Node20/24 CI remains mandatory before merge.
- Superseded the old override ADR with freshly justified versions while retaining original history and support policy. No workflow/version/roadmap edits.
- Verification correction: earlier DOCUSAURUS_SSG_WORKER_THREADS build only proves ordinary build. Source inspection shows actual variable DOCUSAURUS_SSG_WORKER_THREAD_COUNT and future.faster.ssgWorkerThreads required. Temporary uncommitted config enables that flag plus required removeLegacyPostBuildHeadAttribute; testing the real pool separately. This does not enable experimental flags in committed site config.
- OSV independently found webpack-dev-middleware 7.4.5 despite npm audit omitting it. Compatible update resolves 7.4.6; advisory metadata explicitly lists patched 7.4.6 despite stale prose saying no 7.x backport. Final scan will determine residuals.
- Actual pooled test now PASSED: temporary config {future:{v4:{removeLegacyPostBuildHeadAttribute:true},faster:{ssgWorkerThreads:true}} with DOCUSAURUS_SSG_WORKER_THREAD_COUNT=2 and --config docusaurus.pool-test.js. Clean npm ci and this build passed after middleware 7.4.6 update. Temporary config removed; committed config unchanged. The pool exercises private worker state and workerData tuple, not only constructor/run/destroy.
- Final OSV scan: only deepmerge GHSA-gcqx-w48m-w2h4. Final npm audit: five low propagated findings, no moderate/high/critical. Registry latest deepmerge 4.3.1 remains vulnerable. Final raw local evidence /tmp/osmosis-docs-osv-final.json, /tmp/osmosis-docs-audit-final.json, /tmp/osmosis-docs-ci-final2.log, /tmp/osmosis-docs-pool-final.log.
- Tinypool metadata excludes odd Node21 (^20 || >=22); repository CI-supported Node20/24 remain supported. No claim of testing Node21 is made.

## Independent review (dependency_review, 2026-11-16)
Target: effa0ce518fc6f43173e40a9803f6daefc28f721 against 7279fec.
Verdict: pass, subject to exact-head Node20/24 CI before merge.
Findings: no blocking defect. Four Docusaurus roots and transitive Docusaurus packages align at 3.10.2; docs source/config and workflows unchanged. Security overrides are bounded and justified in superseding ADR; lock refresh includes compatible transitive updates, not just five original PR payloads. Source inspection confirms Tinypool 2.1.2 retains Docusaurus constructor/run/destroy and private worker state/workerData tuple API. Initial pooled-build evidence was corrected and superseded by actual flag-enabled two-thread build evidence; temporary config is absent from commit. Dependabot attribution retained.
Residual risk: local evidence uses Node26, so Node20/24 remote builds remain mandatory. Tinypool explicitly excludes Node21 in package metadata; tested project CI versions remain supported. Low-severity deepmerge advisory has durable blocked owner; scans are not a claim of zero vulnerabilities or proven exploitability. Executor verification was reviewed, not repeated.
