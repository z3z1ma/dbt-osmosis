Status: blocked
Created: 2026-11-16
Updated: 2026-11-16

# Upstream deepmerge prototype poisoning follow-up
Scope: track GHSA-gcqx-w48m-w2h4 in Docusaurus dependency graph. Current npm latest is 4.3.1 and vulnerable; audit suggests downgrading Docusaurus, which is not an acceptable remediation. No fixed compatible deepmerge version observed on 2026-11-16.
Acceptance: recheck upstream fixed release and Docusaurus compatibility; update lock once supported, then clean install/build/audit and independent review.
Evidence: npm audit and npm view deepmerge version. Primary advisory https://github.com/advisories/GHSA-gcqx-w48m-w2h4. This is a low-severity docs-toolchain finding; no exploitability claim is made for this repository. Do not claim clean audit.
Blocker: upstream fixed version unavailable. Parent: 2026-11-16-dependency-docs.md.
