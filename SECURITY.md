# Security

This preview is intended for local research. Hosted and multi-user deployments
are not supported. There is no long-term support or guaranteed response timeline;
security fixes target the current preview.

Report suspected vulnerabilities through
[GitHub private vulnerability reporting](https://github.com/tubby-labs-research/tubby-quant/security/advisories/new).
This is the intended channel after publication; maintainers must enable and
verify private reporting as part of publication. If the private report form is
unavailable, do not substitute a public issue: ask a maintainer to enable a
private reporting channel without disclosing vulnerability details.

Include the affected version or commit, a synthetic reproduction, impact and
relevant logs with secrets and personal data removed. Never post secrets,
launch tokens or vulnerability details in public issues, pull requests or
discussions. Keep reports private while maintainers investigate.

The workbench binds to loopback, uses a per-launch token and checks request hosts
and origins. Keep its launch URL private. Do not expose the service through port
forwarding or a public proxy. Local access controls are not a multi-tenant design.

Keep agent connection files and credentials outside Git. Configure only trusted
agent endpoints: a generation request can transmit source excerpts, prior ideas,
results and the research question. Review the provider's data handling before
enabling calls. A health check must not invoke generation.

Install from the project source or a complete release and retain runtime hash verification.
Missing or changed artifacts should fail, not trigger a manual hash bypass.
Review dependency and data-provider changes separately. These precautions and
passing tests are not a complete security audit.
