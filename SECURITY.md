# Security policy

## Supported versions

Security fixes are applied to the latest revision of the default branch. This
project does not currently maintain older release branches.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use GitHub's private
security advisory feature for this repository and include:

- affected revision;
- reproduction steps;
- expected and observed impact;
- suggested mitigation, if known.

## Deployment warning

Diffusers Engine Q21 is a local research tool. The API has no authentication,
authorization, rate limiting, tenant isolation, or content moderation.

- Keep the API and frontend bound to `127.0.0.1` unless an authenticated
  reverse proxy and network access controls are configured.
- Do not expose port 8000 or the `/outputs` route directly to the internet.
- Uploaded images and generated outputs may contain sensitive information.
- Treat model files and generated images as untrusted data when obtained from
  third parties.
- Review dependency and model revisions before upgrading them.

The default configuration binds to localhost and restricts CORS to the local
frontend origins.
