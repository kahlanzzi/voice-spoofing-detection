# The Stage 18 web image (docs/api_design.md §2; docs/setup.md "Running with Docker Compose").
# Build from the repo root: docker compose build web. The context is limited by deploy/web.Dockerfile.dockerignore
# (BuildKit reads it instead of the root .dockerignore for this build only).
# Tags and digests are pinned together (index digests, from docker buildx imagetools inspect, 2026-11-01).

# Build stage: Node 26.13.0 as in web/.nvmrc. web/.npmrc keeps engine-strict, so another Node fails npm ci.
FROM node:26.13.0-alpine@sha256:c6ad9e9c1d80f6eb607c26b4b0a132ff7a2344fad1c78ea3baa2a7d4c8de6d94 AS build
WORKDIR /web
COPY web/package.json web/package-lock.json web/.npmrc ./
RUN npm ci
COPY web/ ./
RUN npm run build

# Final stage: the nginx of Stage 17 Step 8b, with the built app and the config only. No Node, no node_modules,
# no source. A dedicated root, so the image's default index.html and 50x.html are never served.
FROM nginx:1.30.6-alpine@sha256:0e52bd054126cdc040b1276c0af347aeb02a41b9cfe305d6094767ac30e97949
COPY --from=build /web/dist /usr/share/vsd-web
COPY deploy/nginx.conf /etc/nginx/nginx.conf
COPY deploy/nginx_security_headers.conf /etc/nginx/vsd_security_headers.conf
