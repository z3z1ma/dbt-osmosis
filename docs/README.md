# Website

This website is built using [Docusaurus 3](https://docusaurus.io/). The source for the published docs lives under `docs/docs/`, with site configuration in `docs/docusaurus.config.js` and navigation in `docs/sidebars.js`.

## Install dependencies

```bash
npm install
```

## Local development

```bash
npm run start
```

This starts the local docs server with live reload.

## Build

```bash
npm run build
```

This generates static output in `build/`.

## Serve the built site

```bash
npm run serve
```

## Deployment

Using SSH:

```bash
USE_SSH=true npm run deploy
```

Not using SSH:

```bash
GIT_USER=<your GitHub username> npm run deploy
```

If GitHub Pages is the hosting target, deployment pushes the generated site to the `gh-pages` branch.
