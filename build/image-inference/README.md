# image-inference container images

Source for `zot.ingtra.net/library/image-inference:<tag>`.

Each subdirectory builds one tag of the same repository. Image is built on the
Jetson Orin node (aarch64) and pushed manually — Argo CD does not watch this
tree.

## Tag convention

`<model-name>-<short-git-sha>` — e.g. `nitro-e-dist-a1b2c3d`.

## Build

```bash
cd build/image-inference/nitro-e
docker build -t zot.ingtra.net/library/image-inference:nitro-e-dist-$(git rev-parse --short HEAD) .
docker push zot.ingtra.net/library/image-inference:nitro-e-dist-$(git rev-parse --short HEAD)
```
