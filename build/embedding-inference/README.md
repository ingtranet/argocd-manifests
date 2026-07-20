# embedding-inference container images

Source for `zot.ingtra.net/library/embedding-inference:<tag>`.

Each subdirectory builds one tag of the same repository. Images are built and
pushed manually — Argo CD does not watch this tree.

Sibling of `build/image-inference` (image *generation* servers); kept separate
because embedding servers share no base image or dependency set with those.

## Tag convention

`<model-name>-<short-git-sha>` — e.g. `pplx-embed-a1b2c3d`.

## Build

```bash
cd build/embedding-inference/pplx-embed
./build.sh
```
