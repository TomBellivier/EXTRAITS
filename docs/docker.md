# Docker

Two images, built from `docker/`:

| Image | For | Contents |
|---|---|---|
| `ghcr.io/tombellivier/extraits:latest` | CPU | measuring and analysing (`process`, `analyze`, `download-models`, `train validity`) |
| `ghcr.io/tombellivier/extraits:cuda` | NVIDIA GPU | the same, plus training and fine-tuning |

The models are not in the images: before the first `process` (or `run`), the official
models are downloaded into the workspace mounted at `/workspace`, so they are fetched
once and kept between runs. The EasyOCR weights are in the images.

```bash
# measure the images of ./workspace/images_to_process
docker run --rm -v "$PWD/workspace:/workspace" --user "$(id -u):$(id -g)" \
    ghcr.io/tombellivier/extraits process -i /workspace/images_to_process -o /workspace/results.csv

# with an NVIDIA GPU (NVIDIA Container Toolkit installed)
docker run --rm --gpus all --shm-size 8g -v "$PWD/workspace:/workspace" \
    --user "$(id -u):$(id -g)" ghcr.io/tombellivier/extraits:cuda train pose --folds all
```

Every argument after the image name is an `extraits` command. `--user` makes the files
written to the workspace yours rather than root's.

With Docker Compose (`docker-compose.yml` at the root of the repository):

```bash
EXTRAITS_WORKSPACE=~/my-workspace docker compose run --rm extraits \
    process -i /workspace/images -o /workspace/results.csv
docker compose --profile gpu run --rm extraits-gpu train pose
```

Building them yourself:

```bash
docker build -f docker/Dockerfile -t extraits .
docker build -f docker/Dockerfile --build-arg EXTRAS=train -t extraits:train .   # + training
docker build -f docker/Dockerfile.cuda -t extraits:cuda .
```

## Licence

The images bundle Ultralytics YOLO and are distributed under the **AGPL-3.0**
(`/usr/share/doc/extraits/NOTICE` in the image); the models they download are AGPL-3.0
too. The EXTRAITS source code is MIT. See [MODEL_LICENSE.md](../MODEL_LICENSE.md) and
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
