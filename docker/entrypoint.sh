#!/bin/sh
# Entry point of the EXTRAITS images: `extraits <command> ...`.
# Before the first `process` or `run`, download the official models into the workspace
# volume, unless a models folder is given (EXTRAITS_MODELS, --models-dir, --pose-models).
set -e

case "$1" in
    process|run)
        models="${EXTRAITS_MODELS:-${EXTRAITS_WORKSPACE:-/workspace}/models}"
        case " $* " in
            *" --models-dir"*|*" --pose-models"*) ;;
            *)
                if [ ! -d "$models/pose" ]; then
                    echo "No models in $models: downloading the official EXTRAITS models."
                    extraits download-models --models-dir "$models"
                fi
                ;;
        esac
        ;;
esac

exec extraits "$@"
