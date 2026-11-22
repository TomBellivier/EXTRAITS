"""
extraits.process
================

Inference pipeline behind `extraits process`: runs the ensemble of YOLO-pose
models of <models>/pose/ over a set of insect images and writes one CSV row per
image with:

    - the image name and the insect order (one-hot),
    - every measurement of interest, in pixels and in millimetres,
    - a confidence value and a validity prediction for each measurement,
    - an overall pose-confidence value,
    - the detected scale (px/mm) and a confidence value for that scale.

Small, single-responsibility modules, so that each step can be read, tested and
swapped independently:

    config.py          -> all tunable parameters
    definitions.py     -> keypoints and measurements, from kp_infos.yaml
    measurements.py    -> turn keypoints into pixel measurements
    pose_inference.py  -> run the pose ensemble, mean + std of the keypoints
    tta.py             -> test-time augmentation (for the TTA signal)
    confidence.py      -> ALL confidence computations live here
    scale.py           -> scale bar, then ruler fallback (detectors in extraits.scale)
    insect_group.py    -> insect order of every image
    pipeline.py        -> the per-image pipeline (in-memory image)
    image_source.py    -> local folder OR Hugging Face dataset source
    parallel.py        -> bounded, multi-thread, as-completed map
    worker.py          -> per-thread models, on the device handed out
    hardware.py        -> sizes the run to the machine (GPU/CPU, memory)
    csv_writer.py      -> assemble and write the output CSV
    run.py             -> the `extraits process` command
    analyze.py         -> the `extraits analyze` command (figures and summary)
"""
