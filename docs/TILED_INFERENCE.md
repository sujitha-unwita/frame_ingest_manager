# Explicit tiled inference

The class remains named `SahiDetectionEngine` because SAHI still owns tile-box
postprocessing. It no longer uses `AutoDetectionModel` or `get_sliced_prediction`.

1. The orchestrator creates and injects `TileGenerator` into the detector.
2. `start()` loads `ultralytics.YOLO` and performs a warmup prediction with the
   configured batch, channel count and square tile dimensions. TensorRT loads
   at prediction time, so incompatible engines fail before capture starts.
3. Read each frame as grayscale (1 channel) or convert OpenCV BGR to RGB
   (3 channels), according to `inference.input_channels`.
4. Generate row-major overlapping tiles with IDs, offsets and valid dimensions.
   Anchor the final tile to each edge; pad frames smaller than one tile.
5. Convert tiles to float BCHW tensors, normalized by the configured divisor.
   Pad the last batch to its configured size; discard dummy predictions.
6. Predict directly with Ultralytics. Clip boxes to valid, unpadded tile pixels
   and add X/Y tile offsets to obtain whole-frame coordinates.
7. Use SAHI `NMMPostprocess` or `NMSPostprocess` on those whole-frame boxes.
8. Return the existing `Detection` contract. Mapping, raw auditing, cross-camera
   and cross-trigger deduplication, grading and final auditing remain unchanged.

## Configuration to confirm on the Jetson

- `model_path`: the actual engine or model file.
- `tile_size_px`, `tile_batch_size`, `input_channels`: must match the engine's
  export shape/profile. Defaults were not guessed from another engine's name.
- `device_preference`: a TensorRT engine requires a working CUDA runtime.
- `tile_overlap_ratio`, `padding_value`, `normalization_divisor`: preprocessing.
- `confidence_threshold`, `prediction_iou_threshold`,
  `prediction_max_detections`, `prediction_class_agnostic`,
  `prediction_verbose`: Ultralytics settings.
- `sahi_postprocess_type`, `sahi_match_metric`, `sahi_match_threshold`,
  `postprocess_class_agnostic`: per-frame tile merging. This threshold is now
  separate from cross-camera/trigger deduplication settings.

Removed obsolete SAHI slicing settings: `perform_standard_prediction`,
`force_postprocess_type`, `sahi_verbose_level`. There is no extra whole-image
prediction pass. Update deployed YAML together with Python code.

Tile IDs exist internally, but the chosen SAHI object postprocessor does not
retain exact merged membership. Returned `tile_ids` remain null, not guessed.
No reference-project hardcoded edge filters or extra gap merging were copied.

This change does not repair a CPU-only PyTorch installation. The Jetson's
PyTorch, torchvision, TensorRT and CUDA builds still need to be compatible.
