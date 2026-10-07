# Deduplication and settlement

`SahiDetectionEngine` produces raw detections. `CoordinateMapper` places their
boxes in global pixels and millimetres, using encoder travel for global Y.
Horizontal scale comes directly from `inference.layout.mm_per_pixel_x`.
The supplied calibration is 0.1130 mm/pixel horizontally and 0.1101 mm/pixel
vertically. X is not inferred from fabric width. Physical camera slots come
from `camera_position_indices`, independently of the selected camera list.
Camera 3 remains slot 2 even when cameras 1 and 2 are inactive.

Configure the full `fabric_width_mm`, its `fabric_left_edge_mm` measured from
camera 1's left view boundary, and both stencil widths. The current stencil
model assumes each stencil masks that much fabric at its respective edge:

```text
camera step = frame width * (1 - overlap) * horizontal GSD
machine X = physical slot * camera step + local pixel X * horizontal GSD
active left = fabric left + left stencil
active right = fabric left + full fabric width - right stencil
active X = machine X - active left
```

`top_right` instead measures from the active right edge. Boxes entirely outside
the active interval return `None`; callers must exclude these before merging.
Boundary-crossing mapped boxes are clipped and marked `clipped_to_active_area`;
the original image-local box remains unchanged. Reported mapped sizes are the
visible inspected portion, not necessarily the entire physical defect.

`inspection_coverage(frame_width_px)` reports each selected camera's active
interval or `None`. Use the union of these intervals to determine coverage,
not their summed lengths. Unseen fabric remains uninspected. Cameras are
assumed to share frame dimensions/GSD and uniform spacing/overlap.

The YAML fabric width is intentionally unset until a real job width is given.
Confirm the configured zero placement and 50 mm masks before operation. The
active camera list is unchanged; select cameras 3/4 there for the MV1 setup.
Automatic edge tracking and per-camera extrinsic calibration remain open.
If fabric shifts, configuration alone cannot detect that movement. Rebuild the
mapper at job setup when placement changes; do not change coordinates mid-roll
while deduplication retains open groups.
The future audit writer must supply each raw `detection_id` before merging.

`DeduplicationEngine.process_trigger(trigger_id, row_top_px,
detections_by_camera)` accepts one complete camera row. Every configured camera
must have an entry, including an empty list when there are no detections.
`row_top_px` is encoder travel in millimetres divided by the configured
`mm_per_pixel_y`, in the same coordinate system as the mapped boxes.

Processing order:

1. Merge camera observations horizontally using SAHI NMM and configured gaps.
2. Merge the complete row vertically with still-open groups.
3. Release groups whose bottom edge is strictly before
   `row_top_px - vertical_gap_px`.

This uses the observed current row position, rather than the older script's
prediction of the next trigger position. Settlement may happen later, but does
not assume that future encoder travel equals a configured image overlap.
Forward travel is required. Repeated/backward trigger IDs or backward encoder
travel are rejected; a reset or counter wrap requires explicit lifecycle handling.
Bottom-origin coordinates require a known roll length and are not supported by
this streaming mapper.

The engine retains only open groups, not a history of settled results. Groups
contain contributing raw UUIDs and no final defect number. The caller must
retain released groups until persistence succeeds; the merger does not retry
database writes. `finish()` releases the remaining groups once at a real roll
end. Do not use it for a temporary capture pause.

The directional gap rules are adapted from the existing
`../inference/05_dedupe_nmm_final/dedup.py`. NMM uses SAHI's actual member-index
mapping to carry raw UUIDs, rather than guessing membership from containment
of output rectangles. The NumPy API is targeted by the SAHI requirement in
`requirements.txt`:
https://github.com/obss/sahi/blob/main/sahi/postprocess/combine.py

All merge thresholds are under `inference.deduplication` in `config.yaml`.
`ios_threshold` is retained as the existing setting name; it applies to the
configured `match_metric` (IOS by default). Different defect classes always
remain separate.

Verification is pending in a dependency-complete environment. No packages were
installed for this change. When environment setup is requested, use `uv venv`
and `uv pip install` with the selected environment, then run from this directory's
parent (`integrated_code`):

```text
python -m unittest tests.inference_pipeline.test_deduplication_engine
python -m unittest tests.inference_pipeline.test_coordinate_mapper
```

`main.py` creates `AppOrchestrator` and calls `run()`. Its constructor creates
the frame components, detector, mapper, MCU link, WebSocket hub, dashboard
mirror/coordinator, and FastAPI app. Component internals remain component-owned.
The dashboard receives its link and mirror, not the orchestrator; it never
starts or stops the application. `run()` invokes Uvicorn with the existing app,
whose lifespan calls the orchestrator's `start()` and `stop()`.

`AppOrchestrator` loads
the model before starting the MCU, and detects/maps each trigger bundle with
serialized model access. Shutdown waits for ongoing inference before releasing
the model. The callback currently logs mapped observation counts only; these
results are not persisted, retained as history, or published as final defects.

Deduplication is not wired yet: its contract requires persisted raw detection
UUIDs. Audit persistence and publish-after-commit remain subsequent tasks.
Do not use this intermediate stage for production defect accounting.
