# How our four-point defect grading works

The project uses the supplied poster policy, not a claim of ASTM compliance.
Policy values are in `config.yaml` under `inference.grading`, validated by
`FourPointGrading` in `config/models.py`.

This guide explains the implementation, not a proposed future design. The
score is an inspection penalty for one physical defect. It is different from
model confidence: confidence is retained but does not multiply the points.

## Code locations

All paths are relative to `integrated_code/`.

| File | Responsibility |
|---|---|
| `config.yaml` | Adjustable policy under `inference.grading`. |
| `config/models.py` | `FourPointGrading` validates the YAML rules. |
| `inference_pipeline/engines/coordinate_mapper.py` | Converts pixel coordinates to mm, filters stencils, and calculates inspection width. Does not assign points. |
| `inference_pipeline/engines/deduplication_engine.py` | Merges repeated observations and calls grading when a group settles. |
| `inference_pipeline/engines/four_point_grading_engine.py` | Calculates one settled group's score. |
| `app_orchestrator.py` | Creates these objects and passes graded groups to audit. |
| `audit_write/services/audit_writer.py` | Assigns final numbers, writes scores and links raw members. Does not calculate points. |

## Ownership and flow

1. `AppOrchestrator.__init__` creates `FourPointGradingEngine` and injects it
   into `DeduplicationEngine`.
2. Raw detections are mapped and persisted with their detection UUIDs.
3. `DeduplicationEngine.process_trigger` merges observations. Open groups
   have no score because subsequent observations may expand them.
4. On settlement, deduplication calls `FourPointGradingEngine.grade`.
   It returns a scored copy before deduplication commits its state.
5. The orchestrator queues these groups and calls
   `AuditWriter.write_final_defects`. AuditWriter rejects ungraded groups,
   then writes the score with the final row and links its raw members in
   the same transaction. Exact-membership retries reuse the existing row.

`DeduplicationEngine.finish` also grades remaining groups before clearing
them. It is for actual roll completion, not pause or process shutdown;
the roll-end command flow is still pending.

### Method-by-method flow

```text
AppOrchestrator.on_trigger()
    |
    +-- _infer_bundle()
    |      +-- detector.detect()                  Raw detections
    |      +-- coordinate_mapper.map_detection()  Physical coordinates/filtering
    |      +-- _audit_frame()                     Save raw rows; publish live events
    |
    +-- deduplicator.process_trigger()
    |      +-- Merge groups and keep growing ones open
    |      +-- grading_engine.grade()             Score settled groups only
    |      +-- Return graded groups
    |
    +-- _flush_final_defects()
           +-- audit_writer.write_final_defects()
                  Save final rows, scores and raw-member links
```

We wait because the same defect may appear in overlapping cameras or later
frames. Its merged size can grow. Two raw observations of one defect must
not become two separately charged final defects.

The merger currently decides settlement using global pixel coordinates:

```text
frontier = current_frame_row_top_px - configured_vertical_gap_px

group_bottom_px < frontier  -> settle and grade
otherwise                  -> keep open for possible merging
```

An empty detection row can still advance this frontier. This assumes complete,
forward-moving rows; it is not arbitrary late-frame recovery.

## Rules and assumptions

- Use `max(width_mm, height_mm) / mm_per_inch` for the length in inches.
- Up to 3 inches: 1 point; above 3 through 6: 2; above 6 through 9: 4.
- Above 9 inches: 4 points (explicit assumption for the poster's omitted band).
- Configured hole classes receive 4 regardless of size. Confirm YAML labels
  against the deployed model; matching ignores case but does not guess aliases.
- A rectangle reaching both ends of the active inspection width receives 4.
  CoordinateMapper derives that width from consecutive active camera slots,
  configured full-frame pixel width, GSD and overlap, minus the two outer
  stencils once. The orchestrator passes this exact width to the grader.
  For MV1 everything inside these boundaries is assumed to be fabric. This
  full-inspection-width rule does not prove the entire physical fabric is seen.
- The score measures the observed merged bounding rectangle, not exact defect
  area. Clipped or partially observed defects may have unseen extensions.

The grading engine owns no numbering, storage, WebSocket publication, or
roll-total state. Roll acceptance/rejection and points per 100 square yards
are not implemented here. Those require agreed inspected-area accounting,
roll completion, and any per-yard/continuous-defect policy. Partial camera
coverage must not be reported as inspection of the entire fabric width.

## Measuring the defect before scoring

The input is the final merged rectangle, expressed in millimetres:

```text
width_mm  = world_x2_mm - world_x1_mm
height_mm = world_y2_mm - world_y1_mm

size_inches = max(width_mm, height_mm) / mm_per_inch
```

We use the **longer side**, not the area or diagonal. The configured conversion
is `mm_per_inch: 25.4`.

Example: width = 100 mm, height = 20 mm:

```text
Longer side = 100 mm
Size        = 100 / 25.4 = approximately 3.94 inches
Normal size-based score = 2 points
```

## Which rule wins?

The grader checks these rules in order. It takes the first matching rule;
it does **not** add the hole, full-width and size scores together.

### First: hole class

If the class matches `hole_classes`, return `hole_points`, currently **4**.
A 2 mm hole and a 200 mm hole therefore both get 4 points under our policy.

Current labels: `hole`, `small_hole`, `large_hole`.
Matching ignores case and surrounding whitespace, but does not guess synonyms.
For example, `small hole` is not the same label as `small_hole`.

### Second: full inspection width

If the box reaches both inspection boundaries within the configured tolerance,
return `full_width_points`, currently **4**. The tolerance is explained below.

### Third: ordinary size bands

| Longer side in inches | Current score |
|---|---:|
| Up to and including 3 | 1 |
| Above 3, up to and including 6 | 2 |
| Above 6, up to and including 9 | 4 |
| Above 9 | 4 |

Exactly 3 inches gets 1 point, exactly 6 gets 2, exactly 9 gets 4.
There is **no 3-point size band in our current poster policy**.
The above-9-inch case was omitted by the poster; our explicit project
assumption is `oversize_points: 4`.

## Where does the inspection width come from?

It comes from the active cameras' combined view, minus the outer stencils.
The mapper calculates it; the orchestrator passes `mapper.active_width_mm`
to the grading engine. No independent fabric-width setting is used here.

For cameras 3 and 4:

```text
Full frame width    = 4096 pixels (not the SAHI tile size)
Horizontal GSD      = 0.113 mm/pixel
One camera width    = 4096 x 0.113 = 462.848 mm
Overlap             = 10%
Step between starts = 462.848 x 0.90 = 416.5632 mm

Camera 3 start      = index 2 x step = 833.1264 mm
Camera 4 end        = index 3 x step + 462.848 = 1712.5376 mm
Combined view       = 1712.5376 - 833.1264 = 879.4112 mm

Usable width        = 879.4112 - 50 left stencil - 50 right stencil
                    = 779.4112 mm
```

The stencils are removed **once at the two outer edges**, not per camera.
With the current left origin, inspection X runs from 0 to 779.4112 mm.
MV1 assumes everything inside is fabric; this is not automatic edge detection.
Active slots must be consecutive and frame widths must match configuration.

## Full-width tolerance: what does it actually do?

It allows a detected rectangle to stop slightly short of both ends and still
receive the **4-point full-width penalty**.

The actual condition is:

```text
world_x1_mm <= full_width_tolerance_mm
AND
world_x2_mm >= active_width_mm - full_width_tolerance_mm
```

Both conditions must hold. The allowance is applied at each edge separately.

### Example where tolerance changes the grade

For illustration, suppose inspection width is **100 mm** (not our MV1 width).
A non-hole defect is between X = 1 and X = 99 mm and is 10 mm high:

```text
Inspection: 0 |--------------------------------------| 100
Defect:       1 |==================================| 99
```

| Tolerance | Decision | Grade |
|---|---|---|
| 0 mm | Misses both exact boundaries; not full-width. | Size rule: 98 / 25.4 = 3.86 inches -> **2 points**. |
| 2 mm | 1 <= 2 and 99 >= 98; qualifies as full-width. | Full-width rule -> **4 points**. |

So tolerance can replace the normal size-based score with 4 points.
It does **not** move stencil boundaries, alter coordinates, or decide which
detections to keep. Failing the full-width test does not discard the defect.

Our current tolerance is **0.0 mm**: there is no intentional edge allowance.
The setting still exists; discussing its removal did not remove it.
For our approximately 779 mm inspected width, a spanning defect already gets
4 points by size, so the full-width rule normally makes no score difference
in this particular setup.

## YAML reference

These values belong under `inference.grading`:

```yaml
grading:
  mm_per_inch: 25.4
  length_limits_inches: [3.0, 6.0, 9.0]
  length_points: [1, 2, 4]
  oversize_points: 4
  hole_classes: ["hole", "small_hole", "large_hole"]
  hole_points: 4
  full_width_points: 4
  full_width_tolerance_mm: 0.0
```

The first limit pairs with the first point value, and so on. Pydantic checks
positive finite increasing limits, equal list lengths, and scores between
1 and 4. The engine checks that the inspection width is positive and finite
and that tolerance is less than half that width.

Policy changes need confirmation from the inspection/business owner.
Camera commissioning owns GSD, overlap, positions, frame width and stencil
measurements: incorrect geometry can produce incorrect grading.

Note: the existing YAML comment saying full width is NOT the active camera
set is stale. The current code uses the active-camera inspection width as
described here. This documentation update does not change runtime settings.

## Saving and failure handling

`grade()` returns a copy with `four_point_score` filled in. Class, confidence,
coordinates and raw-member IDs remain on the group. Open groups have no score.

AuditWriter rejects an ungraded group. It writes the score to
`defect_final.four_point_score` with the final row and raw-member links in one
database transaction. It does not add together scores from raw observations.

- Invalid/nonfinite coordinates or an empty rectangle raise an error, not a fake score.
- Deduplication grades before committing its new in-memory state.
- The orchestrator retains pending final groups until their write succeeds.
  This queue is in memory; it is not crash-durable storage.
- An exact-membership retry returns the already-saved final row, without
  creating another number or overwriting its stored score.

## What is not implemented by this grader?

- Roll acceptance/rejection and normalized points per 100 square yards.
- Per-linear-yard point caps or continuous-defect charging rules.
- Automatic fabric-edge/background identification.
- Final-score WebSocket updates. The existing live event is sent before
  settlement and is not a final grading result.

In short: **detect -> map -> save raw observations -> merge -> settle ->
grade -> save one final defect with its score**.
