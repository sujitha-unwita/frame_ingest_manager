# Integrated live frame ingestion

The full application now supports `frame_ingest.source_kind: mvs`. The camera adapters,
SDK bindings and buffer pools were adapted from `C:/SUJITHA/Code_Frame_ingest_manager`.
The standalone serial/UDP readers and MCU command client are not started: the existing
`control_service.McuLink` remains the sole serial-port owner.

## Data path and ownership

1. The vision child loads/warms the model, enumerates MVS devices, selects cameras by
   configured identity, opens/configures them and arms every camera before reporting ready.
2. The existing Start Capture workflow waits for both workers and commits the roll before
   sending MCU Resume. MCU electrical pulses expose the cameras; software does not trigger them.
3. Camera acquisition threads copy each SDK frame into a bounded application-owned pool,
   free the SDK buffer immediately, and index the lease using the camera counter.
4. Validated telemetry arrives through the existing control-to-vision queue. The manager
   waits for the exact camera row for that MCU trigger and returns a `TriggerFrameBundle`.
5. Inference reads `CameraFrame.image` directly (uint8 mono/BGR), then uses the existing
   tiling, YOLO, SAHI and coordinate mapping logic. The bundle contains the actual MCU
   trigger ID, encoder count and MCU timestamp; camera counters and device timestamps
   remain separate metadata. Device ticks are not presented as MCU-clock microseconds.
6. Frames with mapped detections are encoded to the audit storage key's configured
   JPG/JPEG/PNG format. Immutable encoded bytes cross the bounded audit queue; no pointer,
   NumPy pool view or temporary file crosses processes. All leases are returned after
   inference and encoding, including on errors/cancellation. Empty camera rows still
   reach deduplication, without uploading images, as in the existing audit flow.
7. Audit uses the existing S3 writes, raw transactions, live events, merging, grading and
   final numbering. JPEG is the existing storage default and is lossy; use a PNG key format
   with `image/png` if lossless audit images are required.

## Layout

- `frame_sources/mvs/`: SDK discovery/configuration, camera grab threads, pixel conversion,
  and the supplied `MvImport` vendor bindings.
- `frame_sources/mvs_frame_source.py`: source lifecycle and conversion into pipeline frames.
- `frame_sources/folder_frame_source.py`: retained deterministic looping folder mock.
- `models/`: MVS settings, captured buffer leases, telemetry, camera frames and bundles.
- `services/buffer_pool.py`: preallocated camera buffers.
- `services/frame_matcher.py`: synchronized counter matching and ownership transfer.
- `services/frame_ingest_manager.py`: shared source lifecycle and complete-bundle validation.

## Configure and run on the Jetson

In the main `config.yaml`:

- `frame_ingest.source_kind` selects `mvs` (already selected).
- The initial serial mapping is `camera_3: DB2239178`, `camera_4: DB2239179`, following the
  standalone cam0/cam1 order. Verify physical ordering against `inference.layout`.
- Set `frame_ingest.mvs.trigger_id_base` and each camera's `frame_number_base` from a
  verified observation of the SAME physical pulse. These are intentionally null until
  verified. Startup refuses to guess alignment or use first-to-first arrival matching.
- `sequence_source: frame_number` uses SDK `nFrameNum`; `trigger_index` is available only
  when its hardware semantics/chunk configuration have been verified. In that mode,
  `frame_number_base` denotes the configured trigger-index counter anchor.
- The relation is `MCU trigger = trigger_id_base + camera counter offset from its base`.
  For example, if verified MCU trigger 100 exposed left camera frame 2 and right camera
  frame 5, configure 100 / 2 / 5 respectively. Missing images/telemetry do not shift the
  relationship. Every physical MCU pulse must cause exactly one exposure per camera.
- Keep the MCU paused while starting/arming. Re-establish anchors after restarting cameras
  or resetting the MCU. Pause/resume within a running session retains the same anchors.
- Install the MVS native runtime and its required environment/library paths on the Jetson.
  The Python bindings are included; the native SDK is not. `sdk.mvimport_dir` resolves
  relative to the main configuration file, and can point to another installed binding set.
- The configured camera width must match `inference.layout.frame_width_px` (4096 currently).
  Coordinate calibration and physical camera ordering are not inferred from enumeration.
- Configure the existing model, PostgreSQL password and S3 credentials as before.

Run the full service, not either standalone runner:

```text
python main.py serve --real
```

Use the existing operator panel/dashboard Start Capture, Stop and Resume controls.
`serve` rejects a live MVS source combined with mock telemetry. `--real` selects the
existing serial transport; use `--port` if the configured device differs.

For folder-only testing, select `frame_ingest.source_kind: folder`, set `folder_path`,
and use the existing mock MCU. MVS settings and native libraries are not loaded for folders.

## Failure and capacity behavior

A complete frame from every configured camera is required by downstream merging. A
missing frame waits up to `bundle_timeout_s` when its telemetry job is processed, then
fails the vision job. It is never replaced by the next camera image. Frames with lost
packets, camera disconnection, duplicate device assignments, and exhausted camera pools
also fail explicitly. Late/already-consumed frames are released. Unadmitted telemetry
rows are released as the next admitted trigger advances the matching frontier.

`slots_per_camera` bounds acquired-frame storage; `max_total_memory_mb` limits those
pools. It does not include the model, conversion/encoding temporaries or encoded audit
queue payloads. Size `processing.audit_queue_capacity` for the actual image sizes and
measure throughput on the Jetson. This implementation uses encoded IPC copies, not
shared memory. It does not guarantee acquisition/inference can sustain the machine rate.

The existing pipeline supports forward, nonnegative encoder positions. MCU counter
reset/wrap requires re-alignment; reverse-roll processing is not added here. Camera
sequence gaps remain gaps, while delayed frames can fill a gap before bundle timeout.

Existing machine-control behavior is preserved: worker errors and application shutdown
do not issue MCU Pause, and workers are not automatically restarted. Stop capture from
the operator controls before shutdown. Existing end-of-roll/recovery limitations remain.

## Verification

Hardware-free tests use the real camera counter logic, buffer pool, bundle matching,
image preprocessing, tiling, SAHI postprocessing, coordinate mapping and image encoding.
Camera SDK I/O and model prediction outputs are replaced. No camera, CUDA, DB or S3
connection is opened by these tests:

```text
python -B -m unittest tests.frame_ingest_manager.test_mvs_ingest tests.frame_ingest_manager.test_frame_source_factory tests.test_live_camera_pipeline tests.test_process_runtime tests.test_app_orchestrator tests.test_audit_worker
```

These tests cannot verify the physical pulse wiring, actual counter anchors, native SDK,
TensorRT engine compatibility or sustained hardware throughput.

Local validation (2026-10-05): 108 focused integration/regression tests passed on
Windows/Python 3.12 with CPU PyTorch 2.8.0, Ultralytics 8.4.0, SAHI 0.12.6 and the
repository's OpenCV 4.11.0.86 override. Deployment still targets Python 3.10/Jetson;
physical camera, native SDK and CUDA/TensorRT verification remain on-device work.
The isolated local test environment is `.venv-ingest`.

A broader run also exercised `tests.audit_write.test_audit_writer`: its 15 existing
tests fail during fixture construction because `four_point_score` is passed into
`MappedDetection`, whose unchanged schema forbids that field. Those unrelated tests
and the audit writer's business logic were left unchanged. The focused passing suite
includes the audit worker tests and a new live-frame-to-raw-write/merge/final-write test.
