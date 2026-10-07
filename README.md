# Integrated Code Workspace

This directory is an isolated starting point for the integrated frame-ingestion,
inference, audit, control, and dashboard application.  It is deliberately a
copy of the currently working control/dashboard service so changes here do not
alter the proven fallback implementation at the repository root.

## Current state

Integration Task 1 is complete: the existing application is copied unchanged
and the new package boundaries exist.  The new packages intentionally contain
no executable production logic yet.  Consequently, running this copy behaves
like the current application:

```powershell
Set-Location integrated_code
python main.py selftest
```

## Ownership boundaries

| Package | Owns | Must not own |
| --- | --- | --- |
| `app_orchestrator.py` | Application composition, startup/shutdown order, routing between services | Business rules, protocol parsing, database queries, image processing |
| `control_service` | MCU serial link, telemetry, commands | Camera frames, inference, database writes, dashboard transports |
| `frame_ingest_manager` | Frame source and trigger/frame assignment (currently folder mock) | Inference, defect numbers, database writes |
| `inference_pipeline` | Detection, mapping, merging and grading | Camera SDK access, serial I/O, dashboard delivery, database numbering |
| `audit_write` | Roll and audit persistence, media storage, transactional final-defect numbering | Detection/merge decisions, coordinate calculation |
| `dashboard_service` | Operator API/WebSocket delivery and event replay | Serial-port ownership, inference, direct database access |

All values likely to change between deployments belong in `config.yaml` and
are represented by validated configuration models. Secrets belong in environment
variables or local secret files, never YAML or source.

## Local database secret

`audit_write.database` in `config.yaml` contains the PostgreSQL host, port,
database name, username, and the environment-variable name. Create the local
secret file with `Copy-Item .env.example .env`, then replace its placeholder.
The audit worker opens the database pool and checks the schema during startup.

## Frame pipeline: three-process runtime

`main.py serve` still starts the application. `AppOrchestrator` now creates a
`ProcessRuntime`, which spawns vision and audit children. Child resources are
constructed inside those children, not copied from the parent.

| Process | Owned work |
| --- | --- |
| Control | MCU serial reader, heartbeats, command/ACK handling, dashboard and worker supervision |
| Vision | Folder or MVS camera source, frame ingest, model/CUDA, coordinate mapping and GPU sampling |
| Audit | DB/S3 clients, roll preparation, raw writes, deduplication/grading and final writes |

Four bounded multiprocessing queues connect them:

| Queue | Direction | Contents |
| --- | --- | --- |
| triggers | Control to vision | Roll ID, trigger metadata, local sequence and admission timestamp |
| audit | Vision to audit | Complete camera row, mapped detections, and file references or owned encoded camera bytes |
| commands | Control to audit | Correlated roll-preparation requests |
| events | Workers to control | Readiness, completion, GPU readings and committed defect events |

There is also a bounded **local asyncio queue** for websocket delivery. Slow
panel sockets do not run inside MCU message dispatch. The existing event store
retains replayable events within its configured retention.

The flow is:

1. Both workers report ready after their startup succeeds.
2. Start Capture waits for audit to commit the roll, then sends MCU resume.
   A concurrent Stop cancels that pending resume; it does not wait for the DB.
3. `AppOrchestrator.on_trigger()` validates and enqueues telemetry immediately.
4. Vision maps every camera, including empty results, and sends a complete row.
5. Audit reads/uploads each affected frame, writes raw detections, returns live
   events, merges persisted observations, and writes settled final defects.
6. Control publishes worker results without waiting for inference or DB/S3.

`processing` in YAML owns queue capacities and deadlines. A full trigger
queue returns an error for that submission without disabling later submissions.
Worker exits are logged and waiting requests receive an error. Timing overruns
produce warnings only. There is no application pipeline-fault state or callback.
Only an explicit operator request sends Pause/Resume through this orchestrator.
If capture is already running, the operator must stop it: pipeline failure
does not stop the physical machine. Observed MCU state remains authoritative.
There is no automatic worker restart.

`queue_depth` means admitted triggers not yet audit-completed, including
in-flight work. `duty_cycle` remains the last inference's average GPU activity
in percent. Status includes `workers`; fault bits come from the MCU's
`status_flags`, not a synthetic pipeline fault. Each child
writes a separate log with its role and PID; per-trigger logs include processing
and end-to-end timings. Queue isolation does not guarantee the LLD deadlines:
measure these on the Jetson under load.

Folder images must remain unchanged while running. IPC carries paths and
metadata, not decoded arrays; audit reads one affected frame at a time and
releases its bytes after upload/write. Size/mtime checks reject changed files.
Live MVS capture instead passes encoded image bytes through the audit queue. Camera
buffers stay in vision until inference/encoding finishes; no shared-memory pointer is
sent to audit. See [live ingestion setup](frame_ingest_manager/README.md) for camera
configuration, required counter anchors, ownership, and hardware verification.

Shutdown sends no MCU command. Stop capture explicitly before closing the app.
Shutdown stops new admission, drains vision then audit, and waits only up to
configured deadlines.
Forced termination logs incomplete work. Queues are not durable; inspect
worker logs, S3 and database state before retrying after a failure.
Shutdown/pause does not finalize open cross-trigger groups: explicit end-of-roll
completion and restart recovery remain separate work.

The mock now rejects an explicit Pause while ARMED with result=3 and leaves
the state unchanged, matching the observed real-MCU response. Running-to-Paused
on an operator Pause still works. This does not claim full firmware equivalence
for other commands or states.

Offline process tests (no GPU, database, S3 or serial device):

```text
python -m unittest tests.test_process_runtime
```

With application dependencies installed, also run:

```text
python -m unittest tests.test_app_orchestrator tests.test_audit_worker
```

### Stage timing

`utils/timing.py` supplies `Timing.measure()` for blocks and
`Timing.record()` for elapsed time since an earlier host timestamp.
Enable or disable it with `logging.timing_enabled` in YAML. Timing rows go
to dedicated CSV files, independently of the normal log level:

```yaml
logging:
  timing_enabled: true
  timing_file: "logs/timings/timing-{timestamp}-{process}-{pid}.csv"
```

Relative paths resolve from the configuration directory. Each process creates
its own file on its first measurement. The filename must include all three
placeholders. Existing files are never overwritten. The normal application log
announces the CSV path; individual timing rows are not duplicated in that log.

Example usage (works around an awaited call without an async context manager):

```python
with self.timing.measure("inference", trigger_id=trigger.trigger_id):
    detections = await self.detector.detect(frame)
```

Example record, not a benchmark:

```csv
timestamp_utc,process,pid,stage,trigger_id,camera_id,sequence,status,elapsed_ms
2026-10-01T10:30:00.000+00:00,visionbar-vision,123,inference,7,camera_3,,ok,201.345
```

Measured boundaries include model/database startup, trigger enqueue, vision
queue wait, frame bundling, inference, coordinate mapping, audit queue put,
audit handoff wait, frame reading, S3 upload, raw DB transaction (including
commit), live-event handoff, deduplication/grading, final writes, panel queue
wait, websocket broadcast, MCU command handling, and worker shutdown.

`vision_total` and `audit_total` include their nested stages; do not add both
totals and their children. `trigger_to_audit` starts at control admission and
ends after that trigger's audit processing, including queue/I/O waits. It
does not include MCU wire travel or guarantee dashboard rendering. Audit
handoff wait includes IPC backpressure and waiting before audit starts.
Websocket broadcast measures the server send attempt, not a browser receipt
acknowledgment.

These are elapsed wall durations measured with the host monotonic clock,
not CPU usage or CUDA-kernel timings. Timers do not synchronize the GPU.
Errors/cancellation produce `status=error` / `status=cancelled` and propagate
unchanged. Each row opens/appends/closes the CSV, so it is saved immediately
without an unbounded memory buffer or an open handle to manage on shutdown.
Disk writes have overhead; disable timing when it is not needed. If CSV writing
fails (for example disk full), the normal log reports the error once and CSV
writing stops for that timer. It does not hide an original processing exception.
