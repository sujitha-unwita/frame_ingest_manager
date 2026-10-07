# Audit configuration

The main `config/config_loader.py` loads `config.yaml` once. All audit settings
are under `audit_write`, validated by `config/audit_models.py`.
The copied `audit_write/config.py` and `config/database.yaml` have been removed.

Connection host, port, database name, username, TLS, and pool settings are YAML
fields. Set the password using the environment variable named by
`audit_write.database.password_env`, or the configured `.env` file. Environment
values take precedence. Relative paths resolve beside the selected main YAML.
The password is private and excluded from serialized settings. It is required
when audit is enabled; it is also checked before opening any pool.

Database URLs and inline passwords are not accepted for the runtime connection.
Test-only settings live in `tests/test.yaml`, outside the application models
and loader. No database runner currently consumes them. Paths in that file are
relative to its directory; `database_url_env` names an external credential,
not a URL stored in YAML. Application loading never reads the test file.
Pagination remains runtime configuration. Storage bucket/region and client options are direct
YAML values, not environment-placeholder substitutions. AWS credentials still
use boto3's normal credential chain.

Pass `config.audit_write.database` to `Database`, and `config.audit_write` to
repositories. Existing default consumers and row validators use
`get_audit_settings()` from the main loader. This accessor returns the latest
successfully loaded configuration; it never reads files. Load once at startup,
not while audit work is running.

AppOrchestrator creates the database and repositories in its constructor when
audit is enabled. Its start() connects and runs SchemaChecker before loading
the model or starting MCU tasks. stop() drains processing and closes the pool.
The application lifespan is AppOrchestrator._lifespan(); audit resource
startup and shutdown are owned by the orchestrator.

SchemaChecker reads configured Supabase migration history and checks visible
table/column names. Missing history, required versions, or columns stop startup.
It does not verify types, constraints, indexes, or write permissions. The audit
role needs read access to migration history and visibility of application tables.
Configure required versions under audit_write.migrations as releases evolve.

Migration `20260924000000_rename_trigger_primary_id.sql` renames the trigger
UUID column to `trigger_primary_id` in `trigger`, `media`, and `defect_live`.
Apply it through deployment tooling before starting this application version.
Existing UUID values and foreign-key links are preserved. Fresh databases must
apply all migrations before the seed. The initial migration remains unchanged.
`trigger_sequence` still stores the numeric MCU `trigger_id`; telemetry and
inference contracts retain that numeric name. Repository `list_for_trigger`
calls take the database UUID as `trigger_primary_id`, not the numeric sequence.

Provision the database and apply migrations with the deployment migration tool
before startup. A manually executed SQL file does not populate Supabase history;
reconcile that history using deployment tooling, not by bypassing the check.
Startup never creates a database or applies migrations. Existing migration SQL
and storage operations remain unchanged. Audit stays disabled in the sample YAML.

The dashboard delegates operator commands to AppOrchestrator. Before sending
Start Capture to the MCU, it calls `AuditWriter.start_roll(camera_ids)`.
The writer creates the roll and registers configured active cameras in one
transaction. It sets `_roll_id` only after commit. Repeated Start requests and
resume reuse that roll. A failed MCU start does not rotate it. A prepared UUID
is retained across database errors; retry checks for an already-committed row.
Roll completion and recovery after application restart remain backlog.

`AuditWriter.write_frame()` receives the numeric trigger sequence, encoder count,
camera ID, whole-frame bytes, and mapped detections. It uses its stored roll ID
and `utils/identifiers.py` to generate UUIDs. Small private methods upload the
frame and construct/insert trigger, media, and defect_live models. All database
inserts share one transaction. The frame is uploaded once for all its detections.
The S3 key and content type come from `audit_write.frame_key_kind` and the
corresponding `storage.keys` entry. Match the configured format to the original
encoded folder images; no crop or conversion is performed. Unknown camera UTC
timestamps use database defaults, not converted MCU uptime values.

The returned `FrameAuditResult` contains the whole-frame key and observations
with persisted detection UUIDs, without mutating the inputs. The orchestrator
then publishes live events containing both coordinate rectangles, nullable tile
IDs/variant, string camera ID, and the key in `crop_thumbnail_ref`. Detection
UUIDs stay internal for later deduplication; no final numbering occurs here.
Only frames with accepted detections are audited in this first flow.
S3 objects are retained and their keys logged on failure for reconciliation:
S3 is outside the database transaction, and a lost commit reply is ambiguous.

Apply `20260924000001_nullable_media_frame_variant.sql` before using captures
with no frame variant. Tile IDs remain absent until inference supplies them.
SAHI merged results do not currently supply tile provenance, so no tile IDs are
fabricated.

## Settled final defects

The orchestrator passes each complete camera row (including empty lists) to
DeduplicationEngine. Open groups stay in the merger. Only settled groups go to
`AuditWriter.write_final_defects()`. It locks the database roll row, verifies and
locks raw members, allocates the next saved defect number, inserts a final row,
and links its raw members in the same transaction. Numbers follow settlement
order, not a full-roll spatial sort. Exact member-group retries return the
existing final row, including after a lost commit reply. Missing members,
different-roll members, partial links, and conflicting links are rejected.
All final writers must use the roll lock; the database unique constraint is
also retained. No schema migration is required for this step.

Settled batches remain pending in the orchestrator until commit succeeds and
are retried before another frame is consumed, or on orderly shutdown. Pending
and open groups are in memory: process-crash recovery is not implemented.
Failures of raw frame writes also still require reconciliation; this change
does not implement automatic raw-frame replay or a telemetry work queue.
Shutdown and pause do not call merger.finish(): finalizing trailing open groups
requires the deferred explicit roll-end workflow. Four-point score and actual
defect area remain null; bounding-box area is not treated as measured area.
Whole-frame media is shared; membership is followed via defect_live.media_id,
not by assigning one final defect to the entire frame. Existing live WebSocket
events are unchanged; a separate final-defect event is not added here.

Before running: configure the model and fabric width, enable audit_write,
provide database and AWS credentials externally, configure the S3 bucket/region,
and apply all migrations. Start Capture is rejected with audit disabled.
Publishing an object key does not make a private S3 image publicly readable;
image retrieval/signed-URL delivery is not implemented by this change.
