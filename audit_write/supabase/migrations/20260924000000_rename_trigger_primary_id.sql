-- Separate the database UUID from the MCU's numeric trigger sequence.
-- Column renames preserve existing values, indexes, and foreign-key links.
-- Keep the original migration unchanged; apply this after it on every database.
alter table public.trigger rename column trigger_id to trigger_primary_id;
alter table public.media rename column trigger_id to trigger_primary_id;
alter table public.defect_live rename column trigger_id to trigger_primary_id;

comment on column public.trigger.trigger_primary_id is
    'Database UUID identifying one camera capture; not the MCU trigger number.';
comment on column public.trigger.trigger_sequence is
    'Original numeric trigger_id received from MCU telemetry.';
