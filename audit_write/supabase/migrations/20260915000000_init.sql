-- VisionBar initial schema (er_diagram_updated.png).
--
-- Append-only: once this file has been pushed, never edit it.  Fix mistakes
-- with a new migration (`supabase migration new <name>`).
--
-- Tables are created parents-first:
--   roll, camera -> trigger -> defect_final -> media -> defect_live

-- ---------------------------------------------------------------------------
-- roll
-- ---------------------------------------------------------------------------
create table public.roll (
    roll_id         uuid primary key default gen_random_uuid(),
    roll_label      text not null check (roll_label <> ''),
    started_at      timestamptz not null default now(),
    ended_at        timestamptz,
    length_mm       double precision check (length_mm >= 0),
    suspect         boolean not null default false,
    suspect_reason  text check (suspect_reason <> ''),
    constraint roll_ended_after_started check (ended_at is null or ended_at >= started_at)
);

create index roll_started_at_idx on public.roll (started_at desc, roll_id desc);
create index roll_label_idx on public.roll (roll_label);

comment on table public.roll is 'One inspected roll of material.';

-- ---------------------------------------------------------------------------
-- camera
-- ---------------------------------------------------------------------------
create table public.camera (
    camera_id  text primary key check (camera_id <> ''),
    active     boolean not null default true
);

comment on table public.camera is 'A camera on the inspection bar.';

-- ---------------------------------------------------------------------------
-- trigger  (one row per camera per physical trigger)
-- ---------------------------------------------------------------------------
create table public.trigger (
    trigger_id        uuid primary key default gen_random_uuid(),
    roll_id           uuid not null references public.roll (roll_id) on delete cascade,
    camera_id         text not null references public.camera (camera_id) on delete restrict,
    trigger_sequence  bigint not null check (trigger_sequence >= 0),
    encoder_count     bigint not null,
    captured_at       timestamptz not null default now(),
    constraint trigger_roll_camera_sequence_key unique (roll_id, camera_id, trigger_sequence)
);

create index trigger_roll_captured_idx on public.trigger (roll_id, captured_at desc, trigger_id desc);
create index trigger_camera_idx on public.trigger (camera_id);

comment on table public.trigger is 'A capture by one camera for one encoder trigger. One trigger fires all cameras.';

-- ---------------------------------------------------------------------------
-- defect_final  (operator-facing defect after deduplication)
-- ---------------------------------------------------------------------------
create table public.defect_final (
    defect_id         uuid primary key default gen_random_uuid(),
    roll_id           uuid not null references public.roll (roll_id) on delete cascade,
    defect_no         integer not null check (defect_no >= 1),
    defect_class      text not null check (defect_class <> ''),
    confidence        real check (confidence between 0 and 1),
    world_x1_mm       double precision,
    world_y1_mm       double precision,
    world_x2_mm       double precision,
    world_y2_mm       double precision,
    defect_area_mm2   double precision check (defect_area_mm2 >= 0),
    four_point_score  real,
    inserted_at       timestamptz not null default now(),
    constraint defect_final_roll_number_key unique (roll_id, defect_no)
);

create index defect_final_roll_inserted_idx on public.defect_final (roll_id, inserted_at desc, defect_id desc);

comment on table public.defect_final is 'Ultimate operator-facing defect after deduplication.';

-- ---------------------------------------------------------------------------
-- media  (media_ref is an S3 object key — never a URL; NULL until uploaded)
-- ---------------------------------------------------------------------------
create table public.media (
    media_id       uuid primary key default gen_random_uuid(),
    roll_id        uuid not null references public.roll (roll_id) on delete cascade,
    trigger_id     uuid not null references public.trigger (trigger_id) on delete cascade,
    camera_id      text not null references public.camera (camera_id) on delete restrict,
    frame_variant  text not null check (frame_variant <> ''),
    media_ref      text check (media_ref <> ''),
    uploaded_at    timestamptz,
    captured_at    timestamptz not null default now(),
    defect_id      uuid references public.defect_final (defect_id) on delete set null
);

create index media_roll_captured_idx on public.media (roll_id, captured_at desc, media_id desc);
create index media_trigger_idx on public.media (trigger_id);
create index media_camera_idx on public.media (camera_id);
create index media_defect_idx on public.media (defect_id) where defect_id is not null;
create index media_pending_upload_idx on public.media (captured_at desc, media_id desc) where media_ref is null;

comment on table public.media is 'A captured image. Bytes live in S3; media_ref holds the object key.';
comment on column public.media.media_ref is 'S3 object key (not a URL). NULL when there is no image.';

-- ---------------------------------------------------------------------------
-- defect_live  (raw model output before deduplication)
-- ---------------------------------------------------------------------------
create table public.defect_live (
    detection_id   uuid primary key default gen_random_uuid(),
    roll_id        uuid not null references public.roll (roll_id) on delete cascade,
    trigger_id     uuid not null references public.trigger (trigger_id) on delete cascade,
    encoder_count  bigint,
    camera_id      text not null references public.camera (camera_id) on delete restrict,
    frame_variant  text,
    media_id       uuid references public.media (media_id) on delete set null,
    tile_ids       integer[],
    defect_class   text not null check (defect_class <> ''),
    confidence     real not null check (confidence between 0 and 1),
    bbox_x1_px     double precision,
    bbox_y1_px     double precision,
    bbox_x2_px     double precision,
    bbox_y2_px     double precision,
    world_x1_mm    double precision,
    world_y1_mm    double precision,
    world_x2_mm    double precision,
    world_y2_mm    double precision,
    model_version  text,
    detector_meta  jsonb,
    defect_id      uuid references public.defect_final (defect_id) on delete set null,
    captured_at    timestamptz not null default now()
);

create index defect_live_roll_captured_idx on public.defect_live (roll_id, captured_at desc, detection_id desc);
create index defect_live_trigger_idx on public.defect_live (trigger_id);
create index defect_live_camera_idx on public.defect_live (camera_id);
create index defect_live_media_idx on public.defect_live (media_id) where media_id is not null;
create index defect_live_defect_idx on public.defect_live (defect_id) where defect_id is not null;
create index defect_live_unmerged_idx on public.defect_live (roll_id, captured_at, detection_id) where defect_id is null;

comment on table public.defect_live is 'Original model output before deduplication.';

-- ---------------------------------------------------------------------------
-- Row Level Security
--
-- Enabled on every table with no policies: the Supabase anon / authenticated
-- API roles see nothing.  The backend connects with the database role in
-- DATABASE_URL, which bypasses RLS.  Add policies in a new migration if the
-- tables must ever be reachable through the Supabase REST API.
-- ---------------------------------------------------------------------------
alter table public.roll         enable row level security;
alter table public.camera       enable row level security;
alter table public.trigger      enable row level security;
alter table public.defect_final enable row level security;
alter table public.media        enable row level security;
alter table public.defect_live  enable row level security;
