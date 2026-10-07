-- Captures without a known A/B variant must not invent one.
alter table public.media alter column frame_variant drop not null;
