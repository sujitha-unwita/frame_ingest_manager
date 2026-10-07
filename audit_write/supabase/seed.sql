-- Sample data for LOCAL development only.
-- Runs on `supabase db reset`; never on a push to a real environment.

insert into public.camera (camera_id, active) values
    ('CAM-01', true),
    ('CAM-02', true),
    ('CAM-03', false);

insert into public.roll (roll_id, roll_label, started_at) values
    ('00000000-0000-0000-0000-000000000001', 'R-DEV-001', now() - interval '1 hour');

insert into public.trigger (trigger_primary_id, roll_id, camera_id, trigger_sequence, encoder_count, captured_at) values
    ('00000000-0000-0000-0000-000000000101', '00000000-0000-0000-0000-000000000001', 'CAM-01', 1, 1200, now() - interval '59 minutes'),
    ('00000000-0000-0000-0000-000000000102', '00000000-0000-0000-0000-000000000001', 'CAM-02', 1, 1200, now() - interval '59 minutes');

insert into public.defect_final (defect_id, roll_id, defect_no, defect_class, confidence,
                                 world_x1_mm, world_y1_mm, world_x2_mm, world_y2_mm, defect_area_mm2, four_point_score) values
    ('00000000-0000-0000-0000-000000000301', '00000000-0000-0000-0000-000000000001', 1, 'hole', 0.93,
     100, 250, 112, 262, 144, 2);

insert into public.media (media_id, roll_id, trigger_primary_id, camera_id, frame_variant, media_ref, uploaded_at, captured_at, defect_id) values
    ('00000000-0000-0000-0000-000000000201', '00000000-0000-0000-0000-000000000001', '00000000-0000-0000-0000-000000000101',
     'CAM-01', 'raw', 'crops/dev/R-DEV-001/seq-000001-CAM-01-raw.jpg', now() - interval '58 minutes', now() - interval '59 minutes',
     '00000000-0000-0000-0000-000000000301');

insert into public.defect_live (detection_id, roll_id, trigger_primary_id, encoder_count, camera_id, frame_variant, media_id,
                                tile_ids, defect_class, confidence,
                                bbox_x1_px, bbox_y1_px, bbox_x2_px, bbox_y2_px,
                                world_x1_mm, world_y1_mm, world_x2_mm, world_y2_mm,
                                model_version, detector_meta, defect_id, captured_at) values
    ('00000000-0000-0000-0000-000000000401', '00000000-0000-0000-0000-000000000001', '00000000-0000-0000-0000-000000000101',
     1200, 'CAM-01', 'raw', '00000000-0000-0000-0000-000000000201',
     array[3, 4], 'hole', 0.93,
     410, 820, 460, 870,
     100, 250, 112, 262,
     'dev-model-0.1', '{"nms_iou": 0.5}', '00000000-0000-0000-0000-000000000301', now() - interval '59 minutes');
