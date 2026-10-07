"""Hardware-free integration tests using the real buffer and camera sequence code."""
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from pydantic import ValidationError
from config.models import FrameIngest
from frame_ingest_manager.models.trigger_telemetry import TriggerTelemetry
from frame_ingest_manager.models.captured_frame import RawFrameInfo
from frame_ingest_manager.frame_sources.mvs.base import CameraSource, DeviceGeometry, GrabbedFrame
from frame_ingest_manager.frame_sources.mvs.mvs_sdk import MvsSdk
from frame_ingest_manager.frame_sources.mvs.pixel_formats import MONO8
from frame_ingest_manager.frame_sources.mvs_frame_source import MvsFrameSource
from frame_ingest_manager.services.frame_ingest_manager import FrameIngestManager
from frame_ingest_manager.services.frame_matcher import FrameMatchError


def configuration(**changes):
    settings = dict(trigger_id_base=100, slots_per_camera=3, bundle_timeout_s=0.08,
        cameras=[{"id": "left", "serial_number": "A", "frame_number_base": 2},
                 {"id": "right", "serial_number": "B", "frame_number_base": 5}])
    settings.update(changes)
    return FrameIngest(source_kind="mvs", camera_ids=["left", "right"],
                       expected_camera_count=2, mvs=settings)


def telemetry(number):
    return TriggerTelemetry(trigger_id=number, encoder_count=number * 10, timestamp_us=number * 1000)


class FakeGrab(GrabbedFrame):
    def __init__(self, number, value=42, lost_packets=0):
        super().__init__(RawFrameInfo(width=4, height=3, pixel_type=MONO8,
            pixel_format_name="Mono8", frame_number=number, trigger_index=0,
            device_timestamp_ticks=number * 100, host_timestamp_ms=123,
            frame_bytes=12, lost_packets=lost_packets))
        self.value = value
        self.freed = False

    def copy_into(self, destination):
        destination[:] = self.value

    def free(self):
        self.freed = True


class FakeCamera(CameraSource):
    def __init__(self, sdk, sdk_config, descriptor, **kwargs):
        super().__init__(**kwargs)
        self.sdk = sdk
        self.closed = False

    def _open_device(self):
        self.sdk.operations.append((self.camera_id, "open"))
        return DeviceGeometry(12, 4, 3, MONO8, 100000000)

    def start(self):
        self._reset_sequence()
        self.sdk.operations.append((self.camera_id, "start"))

    def _stop_device(self):
        self.sdk.operations.append((self.camera_id, "stop"))

    def _close_device(self):
        self.closed = True
        self.sdk.operations.append((self.camera_id, "close"))

    def emit(self, number, value=42, lost_packets=0):
        grabbed = FakeGrab(number, value, lost_packets)
        frame = self._ingest(grabbed)
        if frame is not None:
            self._sink(frame)
        return grabbed


class FakeSdk:
    def __init__(self):
        self.operations = []
        self.finalized = False
        self.descriptors = [SimpleNamespace(index=0, serial_number="A"),
                            SimpleNamespace(index=1, serial_number="B")]

    def initialize(self):
        self.operations.append(("sdk", "initialize"))

    def enumerate(self):
        return self.descriptors

    find = staticmethod(MvsSdk.find)

    def finalize(self):
        self.finalized = True


def make_source(config=None):
    config = config or configuration()
    sdk = FakeSdk()
    source = MvsFrameSource(config, sdk=sdk, camera_factory=FakeCamera)
    manager = FrameIngestManager(config.camera_ids, config.expected_camera_count, source)
    manager.start()
    return source, manager, sdk


class MvsIngestTests(unittest.TestCase):
    def setUp(self):
        self.source, self.manager, self.sdk = make_source()
        self.cameras = self.source.cameras[:]
        self.addCleanup(self.manager.stop)

    def test_discovery_arms_all_cameras_and_matches_different_counter_bases(self):
        self.assertEqual(self.sdk.operations[1:5],
            [("left", "open"), ("right", "open"), ("left", "start"), ("right", "start")])
        self.assertTrue(self.cameras[1].emit(5, 21).freed)
        self.assertTrue(self.cameras[0].emit(2, 11).freed)
        with self.manager.create_bundle(telemetry(100)) as bundle:
            self.assertEqual(bundle.trigger.encoder_count, 1000)
            left = bundle.camera_frames["left"]
            self.assertEqual(left.camera_frame_number, 2)
            self.assertEqual(left.capture_timestamp_us, 100000)
            self.assertEqual(left.device_timestamp_ticks, 200)
            self.assertIsNone(left.hardware_trigger_id)
            self.assertTrue(np.all(left.image == 11))
            self.assertNotIn("image", left.model_dump())
            self.assertEqual(self.cameras[0].pool.free_count(), 2)
        self.assertIsNone(left.image)
        self.assertEqual(self.cameras[0].pool.free_count(), 3)
        bundle.release()
        self.assertEqual(self.cameras[0].pool.free_count(), 3)

    def test_late_out_of_order_camera_arrival_keeps_exact_matching(self):
        self.cameras[0].emit(3, 31)
        self.cameras[0].emit(2, 21)
        self.cameras[1].emit(5, 25)
        with self.manager.create_bundle(telemetry(100)) as first:
            self.assertTrue(np.all(first.camera_frames["left"].image == 21))
        self.cameras[1].emit(6, 36)
        with self.manager.create_bundle(telemetry(101)) as second:
            self.assertTrue(np.all(second.camera_frames["left"].image == 31))

    def test_telemetry_can_arrive_before_frames(self):
        result = []
        consumer = threading.Thread(target=lambda: result.append(self.manager.create_bundle(telemetry(100))))
        consumer.start()
        self.cameras[0].emit(2)
        self.cameras[1].emit(5)
        consumer.join(1)
        self.assertFalse(consumer.is_alive())
        self.assertEqual(len(result), 1)
        result[0].release()

    def test_missing_telemetry_releases_old_row_without_shifting(self):
        for camera, base in zip(self.cameras, [2, 5]):
            camera.emit(base, 10)
            camera.emit(base + 1, 20)
        with self.manager.create_bundle(telemetry(101)) as bundle:
            self.assertTrue(np.all(bundle.camera_frames["left"].image == 20))
        self.assertTrue(all(camera.pool.free_count() == 3 for camera in self.cameras))

    def test_missing_frame_times_out_and_late_frame_is_released(self):
        self.cameras[0].emit(2)
        self.cameras[1].emit(6)
        with self.assertRaisesRegex(FrameMatchError, "right"):
            self.manager.create_bundle(telemetry(100))
        self.cameras[1].emit(5)
        self.cameras[0].emit(3)
        with self.manager.create_bundle(telemetry(101)) as bundle:
            self.assertEqual(bundle.camera_frames["right"].camera_frame_number, 6)
        self.assertTrue(all(camera.pool.free_count() == 3 for camera in self.cameras))

    def test_frames_before_verified_anchor_are_discarded(self):
        self.cameras[0].emit(1)
        self.assertEqual(self.cameras[0].pool.free_count(), 3)

    def test_counter_reset_or_duplicate_telemetry_is_rejected(self):
        self.cameras[0].emit(2)
        self.cameras[1].emit(5)
        self.manager.create_bundle(telemetry(100)).release()
        for number in (100, 0):
            with self.assertRaises(FrameMatchError):
                self.manager.create_bundle(telemetry(number))

    def test_pool_exhaustion_is_a_fault_not_a_shifted_match(self):
        for number in range(2, 6):
            self.cameras[0].emit(number)
        with self.assertRaisesRegex(FrameMatchError, "exhausted"):
            self.manager.check_health()
        self.manager.stop()
        self.assertEqual(self.cameras[0].pool.free_count(), 3)

    def test_corrupted_frame_fault_and_close_return_all_slots(self):
        self.cameras[0].emit(2, lost_packets=1)
        with self.assertRaisesRegex(FrameMatchError, "incomplete"):
            self.manager.check_health()
        self.manager.stop()
        self.assertTrue(self.sdk.finalized)
        self.assertTrue(all(camera.closed and camera.pool.free_count() == 3 for camera in self.cameras))

    def test_fault_wakes_waiter_and_shutdown_releases_pending_frames(self):
        self.cameras[0].emit(2)
        self.source._camera_fault(self.cameras[1], RuntimeError("disconnected"))
        with self.assertRaisesRegex(FrameMatchError, "disconnected"):
            self.manager.create_bundle(telemetry(100))
        self.manager.stop()
        self.assertEqual(self.cameras[0].pool.free_count(), 3)

    def test_partial_startup_failure_closes_opened_camera(self):
        sdk = FakeSdk()
        config = configuration(cameras=[
            {"id": "left", "serial_number": "A", "frame_number_base": 2},
            {"id": "right", "serial_number": "A", "frame_number_base": 5}])
        source = MvsFrameSource(config, sdk=sdk, camera_factory=FakeCamera)
        with self.assertRaisesRegex(FrameMatchError, "same device"):
            source.start()
        self.assertTrue(sdk.finalized)
        self.assertIn(("left", "close"), sdk.operations)

    def test_alignment_and_hardware_trigger_configuration_are_required(self):
        config = configuration(trigger_id_base=None)
        with self.assertRaisesRegex(ValueError, "verified"):
            config.mvs.require_alignment()
        with self.assertRaises(ValidationError):
            configuration(trigger_features={"TriggerMode": "On", "TriggerSource": "Software"})
        with self.assertRaises(ValidationError):
            configuration(cameras=[{"id": "left", "serial_number": "A", "ip": "1.2.3.4"}])

    def test_memory_limit_fails_before_allocating_camera_pool(self):
        sdk = FakeSdk()
        source = MvsFrameSource(configuration(max_total_memory_mb=0.000001),
                                sdk=sdk, camera_factory=FakeCamera)
        with self.assertRaisesRegex(RuntimeError, "max_total_memory_mb"):
            source.start()
        self.assertTrue(sdk.finalized)
        self.assertIn(("left", "close"), sdk.operations)
