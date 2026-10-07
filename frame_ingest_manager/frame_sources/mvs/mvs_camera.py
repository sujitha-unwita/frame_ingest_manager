"""HIKROBOT GigE camera driven by the MCU hardware trigger."""

import ctypes
import logging
import threading

from ...models.captured_frame import RawFrameInfo
from . import pixel_formats
from .base import CameraSource, DeviceGeometry, GrabbedFrame
from .mvs_sdk import MvsError


LOGGER = logging.getLogger(__name__)


class _MvsGrabbedFrame(GrabbedFrame):
    def __init__(self, sdk, camera, sdk_frame, info, camera_id):
        super().__init__(info)
        self._sdk = sdk
        self._camera = camera
        self._sdk_frame = sdk_frame
        self._camera_id = camera_id

    def copy_into(self, destination):
        ctypes.memmove(destination.ctypes.data, self._sdk_frame.pBufAddr, self.info.frame_bytes)

    def free(self):
        code = self._camera.MV_CC_FreeImageBuffer(self._sdk_frame)
        if code != self._sdk.MV_OK:
            LOGGER.warning("[%s] FreeImageBuffer failed: 0x%08x", self._camera_id, code & 0xFFFFFFFF)


class MvsCamera(CameraSource):
    """Opens the device, applies trigger + YAML features, then only waits for
    frames. It never sends a trigger itself - exposures come from the MCU."""

    def __init__(self, sdk, sdk_config, descriptor, **kwargs):
        super().__init__(**kwargs)
        self._sdk = sdk
        self._sdk_config = sdk_config
        self._descriptor = descriptor
        self._camera = None
        self._handle_created = False
        self._device_open = False
        self._grabbing = False
        self._disconnected = threading.Event()
        self._exception_callback = None     # keep ctypes callback alive

    # ------------------------------------------------------------------ #
    def _open_device(self):
        sdk, cfg = self._sdk, self.config
        LOGGER.info("[%s] opening %s", self.camera_id, self._descriptor)
        self._camera = sdk.MvCamera()
        sdk.check(self._camera.MV_CC_CreateHandle(self._descriptor.info), f"[{cfg.id}] Create handle")
        self._handle_created = True
        sdk.check(
            self._camera.MV_CC_OpenDevice(sdk.ACCESS_MODES[self._sdk_config.access_mode], 0),
            f"[{cfg.id}] Open device",
        )
        self._device_open = True
        self._register_exception_callback()

        if self._descriptor.transport == "gige":
            self._configure_packet_size()
        if cfg.load_user_set:
            self._apply_feature("UserSetSelector", cfg.load_user_set)
            self._apply_feature("UserSetLoad", True)
        for name, value in cfg.features.items():
            self._apply_feature(name, value)
        for name, value in cfg.optional_features.items():
            self._apply_feature(name, value, required=False)

        sdk.check(self._camera.MV_CC_SetImageNodeNum(cfg.sdk_buffer_nodes), f"[{cfg.id}] Set buffer nodes")
        # Oldest-first delivery keeps pulse order intact for sequence numbering.
        code = self._camera.MV_CC_SetGrabStrategy(sdk.header.MV_GrabStrategy_OneByOne)
        if code == sdk.errors.MV_E_SUPPORT:
            LOGGER.warning("[%s] SetGrabStrategy unsupported; using SDK default OneByOne", cfg.id)
        else:
            sdk.check(code, f"[{cfg.id}] Set grab strategy")
        return DeviceGeometry(
            payload_bytes=sdk.get_int(self._camera, "PayloadSize"),
            width=sdk.get_int(self._camera, "Width"),
            height=sdk.get_int(self._camera, "Height"),
            pixel_type=sdk.get_enum(self._camera, "PixelFormat"),
            timestamp_tick_hz=self._timestamp_tick_hz(),
        )

    def _apply_feature(self, name, value, required=True):
        try:
            self._sdk.set_feature(self._camera, name, value)
            LOGGER.debug("[%s] %s = %r", self.camera_id, name, value)
        except MvsError as error:
            if required and self.config.feature_errors == "fail":
                raise MvsError(f"[{self.camera_id}] {error}") from error
            reason = "optional feature" if not required else "feature_errors=warn"
            LOGGER.warning("[%s] %s (continuing: %s)", self.camera_id, error, reason)

    def _configure_packet_size(self):
        size = self.config.packet_size
        if size == "auto":
            size = int(self._camera.MV_CC_GetOptimalPacketSize())
            if size <= 0:
                LOGGER.warning("[%s] optimal packet size unavailable (%d); keeping camera value",
                               self.camera_id, size)
                return
        self._apply_feature("GevSCPSPacketSize", size)
        LOGGER.info("[%s] GigE packet size %d", self.camera_id, size)

    def _timestamp_tick_hz(self):
        try:
            value = self._sdk.get_int(self._camera, "GevTimestampTickFrequency")
            if value > 0:
                return float(value)
        except MvsError:
            pass
        return self.config.timestamp_tick_hz

    def _register_exception_callback(self):
        disconnect_code = self._sdk.const.MV_EXCEPTION_DEV_DISCONNECT
        callback_type = ctypes.CFUNCTYPE(None, ctypes.c_uint, ctypes.c_void_p)

        def on_exception(message_type, _user):
            LOGGER.error("[%s] MVS exception 0x%08x", self.camera_id, message_type)
            if message_type == disconnect_code:
                self._disconnected.set()

        self._exception_callback = callback_type(on_exception)
        code = self._camera.MV_CC_RegisterExceptionCallBack(self._exception_callback, None)
        if code != self._sdk.MV_OK:
            LOGGER.warning("[%s] could not register exception callback: 0x%08x",
                           self.camera_id, code & 0xFFFFFFFF)

    # ------------------------------------------------------------------ #
    def _start_device(self):
        self._disconnected.clear()
        self._sdk.check(self._camera.MV_CC_StartGrabbing(), f"[{self.camera_id}] Start grabbing")
        self._grabbing = True
        # Before the grab thread starts, discard any SDK images already queued.
        # The MCU must stay paused until all cameras are armed; clearing a live
        # stream could discard a real trigger frame. This does not reset counters
        # or zero the SDK's underlying memory.
        self._sdk.check(
            self._camera.MV_CC_ClearImageBuffer(),
            f"[{self.camera_id}] Clear startup image buffer",
        )
        LOGGER.info("[%s] SDK image queue cleared at startup", self.camera_id)
        LOGGER.info("[%s] armed; waiting for hardware trigger", self.camera_id)

    def _stop_device(self):
        if self._camera is not None and self._grabbing:
            self._camera.MV_CC_StopGrabbing()
            self._grabbing = False

    def _close_device(self):
        self._stop_device()
        if self._camera is not None and self._device_open:
            self._camera.MV_CC_CloseDevice()
            self._device_open = False
        if self._camera is not None and self._handle_created:
            self._camera.MV_CC_DestroyHandle()
            self._handle_created = False
        self._camera = None

    def _grab(self, timeout_ms):
        if self._disconnected.is_set():
            raise MvsError(f"[{self.camera_id}] camera disconnected")
        sdk = self._sdk
        sdk_frame = sdk.header.MV_FRAME_OUT()
        ctypes.memset(ctypes.byref(sdk_frame), 0, ctypes.sizeof(sdk_frame))
        code = self._camera.MV_CC_GetImageBuffer(sdk_frame, timeout_ms)
        if code in sdk.NO_FRAME_CODES:
            return None
        sdk.check(code, f"[{self.camera_id}] Get image buffer")

        fi = sdk_frame.stFrameInfo
        # Report the SDK values before sequence filtering or telemetry matching.
        # nTriggerIndex is diagnostic only until its hardware-pulse semantics
        # are verified; zero can also mean embedded information is unavailable.
        LOGGER.info(
            "[%s] CAMERA_METADATA: nFrameNum=%d | nTriggerIndex=%d | "
            "nFrameCounter=%d | device_timestamp_ticks=%d | "
            "host_timestamp_ms=%d | nInput=%d | nOutput=%d | nLostPacket=%d",
            self.camera_id, int(fi.nFrameNum), int(fi.nTriggerIndex),
            int(fi.nFrameCounter),
            (int(fi.nDevTimeStampHigh) << 32) | int(fi.nDevTimeStampLow),
            int(fi.nHostTimeStamp), int(fi.nInput), int(fi.nOutput),
            int(fi.nLostPacket),
        )
        width = int(fi.nExtendWidth) or int(fi.nWidth)
        height = int(fi.nExtendHeight) or int(fi.nHeight)
        frame_bytes = int(fi.nFrameLenEx) or int(fi.nFrameLen)
        info = RawFrameInfo(
            width=width,
            height=height,
            pixel_type=int(fi.enPixelType),
            pixel_format_name=pixel_formats.name_of(int(fi.enPixelType)),
            frame_number=int(fi.nFrameNum),
            trigger_index=int(fi.nTriggerIndex),
            device_timestamp_ticks=(int(fi.nDevTimeStampHigh) << 32) | int(fi.nDevTimeStampLow),
            host_timestamp_ms=int(fi.nHostTimeStamp),
            frame_bytes=frame_bytes,
            lost_packets=int(fi.nLostPacket),
        )
        return _MvsGrabbedFrame(sdk, self._camera, sdk_frame, info, self.camera_id)
