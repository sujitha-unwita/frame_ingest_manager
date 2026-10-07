"""Thin wrapper over HIKROBOT's MvImport ctypes bindings.

The bindings are loaded lazily from `sdk.mvimport_dir` so the rest of the
package (and the simulated backend) works on machines without the MVS SDK.
"""

import ctypes
from dataclasses import dataclass
import importlib
import logging
import sys
import threading


LOGGER = logging.getLogger(__name__)


class MvsError(RuntimeError):
    pass


@dataclass
class DeviceDescriptor:
    index: int
    transport: str
    serial_number: str
    ip: str
    user_defined_name: str
    model_name: str
    info: object          # MV_CC_DEVICE_INFO, kept alive by the SDK list

    def __str__(self):
        return (f"#{self.index} {self.model_name} serial={self.serial_number} "
                f"ip={self.ip or '-'} name={self.user_defined_name or '-'} ({self.transport})")


def _c_string(raw):
    return bytes(raw).split(b"\x00", 1)[0].decode("ascii", errors="replace").strip()


def _ip_to_str(value):
    return ".".join(str((value >> shift) & 0xFF) for shift in (24, 16, 8, 0)) if value else ""


class MvsSdk:
    """Process-wide SDK session (MV_CC_Initialize once, Finalize once)."""

    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self, sdk_config):
        self.config = sdk_config
        self._load_modules()
        self._initialized = False
        self._device_list = None
        self._lock = threading.Lock()

    @classmethod
    def get(cls, sdk_config):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(sdk_config)
            return cls._instance

    # ------------------------------------------------------------------ #
    def _load_modules(self):
        path = self.config.mvimport_dir
        if path is None or not path.is_dir():
            raise MvsError(f"sdk.mvimport_dir not found: {path}")
        if str(path) not in sys.path:
            # The vendor modules import each other by bare module name.
            sys.path.insert(0, str(path))
        try:
            self.const = importlib.import_module("CameraParams_const")
            self.header = importlib.import_module("CameraParams_header")
            self.errors = importlib.import_module("MvErrorDefine_const")
            self.control = importlib.import_module("MvCameraControl_class")
        except OSError as error:
            raise MvsError(
                "Could not load libMvCameraControl. Install the MVS SDK and make "
                "sure MVCAM_COMMON_RUNENV is set."
            ) from error
        self.MvCamera = self.control.MvCamera
        self.MV_OK = self.errors.MV_OK
        self.NO_FRAME_CODES = {
            getattr(self.errors, name)
            for name in ("MV_E_NODATA", "MV_E_GC_TIMEOUT")
            if hasattr(self.errors, name)
        }
        self.TRANSPORTS = {
            "gige": self.const.MV_GIGE_DEVICE,
            "usb": self.const.MV_USB_DEVICE,
            "gentl_gige": getattr(self.const, "MV_GENTL_GIGE_DEVICE", 0),
        }
        self.ACCESS_MODES = {
            "exclusive": self.const.MV_ACCESS_Exclusive,
            "exclusive_with_switch": self.const.MV_ACCESS_ExclusiveWithSwitch,
            "control": self.const.MV_ACCESS_Control,
        }

    def check(self, return_code, operation):
        if return_code != self.MV_OK:
            raise MvsError(f"{operation} failed, MVS error=0x{return_code & 0xFFFFFFFF:08x}")

    # ------------------------------------------------------------------ #
    def initialize(self):
        with self._lock:
            if not self._initialized:
                self.check(self.MvCamera.MV_CC_Initialize(), "Initialize MVS SDK")
                self._initialized = True
                LOGGER.info("MVS SDK initialized")

    def finalize(self):
        with self._lock:
            if self._initialized:
                self.MvCamera.MV_CC_Finalize()
                self._initialized = False
                self._device_list = None
                LOGGER.info("MVS SDK finalized")

    def enumerate(self):
        """Enumerate all devices on the configured transport layers."""
        layer_mask = 0
        for layer in self.config.transport_layers:
            layer_mask |= self.TRANSPORTS[layer]
        devices = self.header.MV_CC_DEVICE_INFO_LIST()
        ctypes.memset(ctypes.byref(devices), 0, ctypes.sizeof(devices))
        self.check(self.MvCamera.MV_CC_EnumDevices(layer_mask, devices), "Enumerate cameras")
        self._device_list = devices      # device info pointers live inside

        descriptors = []
        for index in range(devices.nDeviceNum):
            info = ctypes.cast(devices.pDeviceInfo[index], ctypes.POINTER(self.header.MV_CC_DEVICE_INFO)).contents
            if info.nTLayerType in (self.const.MV_GIGE_DEVICE, self.TRANSPORTS["gentl_gige"]):
                gige = info.SpecialInfo.stGigEInfo
                descriptors.append(DeviceDescriptor(
                    index, "gige", _c_string(gige.chSerialNumber), _ip_to_str(gige.nCurrentIp),
                    _c_string(gige.chUserDefinedName), _c_string(gige.chModelName), info,
                ))
            elif info.nTLayerType == self.const.MV_USB_DEVICE:
                usb = info.SpecialInfo.stUsb3VInfo
                descriptors.append(DeviceDescriptor(
                    index, "usb", _c_string(usb.chSerialNumber), "",
                    _c_string(usb.chUserDefinedName), _c_string(usb.chModelName), info,
                ))
        return descriptors

    @staticmethod
    def find(camera_config, descriptors):
        cfg = camera_config
        for descriptor in descriptors:
            if cfg.serial_number is not None and descriptor.serial_number == cfg.serial_number:
                return descriptor
            if cfg.ip is not None and descriptor.ip == cfg.ip:
                return descriptor
            if cfg.user_defined_name is not None and descriptor.user_defined_name == cfg.user_defined_name:
                return descriptor
            if cfg.device_index is not None and descriptor.index == cfg.device_index:
                return descriptor
        found = "\n  ".join(str(d) for d in descriptors) or "(none)"
        raise MvsError(f"[{cfg.id}] camera with {cfg.describe_identity()} not found. Detected:\n  {found}")

    # ------------------------------------------------------------------ #
    # GenICam feature access
    # ------------------------------------------------------------------ #
    def node_type(self, camera, name):
        node_type = ctypes.c_int(0)
        self.check(camera.MV_XML_GetNodeInterfaceType(name, node_type), f"Find node '{name}'")
        return node_type.value

    def set_feature(self, camera, name, value):
        """Set any GenICam node, choosing the SDK call from the node's type."""
        h = self.header
        node_type = self.node_type(camera, name)
        if node_type == h.IFT_IEnumeration:
            if isinstance(value, bool):          # YAML turns bare On/Off into bools
                value = "On" if value else "Off"
            if isinstance(value, int):
                code = camera.MV_CC_SetEnumValue(name, value)
            else:
                code = camera.MV_CC_SetEnumValueByString(name, str(value))
        elif node_type == h.IFT_IInteger:
            code = camera.MV_CC_SetIntValueEx(name, int(value))
        elif node_type == h.IFT_IFloat:
            code = camera.MV_CC_SetFloatValue(name, float(value))
        elif node_type == h.IFT_IBoolean:
            code = camera.MV_CC_SetBoolValue(name, bool(value))
        elif node_type == h.IFT_IString:
            code = camera.MV_CC_SetStringValue(name, str(value))
        elif node_type == h.IFT_ICommand:
            if not value:
                return
            code = camera.MV_CC_SetCommandValue(name)
        else:
            raise MvsError(f"Node '{name}' has unsupported interface type {node_type}")
        self.check(code, f"Set {name}={value!r}")

    def get_int(self, camera, name):
        value = self.header.MVCC_INTVALUE_EX()
        self.check(camera.MV_CC_GetIntValueEx(name, value), f"Get {name}")
        return int(value.nCurValue)

    def get_enum(self, camera, name):
        value = self.header.MVCC_ENUMVALUE()
        self.check(camera.MV_CC_GetEnumValue(name, value), f"Get {name}")
        return int(value.nCurValue)
