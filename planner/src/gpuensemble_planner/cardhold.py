"""Keep this PC's cards at full clocks while a model runs on them.

With layers split over several cards (or over the CPU, or another PC's card),
each card waits for the others in every token. The NVIDIA driver on Windows
takes those waits for light load and lowers the card's clocks (P3 to P8,
"GPU idle"), and the next turn of that card then runs slower: on the E5 rig a
placement split with another PC ran 20-30% below its estimate
(benchmarks/phase2c-three-cards.md).

The NVIDIA Control Panel's power management mode "Prefer maximum performance"
stops that, but only for a card that a graphics program (Vulkan, Direct3D,
OpenGL) has open: it does nothing for CUDA alone. Measured on the E5 with the
setting on (benchmarks/clock-hold.md): a bursty CUDA load let both cards fall
to P3-P8, and with a Vulkan device merely open on the 2080 SUPER (7 MiB of
VRAM, no work) that card stayed at P2 and full clocks through the same load.
The 2080 Ti, which drives the display, kept its clocks only while a program
on the desktop happened to draw.

``CardHold`` opens such a Vulkan device on each local card the plan uses and
closes it when the model stops. It costs about 40 W per card while the model
is loaded and idle. Nothing here changes a setting: when the power management
mode is not "Prefer maximum performance", ``power_mode_note`` says so.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
from typing import Callable

from .devices import Device
from .plan import Plan

PREFER_MAXIMUM_PERFORMANCE = 1
_POWER_MODES = {0: "Adaptive", 1: "Prefer maximum performance", 2: "Driver controlled",
                3: "Prefer consistent performance", 4: "Prefer minimum power", 5: "Optimal power",
                6: "Normal"}


def cards_to_hold(plan: Plan, devices: list[Device]) -> list[int]:
    """nvidia-smi indices of this PC's cards that wait for other stages in each token.

    A card waits when the model is split: over several cards, with layers on
    the CPU, or with a card of another PC. A model on one card alone keeps it
    busy, and nothing is held then.
    """
    by_name = {d.name: d for d in devices}
    stages = [s for s in plan.stages if s.n_layers > 0]
    if len(stages) + (1 if plan.cpu_layers else 0) < 2:
        return []
    held: list[int] = []
    for s in stages:
        dev = by_name[s.device]
        name = dev.llama_device or ""
        if dev.is_remote or not name.startswith("CUDA") or not name[4:].isdigit():
            continue
        if int(name[4:]) not in held:
            held.append(int(name[4:]))
    return held


def parse_uuids(out: str) -> dict[int, bytes]:
    """{index: 16-byte UUID} from ``nvidia-smi --query-gpu=index,uuid --format=csv,noheader``."""
    uuids: dict[int, bytes] = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        text = parts[1].removeprefix("GPU-").replace("-", "")
        try:
            raw = bytes.fromhex(text)
        except ValueError:
            continue
        if len(raw) == 16:
            uuids[int(parts[0])] = raw
    return uuids


def _nvidia_uuids() -> dict[int, bytes]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        raise OSError("nvidia-smi was not found")
    out = subprocess.run([exe, "--query-gpu=index,uuid", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True, timeout=30).stdout
    return parse_uuids(out)


# --- Vulkan, through ctypes: an instance and one logical device per card, nothing else. ---

_P = ctypes.c_void_p


class _AppInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", _P), ("pApplicationName", ctypes.c_char_p),
                ("applicationVersion", ctypes.c_uint32), ("pEngineName", ctypes.c_char_p),
                ("engineVersion", ctypes.c_uint32), ("apiVersion", ctypes.c_uint32)]


class _InstanceInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", _P), ("flags", ctypes.c_uint32),
                ("pApplicationInfo", ctypes.POINTER(_AppInfo)), ("enabledLayerCount", ctypes.c_uint32),
                ("ppEnabledLayerNames", _P), ("enabledExtensionCount", ctypes.c_uint32),
                ("ppEnabledExtensionNames", _P)]


class _QueueInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", _P), ("flags", ctypes.c_uint32),
                ("queueFamilyIndex", ctypes.c_uint32), ("queueCount", ctypes.c_uint32),
                ("pQueuePriorities", ctypes.POINTER(ctypes.c_float))]


class _DeviceInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", _P), ("flags", ctypes.c_uint32),
                ("queueCreateInfoCount", ctypes.c_uint32), ("pQueueCreateInfos", ctypes.POINTER(_QueueInfo)),
                ("enabledLayerCount", ctypes.c_uint32), ("ppEnabledLayerNames", _P),
                ("enabledExtensionCount", ctypes.c_uint32), ("ppEnabledExtensionNames", _P),
                ("pEnabledFeatures", _P)]


class _IDProperties(ctypes.Structure):  # VkPhysicalDeviceIDProperties (Vulkan 1.1)
    _fields_ = [("sType", ctypes.c_int), ("pNext", _P), ("deviceUUID", ctypes.c_ubyte * 16),
                ("driverUUID", ctypes.c_ubyte * 16), ("deviceLUID", ctypes.c_ubyte * 8),
                ("deviceNodeMask", ctypes.c_uint32), ("deviceLUIDValid", ctypes.c_uint32)]


class _Properties2(ctypes.Structure):  # VkPhysicalDeviceProperties2; the properties are 824 bytes on x64
    _fields_ = [("sType", ctypes.c_int), ("pNext", _P), ("properties", ctypes.c_ubyte * 1024)]


_ST_APPLICATION_INFO, _ST_INSTANCE_CREATE_INFO, _ST_DEVICE_QUEUE_CREATE_INFO, _ST_DEVICE_CREATE_INFO = 0, 1, 2, 3
_ST_PROPERTIES_2, _ST_ID_PROPERTIES = 1000059001, 1000071004
_VK_API_1_1 = (1 << 22) | (1 << 12)


class CardHold:
    """Vulkan devices held open on some of this PC's cards; ``close()`` lets them go."""

    def __init__(self) -> None:
        self.held: list[int] = []
        self._vk = None
        self._instance = _P()
        self._devices: list[_P] = []

    @classmethod
    def open(cls, indices: list[int], uuids: dict[int, bytes] | None = None) -> "CardHold":
        """Hold the cards with these nvidia-smi indices. Raises OSError when Vulkan cannot be used."""
        hold = cls()
        if not indices:
            return hold
        if sys.platform != "win32":
            raise OSError("only done on Windows")
        uuids = _nvidia_uuids() if uuids is None else uuids
        wanted = {uuids[i]: i for i in indices if i in uuids}
        if not wanted:
            raise OSError("nvidia-smi gave no UUID for these cards")
        try:
            vk = ctypes.WinDLL("vulkan-1.dll")
        except OSError as e:
            raise OSError(f"no Vulkan loader ({e})") from None
        hold._vk = vk
        try:
            app = _AppInfo(_ST_APPLICATION_INFO, None, b"GPUEnsemble clock hold", 1, b"GPUEnsemble", 1, _VK_API_1_1)
            info = _InstanceInfo(_ST_INSTANCE_CREATE_INFO, None, 0, ctypes.pointer(app), 0, None, 0, None)
            r = vk.vkCreateInstance(ctypes.byref(info), None, ctypes.byref(hold._instance))
            if r != 0:
                raise OSError(f"vkCreateInstance returned {r}")
            count = ctypes.c_uint32()
            vk.vkEnumeratePhysicalDevices(hold._instance, ctypes.byref(count), None)
            physical = (_P * count.value)()
            vk.vkEnumeratePhysicalDevices(hold._instance, ctypes.byref(count), physical)
            get_props2 = vk.vkGetPhysicalDeviceProperties2
            for pd in physical:
                ids = _IDProperties(_ST_ID_PROPERTIES, None)
                props = _Properties2(_ST_PROPERTIES_2, ctypes.cast(ctypes.pointer(ids), _P))
                get_props2(_P(pd), ctypes.byref(props))
                index = wanted.get(bytes(ids.deviceUUID))
                if index is None or index in hold.held:
                    continue
                priority = ctypes.c_float(1.0)
                queue = _QueueInfo(_ST_DEVICE_QUEUE_CREATE_INFO, None, 0, 0, 1, ctypes.pointer(priority))
                dinfo = _DeviceInfo(_ST_DEVICE_CREATE_INFO, None, 0, 1, ctypes.pointer(queue), 0, None, 0, None, None)
                device = _P()
                if vk.vkCreateDevice(_P(pd), ctypes.byref(dinfo), None, ctypes.byref(device)) == 0:
                    hold._devices.append(device)
                    hold.held.append(index)
        except Exception:
            hold.close()
            raise
        hold.held.sort()
        return hold

    def close(self) -> None:
        vk, self._vk = self._vk, None
        if vk is None:
            return
        for device in self._devices:
            vk.vkDestroyDevice(device, None)
        self._devices.clear()
        if self._instance:
            vk.vkDestroyInstance(self._instance, None)
            self._instance = _P()


def power_mode() -> int | None:
    """The NVIDIA driver's global power management mode (``_POWER_MODES``), or None if it cannot be read."""
    if sys.platform != "win32":
        return None
    try:
        nv = ctypes.WinDLL("nvapi64.dll" if ctypes.sizeof(_P) == 8 else "nvapi.dll")
    except OSError:
        return None
    query = nv.nvapi_QueryInterface
    query.restype, query.argtypes = _P, [ctypes.c_uint]

    def fn(fid: int, *args):
        address = query(fid)
        if not address:
            raise OSError(f"NvAPI function {fid:#x} is missing")
        return ctypes.CFUNCTYPE(ctypes.c_int, *args)(address)

    class Setting(ctypes.Structure):  # NVDRS_SETTING_V1
        _fields_ = [("version", ctypes.c_uint32), ("settingName", ctypes.c_uint16 * 2048),
                    ("settingId", ctypes.c_uint32), ("settingType", ctypes.c_uint32),
                    ("settingLocation", ctypes.c_uint32), ("isCurrentPredefined", ctypes.c_uint32),
                    ("isPredefinedValid", ctypes.c_uint32), ("predefined", ctypes.c_uint32 * 1025),
                    ("current", ctypes.c_uint32 * 1025)]

    session = _P()
    try:
        if fn(0x0150E828)() != 0:  # NvAPI_Initialize
            return None
        if fn(0x0694D52E, ctypes.POINTER(_P))(ctypes.byref(session)) != 0:  # NvAPI_DRS_CreateSession
            return None
        if fn(0x375DBD6B, _P)(session) != 0:  # NvAPI_DRS_LoadSettings
            return None
        profile = _P()
        if fn(0x617BFF9F, _P, ctypes.POINTER(_P))(session, ctypes.byref(profile)) != 0:  # GetCurrentGlobalProfile
            return None
        setting = Setting(version=ctypes.sizeof(Setting) | (1 << 16))
        # NvAPI_DRS_GetSetting, PREFERRED_PSTATE_ID
        if fn(0x73BF8338, _P, _P, ctypes.c_uint32, _P)(session, profile, 0x1057EB71, ctypes.byref(setting)) != 0:
            return 0  # not set: the driver's default, Adaptive / Normal
        return int(setting.current[0])
    except OSError:
        return None
    finally:
        if session:
            try:
                fn(0xDAD9CFF8, _P)(session)  # NvAPI_DRS_DestroySession
            except OSError:
                pass


def power_mode_note(mode: int | None) -> str | None:
    """What to tell when the power management mode keeps holding the cards from working."""
    if mode is None or mode == PREFER_MAXIMUM_PERFORMANCE:
        return None
    return (f"the NVIDIA Control Panel's power management mode is \"{_POWER_MODES.get(mode, mode)}\": the cards "
            "will still lower their clocks while they wait for the other stages. Set Manage 3D settings > "
            "Power management mode to \"Prefer maximum performance\" to keep them at full clocks")


def hold_for(plan: Plan, devices: list[Device], log: Callable[[str], None]) -> CardHold | None:
    """Hold the plan's waiting local cards, saying what was done; None when nothing is held."""
    indices = cards_to_hold(plan, devices)
    if not indices or sys.platform != "win32":
        return None
    try:
        hold = CardHold.open(indices)
    except OSError as e:
        log(f"note: the cards may lower their clocks while they wait for the other stages (could not hold "
            f"them: {e})")
        return None
    if not hold.held:
        log("note: the cards may lower their clocks while they wait for the other stages (Vulkan does not "
            "show them)")
        return None
    names = ", ".join(f"CUDA{i}" for i in hold.held)
    log(f"note: keeping {names} at full clocks while the model runs (a Vulkan device is held open on each; "
        "`--no-hold-clocks` leaves them to the driver)")
    note = power_mode_note(power_mode())
    if note:
        log(f"note: {note}")
    return hold
