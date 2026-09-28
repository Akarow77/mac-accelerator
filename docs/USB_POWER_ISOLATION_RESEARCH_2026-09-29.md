# USB power isolation research — 2026-09-29

Research only after the user disconnected for travel. No device access, settings,
power-role writes or kernel installation performed for this investigation.

## Device evidence and limits

The sanitized incident summary is in [power stability status](POWER_STABILITY_STATUS.md).
The second reset occurred on restored master and its current-boot log reports
UVLO and SMPL. User reports no such problem with the Mac disconnected. This
supports investigating the combined power/data connection but does not identify
backfeed, the Mac, cable, power mux, PD negotiation or a device fault uniquely.
`sink` indicates a power role, not the physical routing of every power source.

## Important software finding: input_suspend is not selective

Official public kernel master resolved to
`b53ae06564df3d34f6914c5d0699594913b16626` during research. This has **not** been
matched to the device's installed kernel build; do not assume binary equivalence.

[`smblib_set_prop_input_suspend`](https://github.com/commaai/agnos-kernel-sdm845/blob/b53ae06564df3d34f6914c5d0699594913b16626/drivers/power/supply/qcom/smb-lib.c#L2030)
votes USB input current to zero and also votes DC suspend. Therefore the generic
battery input_suspend interface is not a Mac-port-only switch. It must not be
used as an unverified data-only workaround on this device.

The USB current setter in that source also branches on PD state and
system_suspend_supported. Forcing current to zero or changing power roles is
not established to preserve the vehicle supply or USB enumeration. A dedicated
software solution requires the correct installed driver and board power-path
mapping, plus bench validation; no ready validated setting was found.

## Hardware candidates, not purchase recommendations

- [PortaPow USB Power Blocker](https://portablepowersupplies.co.uk/product/usb-power-blocker):
  manufacturer's specification says USB 2.0 only, warns many devices lose data,
  and says it does not offer USB-C because most tested devices did not work.
  This is not an appropriate verified solution for the current 5 Gbps link.
- [Acroname USBHub3c port API](https://acroname.com/reference/devices/usbhub3c/entities/port.html):
  provides separate VBUS, USB 2, SuperSpeed and CC control plus voltage/current
  readings. This demonstrates a lab tool suitable for investigating separation;
  it does not prove that a 3X enumerates with VBUS disabled or qualify in-car use.
- Generic powered hubs or products advertised as data-only ports are not proof
  that downstream power is absent. Do not substitute a data blocker (which
  disables data), cut cable conductors, or connect vehicle harness outputs to
  ordinary laptop/test-equipment ports.

## Next controlled investigation

1. Preserve normal vehicle-only stability baseline and exact cable/port topology.
2. Verify installed-kernel source and physical power routing before any setter.
3. On an isolated bench, instrument input voltage/current and negotiated data
   speed while comparing direct Mac and appropriately controlled USB interfaces.
4. Require no resets, no unwanted power flow and sustained data integrity before
   returning to inference/thermal tests. Do not defeat UVLO or watchdog protection.

Until then, leave Mac USB disconnected in the vehicle; Wi-Fi SSH can be evaluated
for development access, not represented as a qualified real-time inference link.

## Chestnut comparison: power restriction is not USB input isolation

User again reports that rebooting stops with the Mac disconnected. Continued
research remains offline with respect to the device.

Official [Chestnut setup](https://comma.ai/shop/chestnut) and
[comma_hack_7 examples](https://github.com/commaai/comma_hack_7#run-on-chestnut)
specify a separate 12 V supply and USB data connection. The current ready-to-drive
product targets comma four; historical 3X GPU-over-USB support is documented in
the [0.9.9 release](https://blog.comma.ai/099release/). These should not be
conflated into a claim that the current kit qualifies our 3X/Mac topology.

### Actual restrictions and monitoring found

- Official openpilot master resolved to `49bbba371c29c5612e69b8ea9f906e151410e8bc`.
  Its [modeld startup](https://github.com/commaai/openpilot/blob/49bbba371c29c5612e69b8ea9f906e151410e8bc/openpilot/selfdrive/modeld/modeld.py#L9)
  defaults `AM_POWER_LIMIT` to 100. It is overridable, and refers to GPU power,
  not a USB-port current cap or a hard limit on total system input power.
- tinygrad master resolved to `f9d3f00ff6b2a5530ab486f6d2fe065877d306e4`.
  `runtime/support/am/amdev.py` applies a positive AM_POWER_LIMIT through the AMD
  SMU, and `runtime/support/am/ip.py::set_power_limit` sends SetPptLimit. This
  cannot control the Mac's Apple GPU or selectively stop 3X USB input power.
- [Chestnut firmware](https://github.com/tinygrad/asm2464pd-firmware/blob/master/handmade/src/main.c)
  exposes INA231 voltage/current and a BOB fault flag; firmware declares a
  self-powered device and has separate PCIe power/reset controls.
- [Monitoring](https://github.com/commaai/openpilot/blob/49bbba371c29c5612e69b8ea9f906e151410e8bc/openpilot/system/hardware/chestnut/monitoring.py)
  reads that status and PCIe state, while status handling reports lost supply,
  link failure and temperature faults. Monitoring is not input-power isolation.
- [Firmware flashing](https://github.com/commaai/openpilot/blob/49bbba371c29c5612e69b8ea9f906e151410e8bc/openpilot/system/hardware/chestnut/flash.py)
  uses the comma's smb2-vbus regulator to cycle bridge power. Its comments say
  the ASM bridge enumerates on USB-C alone. Thus even Chestnut is not accurately
  described as a universal VBUS-disconnected data-only cable arrangement.

### Key topology difference

Chestnut is a USB peripheral driven by tinygrad on the comma/PC host; GPU power
comes from the separate supply, while the bridge can use USB power. Our existing
NCM/ADB path makes the Mac the host and 3X the USB gadget, with a recorded sink
role on 3X. This reverses the data-host topology and changes power negotiation;
same connector does not imply same electrical behavior.

No inspected Chestnut code establishes a reusable setting that selectively
blocks Mac-to-3X power while preserving vehicle power and USB3 data. A host-role
swap alone also does not make macOS act as a compatible USB gadget. Hardware
isolation/bridging or board-specific selective input control still needs bench
verification. Useful ideas to borrow now: explicit power/link telemetry, bounded
GPU load and handling of power loss, rather than blindly copying a power setter.

## New preferred wired candidate: two USB Ethernet adapters

Proposed, not yet tested: each computer hosts and powers its own USB Ethernet
adapter; connect the adapters using an ordinary non-PoE Ethernet link. There is
no direct USB VBUS connection between the Mac and 3X in this topology. This is
not a claim of complete system galvanic isolation (shield/ground paths need
separate consideration), nor a promise that the 3X can power every dongle.

The official AGNOS builder selects `tici_defconfig`. The inspected
[kernel configuration](https://github.com/commaai/agnos-kernel-sdm845/blob/b53ae06564df3d34f6914c5d0699594913b16626/arch/arm64/configs/tici_defconfig#L1815)
enables USB_RTL8152, USB_NET_AX88179_178A, USB_USBNET and DWC3 dual-role support.
These are grounds to investigate compatible gigabit adapters, not verification
of the installed kernel, particular adapter revision or macOS support. Do not
assume newer 2.5/5GbE chip revisions work with this older driver.

The authenticated TCP framing can potentially be reused, but current USB-NCM
interface discovery/supervision is not automatically an Ethernet implementation.
Interface binding, IP addressing and reconnect handling must be adapted and
tested separately. Added adapter power draw and TCP CPU overhead are unknown.
No adapter purchase or device reconfiguration was performed.

### Is 1 GbE enough?

For the **compact input-sized protocol**, probably enough bandwidth; for the
current **full padded camera buffers**, not enough at 20 Hz. These are arithmetic
budgets from measured buffer sizes, not measurements on Ethernet hardware.

| Payload at 20 Hz | Application bandwidth before network overhead |
| --- | ---: |
| 393,264-byte synthetic request, 3X to Mac | 62.92 Mbps |
| 36,960-byte result, Mac to 3X | 5.91 Mbps |
| Two 4,804,608-byte raw camera buffers per cycle, 3X to Mac | 1,537.47 Mbps |
| Two 3,735,552-byte NV12 regions, Chestnut-style extent for 1928x1208 | 1,195.38 Mbps |

Gigabit Ethernet is full duplex: requests and results use opposite directions,
so the first two rows do not compete for one shared 1 Gbps direction. Ideal
serialization of one compact request is ~3.15 ms; that is a lower bound which
excludes Ethernet/IP/TCP/authentication overhead, USB adapters, scheduling and
copies, not a predicted request latency. Two raw buffers already require
~76.87 ms at an ideal 1 Gbps, exceeding a 50 ms cycle before inference.

Therefore 1 GbE is a reasonable **bench candidate if the 3X produces compact,
calibrated model inputs** (including required temporal/channel layout) before
transmission. A 512x256 shape alone does not specify payload size; dtype, channel
count, camera count and history representation matter. The ~393 KB test input
does not establish that a continuous real-camera preprocessing path is ready.
Preprocessing must also avoid starving the local model. Merely choosing a faster
adapter does not fix current Python copying/CPU overhead or power stability.

First establish host/driver compatibility and reset-free power operation, then
measure actual authenticated input/result p95/p99/max latency, frame age, local
model coexistence and thermal behavior. Neither 1 GbE bandwidth calculations nor
the prior ~41 ms synthetic USB result qualify live control or guarantee 20 Hz on
Ethernet. No Ethernet implementation or hardware test has been completed.

### Follow-up: does Chestnut send raw frames?

The distinction is **camera allocation vs NV12 image region vs warped model
input**. In the inspected official commit `49bbba371c29c5612e69b8ea9f906e151410e8bc`,
[modeld](https://github.com/commaai/openpilot/blob/49bbba371c29c5612e69b8ea9f906e151410e8bc/openpilot/selfdrive/modeld/modeld.py)
does the following:

1. Computes `frame_copy_size = stride * (y_height + uv_height)`.
2. Copies that NV12 region from each VisionIPC frame into one preallocated input
   buffer, together with transforms and non-recurrent policy inputs.
3. Uploads with `self.input_device.copy_from(self.input_host)`.
4. Runs `self.run_warp(...)`, then the model, on the model-device path.

[helpers.py](https://github.com/commaai/openpilot/blob/49bbba371c29c5612e69b8ea9f906e151410e8bc/openpilot/selfdrive/modeld/helpers.py)
selects `USB+AMD:LLVM` for Chestnut. Thus this implementation **does not first
shrink frames to compact model inputs on the comma before USB upload**. It also
does not upload all allocation padding. NV12 here is the camera pipeline output,
not unprocessed sensor Bayer RAW; the warp is the later model preprocessing.

For 1928x1208, the pinned
[NV12 layout helper](https://github.com/commaai/openpilot/blob/49bbba371c29c5612e69b8ea9f906e151410e8bc/openpilot/system/camerad/cameras/nv12_info.py)
gives stride 2048, Y height 1216 and UV height 608. That is **3,735,552 bytes per
image**, versus the 4,804,608-byte allocation our snapshot probe transmitted.
Omitting the extra allocation area would reduce that probe's payload by **22.25%**,
without implying an equal latency improvement. The two-image NV12 budget remains
~1.195 Gbps at 20 Hz (~59.77 ms per pair even at ideal 1 Gbps).

This is a concrete candidate missed by our full-allocation snapshot comparison:
validate the actual VisionIPC stride/UV offset/format and send only the required
planes, preserving row padding needed by the warp. Never blindly truncate every
camera using these constants. It is not implemented or measured in the Mac
transport yet. With 1 GbE, calibrated preprocessing on the 3X before transmission
is still needed for this two-camera 20 Hz uncompressed design. Chestnut's upload
batching, persistent GPU state and bounded plane copies are useful separately
from its faster USB GPU link; they do not make 1 GbE carry 1.195 Gbps.

### What about 5 GbE at both ends?

It is a plausible bandwidth candidate, **not yet a supported configuration**.
Using the 7,471,104-byte NV12 pair above, ideal link serialization alone is:

| Ethernet rate | Ideal NV12-pair serialization (no overhead) |
| --- | ---: |
| 1 Gbps | 59.77 ms |
| 2.5 Gbps | 23.91 ms |
| 5 Gbps | 11.95 ms |

Neither table nor link negotiation predicts application latency. Both USB
adapters, host USB transfer overhead, drivers, TCP, copies, authentication and
scheduling remain. In particular a 5 Gbps USB uplink cannot deliver 5 Gbps of
application data after encoding/protocol overhead. The previous direct NCM link
already negotiated 5 Gbps yet achieved only 470 Mbps in a short 3X-to-Mac bulk
test, with substantial sender CPU time in later profiles. A different host NIC
driver might improve that path, but no such improvement has been measured.

Compatibility is a separate blocker. The inspected official kernel's
[USB network Kconfig](https://github.com/commaai/agnos-kernel-sdm845/blob/b53ae06564df3d34f6914c5d0699594913b16626/drivers/net/usb/Kconfig)
describes RTL8152/RTL8153 and AX88179 gigabit support and has no AQC111 entry.
This does not establish the installed device's modules or whether an adapter
offers another usable mode. A product's general Linux support statement is not
proof of plug-and-play operation on this AGNOS build. For example,
[StarTech US5GA30](https://www.startech.com/en-de/networking-io/us5ga30) specifies
AQC111U and USB power; its listed OS support is not a verification for M2 macOS
or 3X. This is a compatibility example, **not a purchase recommendation**.

Before choosing adapters, verify exact chipset/revision, installed AGNOS kernel
and driver binding, actual 3X host-mode USB speed, M2 macOS support, and available
power/thermal margin. Each endpoint powering its own NIC removes the direct
inter-device USB VBUS path, but the 3X NIC still draws power and can introduce a
new stability problem. Use non-PoE Ethernet; no kernel modification is authorized
or performed by this research update.

The minimum useful next test is enumeration/power stability, then authenticated
real-size pair latency under local-model load, not just a 5 Gbps link icon or an
iperf peak. 5 GbE could make a Mac-side warp design feasible on bandwidth; only
measurements can determine whether the remaining preprocessing/inference fits.
