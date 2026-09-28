# USB architecture and power limitations — 2026-09-29

Technical source review and arithmetic budgets, not a hardware compatibility
certification or deployment guide. See [power stability status](POWER_STABILITY_STATUS.md)
and the research summary in [English](RESEARCH_SUMMARY.md) / [한국어](RESEARCH_SUMMARY.ko.md).

## Power topology

Chestnut is a USB peripheral controlled by the comma host, with separate GPU
power. In the Mac NCM/ADB experiment the Mac is the host and 3X the gadget; saved
3X telemetry reports a sink role. Sharing a connector type does not imply the
same data protocol, power role or supply-routing behavior.

The inspected official modeld defaults `AM_POWER_LIMIT` to 100 for the AMD GPU.
This is GPU power budgeting, not a Mac USB-output limit or selective 3X input
isolation. Chestnut monitoring reads supply/link/temperature health. The bridge
can enumerate on USB power alone; it is not a universal VBUS-disconnected link.

Public AGNOS kernel source shows generic battery `input_suspend` affecting both
USB and DC. A supported setting that isolates the physical auxiliary input while
preserving system power and USB3 data has not been validated. Software source
alone does not establish the 3X board's complete physical power routing.

## Follow-up: does Chestnut send raw frames?

The distinction is camera allocation versus NV12 image extent versus warped
model input. In official openpilot commit
`49bbba371c29c5612e69b8ea9f906e151410e8bc`, modeld:

1. Computes `frame_copy_size = stride * (y_height + uv_height)`.
2. Copies each VisionIPC frame's NV12 extent, transforms and non-recurrent inputs
   into a preallocated host input buffer.
3. Uploads with `self.input_device.copy_from(self.input_host)`.
4. Runs warp and inference on the model-device path.

Chestnut selection uses `USB+AMD:LLVM`. This path does not first shrink frames
to compact model tensors on the comma, nor upload every byte of allocation
padding. NV12 is camera pipeline output, not sensor Bayer RAW.

The inspected 1928×1208 layout has stride 2048, Y height 1216 and UV height 608:
3,735,552 bytes/image versus the experiment's 4,804,608-byte snapshot allocation.
The 22.25% size difference is not an implemented or measured latency reduction.
These constants are layout-specific, not a universal frame-truncation rule.

## Bandwidth calculations

Application payload at 20 Hz, excluding framing and network overhead:

| Workload | Directional bandwidth |
| --- | ---: |
| 393,264-byte synthetic request | 62.92 Mbps to Mac |
| 36,960-byte result | 5.91 Mbps from Mac |
| Two 4,804,608-byte camera allocations | 1,537.47 Mbps to Mac |
| Two 3,735,552-byte NV12 extents | 1,195.38 Mbps to Mac |

Thus 1 GbE cannot carry this uncompressed NV12 pair at 20 Hz, although the
compact synthetic workload has a much smaller bandwidth budget. Its size alone
does not establish readiness of continuous calibrated preprocessing on 3X.

For a 7,471,104-byte NV12 pair, ideal serialization is 59.77 ms at 1 Gbps,
23.91 ms at 2.5 Gbps and 11.95 ms at 5 Gbps. These are lower bounds, not predicted
latencies: USB adapters, encoding, TCP, authentication, copies, scheduling,
preprocessing and inference are excluded. No Ethernet result is claimed.

The short measured 470 Mbps NCM upload is not Chestnut throughput or the maximum
of every 5 Gbps USB implementation. The inspected AGNOS configuration includes
RTL8152-family and AX88179 drivers, but that does not verify installed-kernel
support, newer chipset revisions, adapter power draw or any 2.5/5GbE configuration.

## Sources and revision scope

Reviewed AGNOS kernel: `b53ae06564df3d34f6914c5d0699594913b16626`, not established
as identical to the installed binary. No power workaround or Ethernet deployment
was validated by this source review.

- [Official modeld](https://github.com/commaai/openpilot/blob/49bbba371c29c5612e69b8ea9f906e151410e8bc/openpilot/selfdrive/modeld/modeld.py)
- [Model device selection](https://github.com/commaai/openpilot/blob/49bbba371c29c5612e69b8ea9f906e151410e8bc/openpilot/selfdrive/modeld/helpers.py)
- [NV12 layout](https://github.com/commaai/openpilot/blob/49bbba371c29c5612e69b8ea9f906e151410e8bc/openpilot/system/camerad/cameras/nv12_info.py)
- [Chestnut monitoring](https://github.com/commaai/openpilot/blob/49bbba371c29c5612e69b8ea9f906e151410e8bc/openpilot/system/hardware/chestnut/monitoring.py)
- [Bridge power and flashing](https://github.com/commaai/openpilot/blob/49bbba371c29c5612e69b8ea9f906e151410e8bc/openpilot/system/hardware/chestnut/flash.py)
- [Input suspend](https://github.com/commaai/agnos-kernel-sdm845/blob/b53ae06564df3d34f6914c5d0699594913b16626/drivers/power/supply/qcom/smb-lib.c#L2030)
- [AGNOS USB network configuration](https://github.com/commaai/agnos-kernel-sdm845/blob/b53ae06564df3d34f6914c5d0699594913b16626/arch/arm64/configs/tici_defconfig#L1815)
