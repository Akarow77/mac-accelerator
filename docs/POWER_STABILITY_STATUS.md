# Priority 1: power stability — 2026-09-29

**Unresolved. Direct Mac USB use in a vehicle is paused.** This project is a
research preview, not a road-use accelerator. No model output is authorized for
vehicle control by these tests. Latency improvements do not supersede this gate.

## 현재 결론

맥 연결을 뺀 뒤 재부팅이 없다는 사용자 관찰과, 공식 sunnypilot master로
복원한 뒤에도 기록된 저전압 차단을 근거로 **USB 연결을 포함한 전원 경로**를
먼저 조사합니다. 역류·맥 포트 결함·차량 전원 선택 오류 등 구체적 원인은
아직 확정하지 않았습니다. 단순히 GPU가 느려서 생긴 문제로 분류하지 않습니다.

## Observed vs inferred

| Evidence | What it supports | What it does not prove |
| --- | --- | --- |
| Whole-device boot ID changed; fresh boot dmesg reported UVLO and SMPL | Low-voltage/power-loss reset indication | Which supply, cable or internal circuit caused it |
| Repeated on restored official sunnypilot master | Not confined to the current accelerator branch | Absence of every earlier indirect effect or OS/hardware fault |
| User reports no reboot after Mac disconnect | Mac-connected topology is a leading suspect | Backfeed or PD conflict specifically |
| An earlier preserved boot had a FIFO NULL-pointer kernel panic | A separate early-boot kernel failure was captured | That it caused the initial onroad reset; second reboot had no pstore panic |
| Sampled USB voltage ranged down to about 4.39 V | Useful power-path diagnostic | A captured transient waveform or measured UVLO trip threshold |
| USB power role reported sink | The controller's reported role | Mapping of both physical ports and vehicle supply |

The experimental camera/transport clients had already ended. After vehicle
connection, diagnosis used read-only commands until the explicitly requested
restoration. Manager was normally stopped, clean master `a5f44653d` selected and
relaunched; the device was not commanded to reboot. Master restoration did not
fix the subsequent low-voltage reset. Research branch and Mac work were retained.

Private logs were saved locally before disconnection. This public report omits
boot/device identifiers, network addresses, raw journal/kernel dumps, vehicle
data, recordings and credentials. It contains no root-cause certainty claim.

## Resolution candidates, in order

1. **Non-PoE Ethernet between independently powered USB network adapters.**
   Mac and 3X each host their own adapter; no direct Mac-to-3X USB VBUS path.
   Public AGNOS kernel config includes Realtek RTL8152-family and ASIX AX88179
   drivers. Installed-kernel support, adapter revision, host-mode behavior,
   3X-side power draw and performance remain untested. TCP protocol reuse is
   plausible; the current NCM supervisor is not a drop-in Ethernet setup.
2. **Instrumented USB power/data separation on an isolated bench.**
   A programmable interface with independent VBUS/data/CC control can test
   enumeration and unwanted current paths. Compatibility with 3X and vehicle
   use must not be inferred from the interface's feature list. A generic powered
   hub is not guaranteed to block downstream power or solve source selection.
3. **Board-specific selective input control, only after source/path mapping.**
   Generic `input_suspend` is rejected: inspected official kernel code suspends
   USB and DC together. Power-role writes and current-limit changes are not
   validated fixes. No kernel patch/flash or sysfs workaround has been applied.

Wi-Fi SSH is the interim development-access option without direct USB power
coupling. It is not a qualified low-latency inference transport.

## What Chestnut actually does

Chestnut uses separate 12 V GPU power and a USB peripheral controlled by the
comma host. The bridge can enumerate on USB power alone. Current official
modeld defaults AMD GPU power to 100 W; this is **GPU power budgeting**, not
blocking USB power into a comma. It monitors voltage/current, supply faults,
PCIe link, GPU temperature and other health data. Our Mac-host/3X-gadget topology
is different. See the [source-linked investigation](USB_POWER_ISOLATION_RESEARCH_2026-09-29.md).

## Validation gates before returning to latency work

- Confirm installed kernel/config and exact port/cable/power topology. Do not
  treat an offroad parameter as proof that the device is physically detached.
- Use a detached, controlled bench and reproduce the normal-power baseline;
  record boot identity, reset reasons, power-role changes and voltage/current.
- Test adapter enumeration and link stability without inference, then sustained
  authenticated synthetic traffic. Monitor supply and local process behavior.
- Suggested initial soak: at least 30 minutes per candidate and controlled
  reconnect tests, with no resets, UVLO indications or data-integrity failures.
  This is an initial diagnostic gate, **not** an automotive safety standard or
  driving qualification. Software polling can miss fast dips; use appropriate
  instrumentation before claiming electrical stability.
- Only after that, resume camera/preprocessing/inference work. Do not disable
  UVLO/watchdogs, alter vehicle wiring or try an unverified power change on-road.

## Sources

- [AGNOS build configuration](https://github.com/commaai/agnos-builder/blob/master/build_kernel.sh)
- [Pinned kernel network configuration](https://github.com/commaai/agnos-kernel-sdm845/blob/b53ae06564df3d34f6914c5d0699594913b16626/arch/arm64/configs/tici_defconfig#L1815)
- [Pinned input-suspend implementation](https://github.com/commaai/agnos-kernel-sdm845/blob/b53ae06564df3d34f6914c5d0699594913b16626/drivers/power/supply/qcom/smb-lib.c#L2030)
- [Chestnut setup](https://comma.ai/shop/chestnut)
- [Pinned GPU power default](https://github.com/commaai/openpilot/blob/49bbba371c29c5612e69b8ea9f906e151410e8bc/openpilot/selfdrive/modeld/modeld.py#L9)
- [Independent VBUS/data/CC laboratory controls](https://acroname.com/reference/devices/usbhub3c/entities/port.html)
