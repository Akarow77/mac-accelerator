# Power stability status — 2026-09-29

Research summary: [English](RESEARCH_SUMMARY.md) · [한국어](RESEARCH_SUMMARY.ko.md).

**Unresolved. Direct Mac USB use in a vehicle is paused.** This is a research
preview, not a road-use accelerator. Historical latency measurements do not
qualify vehicle control or demonstrate an electrical fix.

## Observations

Whole-device resets occurred in the vehicle + direct Mac USB topology, including
after restoration to clean sunnypilot master. Operator observations report no
resets after disconnecting the Mac or while vehicle-detached. Supply, workload
and observation duration were not matched in a controlled A/B experiment.

| Evidence | Interpretation limit |
| --- | --- |
| Boot identity changed; subsequent dmesg reported UVLO and SMPL | Power-reset indication, not identification of a physical supply or failed part |
| Repeated on restored sunnypilot master | Not confined to the accelerator branch; OS/hardware faults are not excluded |
| USB-input samples ~4.39 V / 1.35 A and ~4.84 V / 1.30 A | Sequential driver readings, not reset-time waveforms or Mac-connector measurements |
| PD active, 5 V, 3 A ceiling; sink role | Ceiling is not actual draw; role does not map all physical supply paths |
| Preserved early-boot FIFO NULL-pointer panic matches GLINK IPC source | Not sufficient to diagnose a USB-NCM/camera fault or link it to every reset |
| Following reset had no new pstore panic | Does not establish the previous panic's cause |

The physical root cause remains unconfirmed. Source selection, electrical
transients and vehicle-associated workload/state changes have not been isolated.
Raw device logs, network addresses, identifiers and credentials are not published.

## Source-backed interpretation

- PMIC UVLO/SMPL messages decode latched reset-reason registers; they are not a
  voltage waveform or a complete chronology across multiple boots.
- `usbpd_create failed: -517` corresponds to deferred probing in the inspected
  source. Later PD activation prevents interpreting that line alone as a
  permanent PD failure.
- The policy engine prints a Type-C Source attachment message from its Sink
  state. The message names the peer, not proof that 3X is supplying the Mac.
- Generic battery `input_suspend` votes both USB and DC suspension. It is not
  a validated auxiliary-port-only switch. No power-setting workaround is supported.
- The inspected `shutdown.target` journal entries were user-session cleanup,
  not evidence of PID 1 initiating whole-system shutdown.

Installed logs report Linux 4.9.103 built 2026-09-02. Equivalence to the reviewed
public kernel commit has not been established. No board-level power-routing or
input-isolation claim follows from these source observations.

## References

- [Reset-reason decoding](https://github.com/commaai/agnos-kernel-sdm845/blob/b53ae06564df3d34f6914c5d0699594913b16626/drivers/input/misc/qpnp-power-on.c)
- [Deferred-probe errno](https://github.com/commaai/agnos-kernel-sdm845/blob/b53ae06564df3d34f6914c5d0699594913b16626/include/linux/errno.h#L18)
- [PD policy engine](https://github.com/commaai/agnos-kernel-sdm845/blob/b53ae06564df3d34f6914c5d0699594913b16626/drivers/usb/pd/policy_engine.c#L2993)
- [GLINK FIFO/IRQ implementation](https://github.com/commaai/agnos-kernel-sdm845/blob/b53ae06564df3d34f6914c5d0699594913b16626/drivers/soc/qcom/glink_smem_native_xprt.c#L314)
- [Input-suspend implementation](https://github.com/commaai/agnos-kernel-sdm845/blob/b53ae06564df3d34f6914c5d0699594913b16626/drivers/power/supply/qcom/smb-lib.c#L2030)
