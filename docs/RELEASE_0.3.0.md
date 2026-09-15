# Mac Accelerator 0.3.0 — independent project preview

The Mac app now has its own non-fork source repository, separate from sunnypilot.

- Native arm64 **Mac Accelerator.app**, macOS 15+.
- Independent bundle ID, preferences and authentication-key location.
- Core ML server, conversion tooling, authenticated protocol and synthetic
  benchmarks included. No sunnypilot/tinygrad runtime imports or checkout required.
- Model importer makes independent local copies and refuses to overwrite files.
- No firmware installer, camera/modeld hook, vehicle parameter writer or control
  output. Default is localhost; optional USB-NCM remains synthetic bench-only.
- Original research branch and its 0.2.2 release remain unchanged.

## Verification on MacBook Air M2

35 unit tests passed in the new Python environment; Python lint, shell syntax,
native arm64 build and strict ad-hoc code signature verified. The native window
opened with the standalone project selected and Mac-only mode enabled.

A separate localhost process test completed 20/20 measured frames with the copied
Big Model package: mean request/response 30.32ms, maximum 33.44ms. The test uses a
500ms diagnostic deadline; this short sample is **not sustained timing, USB,
live-camera or road-use qualification**. Historical 0.2.2 M2 findings are preserved
with their original limitations. Conversion CLI startup was checked; this extraction
reuses the previously converted model and does not claim a new full conversion run.

## Setup

See the [standalone README](https://github.com/Akarow77/mac-accelerator/blob/v0.3.0/README.md).
The ZIP contains the app only, not Python, models, metadata, credentials or logs.
Prepare the project environment and import trusted matching model artifacts.
Metadata uses pickle and must never come from an untrusted source.

Ad-hoc signed, not Apple-notarized. This is an independent research preview, not an
official Apple, comma.ai or sunnypilot product. Do not disable Gatekeeper globally.
Old and new apps use separate keys but the same default server port; stop the old
server before starting this app.
