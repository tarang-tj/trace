# TRACE studio verification

Final local review: 2026-10-07. The reference is the generated
[studio concept](studio-concept.png), at its native 1505 × 1045 resolution.
The implementation specification is [design-system.md](design-system.md).

## Rendered evidence and browser route

Playwright drove the production Vite build served by the local FastAPI application at
`http://127.0.0.1:8765`. The in-app Browser had no connected instance. The fallback used
installed Google Chrome through Playwright, with an isolated temporary profile; the Playwright
downloaded browser cache was unavailable. No personal browser session was used.

The final desktop captures are exactly 1505 × 1045 at device scale 1:
[initial](../artifacts/studio/desktop-initial.png) and
[inspected](../artifacts/studio/desktop-inspected.png). The concept and both rendered captures
were opened with `view_image` at original resolution in the same final QA pass. The
[mobile capture](../artifacts/studio/mobile.png) uses a 390 × 844 viewport and preserves the
full scrolling page. The [Research page](../artifacts/studio/research.png) was also visually
reviewed. Machine-readable results are in [qa.json](../artifacts/studio/qa.json).

## Comparison ledger

| Point | Final comparison and resolution |
|---|---|
| Copy | The initial application copy matches the ordered allowlist. No added, removed, renamed, or reordered text. Closed native-select alternatives are excluded because only the selected option is visible. Native file-control text comes from Chrome. |
| Layout | Header, large introductory heading, paired input views, wide map, upload row, and right evidence rail retain the reference hierarchy. The desktop rail is approximately 30% of the workspace. The initial footer fits the concept viewport. |
| Typography | Main title uses regular Times New Roman at 48 px/50.4 px; rail titles use 28 px/33.6 px. Controls use 14 px Arial. The earlier heavier, wider serif treatment was corrected. Raster-reference glyph shapes are approximated rather than claimed identical. |
| Palette | White surfaces, dark ink, fine gray borders, and teal selection/run controls match the specified treatment. No shadows or extra colored panels were introduced. |
| Spacing | 28 px content margins, a 16 px workspace gutter, equal input frames, and a 214 px map preserve the composition. An extra evidence-row bottom gap was removed to align the separators and action heading. |
| Icons and controls | Crosshair brand, monitor model control, triangular run icon, outlined action numbers, and thin tab underline retain the reference shapes. The matching favicon fixes the only final browser console error. Native select and file controls retain platform rendering. |
| Assets and data | Verified procedural image pairs replace the mock tabletop photographs. Square sensor images are letterboxed rather than cropped or stretched. After inspection, the heatmap is generated from the actual checkpoint. |
| Mobile | Input views stack, followed by uploads and evidence. Labels, values, action rows, and export remain readable. The 390 px check found no horizontal overflow. |

## Functional deviations

The reference includes an illustrative heatmap before inspection. TRACE shows the reference image
in that slot until real inference completes. Values and action order then reflect model output;
the inspected scene has 5.1% changed area and 3.2 percentage points of probe disagreement.
Verification describes computational disagreement, not proof of causality.

Action rows are executable buttons for simulated inputs. They acquire an independent new sensor
observation and rerun inference. Uploaded-image recommendations explicitly require another operator
capture. Busy/error states, signed action values, checkpoint metadata, and the Research page are
necessary additions for a working research system. Research displays the observed harmful-action
rate and the absence of a demonstrated advantage over fixed or heuristic action selection.

The shell and interaction hierarchy are faithfully verified within these documented deviations.
The concept's photographs and illustrative pre-inspection result are intentionally not reproduced;
no pixel-identical or photorealistic equivalence is claimed. No remaining material mismatch was
found in the specified shell.

## Interaction verification

The final run completed inspection, probability/error tab switching, evidence download, a simulated
follow-up with an unchanged reference and changed observation, scene reset and selection, paired
uploads, checkpoint metadata, and Research navigation. Research showed six frozen detector suites
and seven policy strategies. The exported checkpoint matched the frozen TRACE SHA-256
`876e12fcbeeae681ad8a48f79ee8cf9ccdc30877411c51bd0074cbd4a0946707`.
No JavaScript exceptions or browser console errors remained.

To repeat against the running server:

```bash
cd web
TRACE_BROWSER_EXECUTABLE='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' \
TRACE_EXPECTED_SHA=876e12fcbeeae681ad8a48f79ee8cf9ccdc30877411c51bd0074cbd4a0946707 \
TRACE_EXPECT_RESEARCH=1 node qa.mjs
```
