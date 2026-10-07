# Studio implementation specification

Reference: `studio-concept.png`, 1505 × 1045. White background, dark ink, teal accent.
The studio is a working inspection instrument with two input views, a map, and an evidence rail.

## Tokens and families

- Canvas/surface: #ffffff; ink #182327; muted #68737e; border #d7dfe3; accent #007e79.
- Heading/logo/evidence titles: Times New Roman serif, regular. Main heading 48 px/1.05; rail title 28 px.
- Chrome/body: system sans-serif. Toolbar 14 px; captions 14 px; body 16 px; subtitle 18 px.
- Spacing: 8, 12, 16, 24, 32, 40 px. Panels 1 px border, 2 px radius, no shadows.
- Buttons: 40 px height, outlined neutral or solid teal. Navigation has teal underline.
- Logo: crosshair icon, 24 px, 1.4 px stroke. Run: outlined triangular play icon, 16 px.
- Header 54 px; content margin 28 px; main workspace approximately 70%/30%, 16 px gutter.
- Image pair: two equal views, 16 px gap, fine central divider. Map is wide, 214 px tall.
- Right rail: open evidence rows, separator, numbered action rows, export anchored at bottom.

## Allowed initial copy

TRACE; Inspect; Research; Local model; Look again. Know what changed.; Compare a reference
with an observation, then choose the next best look.; Scene; Moved component; New scene;
Run inspection; Reference; Observation; Change probability; Uncertainty; Upload your pair;
Reference image; Observation image; Evidence; Changed area; Verification; Run an inspection
to view the model evidence.; Next best look; Relight; Re-register; Inspect closely;
Recommendations appear after inspection.; Export evidence; Simulated scenes. Validate on
captured image pairs before deployment.

## Functional necessities and intentional deviations

The mock tabletop photos cannot supply verified pixel labels. The live images therefore show
actual deterministic procedural sensor scenes, retaining the two-view framing and map treatment.
The initial map shows the reference until inference returns; it never displays fabricated results.
After inspection, numbered actions become buttons with actual learned values. Simulated action
buttons acquire a new observation from the renderer. For uploads, recommendations are advisory
and require the operator to capture and upload another image. Error messages and a busy state
are necessary functional additions. Verification reports branch disagreement, never causal proof.
The Research navigation opens real experiment results in the same typography and white surface.
The Local model control opens checkpoint metadata. No invented metrics or provider API is used.

## Responsive behavior and ownership

At widths below 900 px, the rail follows the input/map panel; below 560 px, input views stack.
Header/navigation remain readable, upload controls wrap, all controls have accessible names,
and focus is visible. `App` composes `Workspace`, `Evidence`, and `Research`; API state is owned
by `useInspection`. Images and export evidence come only from the local inference API.
