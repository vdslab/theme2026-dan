# Local case study for VisNotes

## Scope

Internal draft using the existing original input. Public use of the data and derived figures is pending a decision. The example follows **Leiomyosarcoma** (vertex ID `198`), selected by constrained incidence, then by total incidence and original input order. All 22 incident hyperedges are displayed in each panel, using their layers from the full layout. The selection depends on input incidence and constraints.

## Reading Fig. 1

The planar panel places the selected relationships at Z=0. The two stacked panels show input-order first-fit from Qu et al. (2017, Section 6) and the proposed arrangement for seed 0. Both stacked views use a 45° azimuth, a 35° elevation, and one layer step equal to 1/100 of the full input XY span. All panels use the same display scale. The marker identifies the selected vertex, with a vertical segment connecting its layers in the two stacked panels. Degenerate boundaries are marked as dashed outlines when present.

| Quantity | First-fit | Proposed |
| --- | ---: | ---: |
| Full-layout layers | 45 | 39 |
| Full-layout overlap violations | 0 | 0 |
| Full-layout total pillar length | 9,492 | 5,891 |
| Selected vertex span | 37 | 33 |

The local subset contains 212 pairs of hyperedges that overlap in XY. This view supports inspection of the selected vertex and its incident layers at a fixed scale. Reading accuracy and task time have yet to be assessed.

[Teaser](fig1-teaser.svg) · [Figure provenance](figure-manifest.json)
