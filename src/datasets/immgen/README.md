Human annotations were generated manually. The GPT-4 conversation is attached below

## GPT-4 conversation


### Expand the each of the sample names (below) to a short sentence describing the cell type with full names. 
For context, here some information for the overall experiment
Title	RNAseq profiling of defined immunocyte subsets from human blood, healthy volunteers
Organism	Homo sapiens
Experiment type	Expression profiling by high throughput sequencing
Summary	In the context of the Human Cell Atlas, the ImmGen consortium generated reference profiles by bulk RNAseq on finely sorted populations of immunocytes from human blood, healthy volunteers.

Overall design	Specific cell populations were sorted by flow cytometry to high purity (2 rounds of sorting, 1,000 cells/sample), as biological duplicates. The cells profiled according to the ImmGen ULI low-input (SmartSeq2) protocol.

The sample names:
- B.NveIgD+27-.Bl#1
- B.MemIgD-27+38-.Bl#1
- B.TransIgD+27+.Bl#1
- ILC.NKimm.56hi16-.Bl#1
- ILC.NKmat.56lo16hi57-.Bl#1
- ILC.NKmem.56lo16hi57hi.Bl#1
- T.4Nve.CD3+4+RA+62L+.Bl#1
- T.4EffMem.CD3+4+RA-62L-.Bl#1
- T.8Nve.CD3+8+RA+62L+.Bl#1
- T.8EffMem.CD3+8+RA-62L-.Bl#1
- T.NKT.Va24+.Bl#1
- T.MAIT.4+.Bl#1
- T.MAIT.8+.Bl#1
- T.Treg.rest.Bl#1
- T.Treg.act.Bl#1
- Mo.16+.Bl#1
- Mo.14+.Bl#1
- DC.DC1.141+.Bl#1
- DC.DC5.AXL+SIGLEC6+.Bl#1
- DC.DC6.123+.Bl#1


1. B.NveIgD+27-.Bl#1: This sample represents naive B cells that express IgD but not CD27 from human blood.
2. B.MemIgD-27+38-.Bl#1: This sample represents memory B cells that do not express IgD but do express CD27 and not CD38 from human blood.
3. B.TransIgD+27+.Bl#1: This sample represents transitional B cells that express both IgD and CD27 from human blood.
4. ILC.NKimm.56hi16-.Bl#1: This sample represents immature natural killer cells from the innate lymphoid cell group with high expression of CD56 but not CD16 from human blood.
5. ILC.NKmat.56lo16hi57-.Bl#1: This sample represents mature natural killer cells from the innate lymphoid cell group with low expression of CD56, high expression of CD16, and no expression of CD57 from human blood.
6. ILC.NKmem.56lo16hi57hi.Bl#1: This sample represents memory natural killer cells from the innate lymphoid cell group with low expression of CD56, high expression of CD16, and high expression of CD57 from human blood.
7. T.4Nve.CD3+4+RA+62L+.Bl#1: This sample represents naive CD4 T cells that express CD3, CD4, CD45RA, and CD62L from human blood.
8. T.4EffMem.CD3+4+RA-62L-.Bl#1: This sample represents effector memory CD4 T cells that express CD3, CD4, but not CD45RA or CD62L from human blood.
9. T.8Nve.CD3+8+RA+62L+.Bl#1: This sample represents naive CD8 T cells that express CD3, CD8, CD45RA, and CD62L from human blood.
10. T.8EffMem.CD3+8+RA-62L-.Bl#1: This sample represents effector memory CD8 T cells that express CD3, CD8, but not CD45RA or CD62L from human blood.
11. T.NKT.Va24+.Bl#1: This sample represents NKT cells that express Va24 from human blood.
12. T.MAIT.4+.Bl#1: This sample represents MAIT cells that express CD4 from human blood.
13. T.MAIT.8+.Bl#1: This sample represents MAIT cells that express CD8 from human blood.
14. T.Treg.rest.Bl#1: This sample represents resting regulatory T cells from human blood.
15. T.Treg.act.Bl#1: This sample represents activated regulatory T cells from human blood.
16. Mo.16+.Bl#1: This sample represents monocytes that express CD16 from human blood.
17. Mo.14+.Bl#1: This sample represents monocytes that express CD14 from human blood.
18. DC.DC1.141+.Bl#1: This sample represents type 1 dendritic cells that express CD141 from human blood.
19. DC.DC5.AXL+SIGLEC6+.Bl#1: This sample represents type 5 dendritic cells that express AXL and SIGLEC6 from human blood.
20. DC.DC6.123+.Bl#1: This sample represents type 6 dendritic cells that express CD123 from human blood.

### Convert it to a json dict

Here is the JSON dictionary representation of the cell types:

```json
{
    "B.NveIgD+27-.Bl#1": "Naive B cells that express IgD but not CD27 from human blood",
    "B.MemIgD-27+38-.Bl#1": "Memory B cells that do not express IgD but do express CD27 and not CD38 from human blood",
    "B.TransIgD+27+.Bl#1": "Transitional B cells that express both IgD and CD27 from human blood",
    "ILC.NKimm.56hi16-.Bl#1": "Immature natural killer cells from the innate lymphoid cell group with high expression of CD56 but not CD16 from human blood",
    "ILC.NKmat.56lo16hi57-.Bl#1": "Mature natural killer cells from the innate lymphoid cell group with low expression of CD56, high expression of CD16, and no expression of CD57 from human blood",
    "ILC.NKmem.56lo16hi57hi.Bl#1": "Memory natural killer cells from the innate lymphoid cell group with low expression of CD56, high expression of CD16, and high expression of CD57 from human blood",
    "T.4Nve.CD3+4+RA+62L+.Bl#1": "Naive CD4 T cells that express CD3, CD4, CD45RA, and CD62L from human blood",
    "T.4EffMem.CD3+4+RA-62L-.Bl#1": "Effector memory CD4 T cells that express CD3, CD4, but not CD45RA or CD62L from human blood",
    "T.8Nve.CD3+8+RA+62L+.Bl#1": "Naive CD8 T cells that express CD3, CD8, CD45RA, and CD62L from human blood",
    "T.8EffMem.CD3+8+RA-62L-.Bl#1": "Effector memory CD8 T cells that express CD3, CD8, but not CD45RA or CD62L from human blood",
    "T.NKT.Va24+.Bl#1": "NKT cells that express Va24 from human blood",
    "T.MAIT.4+.Bl#1": "MAIT cells that express CD4 from human blood",
    "T.MAIT.8+.Bl#1": "MAIT cells that express CD8 from human blood",
    "T.Treg.rest.Bl#1": "Resting regulatory T cells from human blood",
    "T.Treg.act.Bl#1": "Activated regulatory T cells from human blood",
    "Mo.16+.Bl#1": "Monocytes that express CD16 from human blood",
    "Mo.14+.Bl#1": "Monocytes that express CD14 from human blood",
    "DC.DC1.141+.Bl#1": "Type 1 dendritic cells that express CD141 from human blood",
    "DC.DC5.AXL+SIGLEC6+.Bl#1": "Type 5 dendritic cells that express AXL and SIGLEC6 from human blood",
    "DC.DC6.123+.Bl#1": "Type 6 dendritic cells that express CD123 from human blood"
}
```

