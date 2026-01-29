# Data Availability

This document describes all datasets used in this project.

This file was edited by AI, if you think there has been an omission or error, please contact `evansna@ohsu.edu`. 

---

## CCLE (Cancer Cell Line Encyclopedia) Data

### CCLE Expression Data (mRNA)
**Description:** TPM log(p+1) expression for human protein-coding genes  
**Source:** https://depmap.org/portal/ (DepMap Portal)  
**File:** `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv`  
**License:** Research use only; commercial use requires license from Broad Institute  
**Citation:** Ghandi M, et al. (2019) Next-generation characterization of the Cancer Cell Line Encyclopedia. *Nature* 569, 503–508. https://doi.org/10.1038/s41586-019-1186-3  
**Used in:** `notebooks/07_RNA_trametinib_resp.ipynb`, `notebooks/09_regulator_dusp6_expr_correlation_breast.ipynb`

### CCLE miRNA Expression Data
**Description:** miRNA expression data for CCLE cell lines (GCT format)  
**Source:** https://depmap.org/portal/ (DepMap Portal)  
**File:** `CCLE_miRNA_20180525.gct`  
**License:** Research use only; commercial use requires license from Broad Institute  
**Citation:** Ghandi M, et al. (2019) Next-generation characterization of the Cancer Cell Line Encyclopedia. *Nature* 569, 503–508. https://doi.org/10.1038/s41586-019-1186-3  
**Used in:** `notebooks/06_miRNA_trametinib_resp.ipynb`, `notebooks/09_regulator_dusp6_expr_correlation_breast.ipynb`

### CCLE Somatic Mutations
**Description:** Somatic mutations for CCLE cell lines  
**Source:** https://depmap.org/portal/ (DepMap Portal)  
**File:** `OmicsSomaticMutations.csv`  
**License:** Research use only; commercial use requires license from Broad Institute  
**Citation:** Ghandi M, et al. (2019) Next-generation characterization of the Cancer Cell Line Encyclopedia. *Nature* 569, 503–508. https://doi.org/10.1038/s41586-019-1186-3  
**Used in:** `notebooks/08_MUT_trametinib_resp.ipynb`, `notebooks/old/02_dxdt_explanations.ipynb`, `notebooks/_explain_mut.ipynb`

### CCLE Cell Line Information
**Description:** Cell line metadata (DepMap_ID, CCLE_Name, lineage, primary_disease, Subtype)  
**Source:** https://depmap.org/portal/ (DepMap Portal)  
**File:** `ccle_info.txt`  
**License:** Research use only; commercial use requires license from Broad Institute  
**Citation:** Ghandi M, et al. (2019) Next-generation characterization of the Cancer Cell Line Encyclopedia. *Nature* 569, 503–508. https://doi.org/10.1038/s41586-019-1186-3  
**Used in:** `notebooks/06_miRNA_trametinib_resp.ipynb`, `notebooks/07_RNA_trametinib_resp.ipynb`, `notebooks/08_MUT_trametinib_resp.ipynb`, `notebooks/09_regulator_dusp6_expr_correlation_breast.ipynb`

---

## GDSC (Genomics of Drug Sensitivity in Cancer) Data

### Sanger Dose-Response Data
**Description:** Drug dose-response data (GDSC1/GDSC2) including IC50/AUC values  
**Source:** https://www.cancerrxgene.org/ (Wellcome Sanger Institute)  
**File:** `sanger-dose-response.csv`  
**License:** Research and educational use only  
**Citation:** Yang W, et al. (2013) Genomics of Drug Sensitivity in Cancer (GDSC): a resource for therapeutic biomarker discovery in cancer cells. *Nucleic Acids Res.* 41(Database issue): D955-D961. https://doi.org/10.1093/nar/gks1111  
**Used in:** `notebooks/06_miRNA_trametinib_resp.ipynb`, `notebooks/07_RNA_trametinib_resp.ipynb`, `notebooks/08_MUT_trametinib_resp.ipynb`

---

## DepMap Data

### DepMap Cell Line Metadata
**Description:** DepMap cell line information (DepMap_ID, CCLE_Name)  
**Source:** https://depmap.org/portal/  
**File:** `DepMap-2019q1-celllines_v2.csv`  
**License:** Research use only; commercial use requires license from Broad Institute  
**Citation:** Ghandi M, et al. (2019) Next-generation characterization of the Cancer Cell Line Encyclopedia. *Nature* 569, 503–508. https://doi.org/10.1038/s41586-019-1186-3  
**Used in:** `notebooks/_prism.ipynb`, `notebooks/old/02_dxdt_explanations.ipynb`, `notebooks/_explain_mut.ipynb`

---

## LINCS/CLUE Data

### LINCS Compound Information
**Description:** CLUE compound metadata (inchi_key, pert_id, target information)  
**Source:** https://clue.io/ (Connectivity Map)  
**File:** `compoundinfo_beta.txt`  
**License:**  
**Citation:** Subramanian A, et al. (2017) A Next Generation Connectivity Map: L1000 Platform and the First 1,000,000 Profiles. *Cell* 171(6), 1437-1452.e17. https://doi.org/10.1016/j.cell.2017.10.049  
**Used in:** `workflow/scripts/make_bio_network.py`, `notebooks/_dev.ipynb`, `notebooks/old/02_dxdt_explanations.ipynb`

### LINCS Cell Information
**Description:** LINCS cell line metadata (cell_iname, ccle_name)  
**Source:** https://clue.io/ (Connectivity Map)  
**File:** `cellinfo_beta.txt`  
**License:**  
**Citation:** Subramanian A, et al. (2017) A Next Generation Connectivity Map: L1000 Platform and the First 1,000,000 Profiles. *Cell* 171(6), 1437-1452.e17. https://doi.org/10.1016/j.cell.2017.10.049  
**Used in:** `notebooks/_prism.ipynb`, `notebooks/old/02_dxdt_explanations.ipynb`, `notebooks/_explain_mut.ipynb`, `notebooks/_explain_gseas.ipynb`

---

## Biological Network Databases

### Targetome Extended
**Description:** Drug-target interactions with affinity data (Kd/Ki values). Filtered for Kd/Ki ≤ 1000 nM, direct binding assays only, excludes ">" relations  
**Source:** Targetome Extended database  
**File:** `targetome_extended-01-23-25.csv`  
**License:**  
**Citation:** Blucher AS, Choonoo G, Kulesz-Martin M, Wu G, McWeeney SK. Evidence-Based Precision Oncology with the Cancer Targetome. Trends Pharmacol Sci. 2017 Dec;38(12):1085-1099. doi: 10.1016/j.tips.2017.08.006. Epub 2017 Sep 27. PMID: 28964549; PMCID: PMC5759325.  
**Used in:** `workflow/scripts/make_bio_network.py`, `notebooks/_checks.ipynb`, `notebooks/_proc.ipynb`

### OmniPath
**Description:** Protein-protein interactions, transcription factor binding, and regulatory networks  
**Source:** https://omnipathdb.org/ (accessed via pypath-omnipath Python package)  
**License:** Varies by resource (academic license default)  
**Citation:** Türei D, et al. (2016) OmniPath: guidelines and gateway for literature-curated signaling pathway resources. *Nature Methods* 13, 966–967. https://doi.org/10.1038/nmeth.4077  
**Used in:** `workflow/scripts/make_bio_network.py`

### DOROTHEA
**Description:** Transcription factor-target gene interactions (confidence levels A-D)  
**Source:** https://saezlab.github.io/dorothea/ (accessed via OmniPath)  
**License:** GPL-3  
**Citation:** Garcia-Alonso L, et al. (2019) Benchmark and integration of resources for the estimation of human transcription factor activities. *Genome Research* 29, 1363-1375. https://doi.org/10.1101/gr.240663.118  
**Used in:** `workflow/scripts/make_bio_network.py`

