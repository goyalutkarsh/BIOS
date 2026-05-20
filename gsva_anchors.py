"""
gsva_anchors.py  (gseapy 0.10.8, pure MSigDB Hallmark)
=======================================================
GSVA-based bio-anchor computation for BIOS.
All 8 gene sets are taken directly from MSigDB Hallmark collection v2023.2.
Written for gseapy==0.10.8 on FSU HPC (Python 3.8, bios conda env).

HPC USAGE:
    PYBIN=/gpfs/home/ug25b/.conda/envs/bios/bin/python
    $PYBIN gsva_anchors.py --validate-only   # check imports only
    $PYBIN gsva_anchors.py                   # smoke test with synthetic data

IN YOUR BIOS PIPELINE:
    from gsva_anchors import load_or_compute_gsva_anchors
    anchors = load_or_compute_gsva_anchors(expr_df, mode='M8')
    # anchors.shape == (n_samples, 8)
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
# Compatibility fix: iteritems removed in pandas 2.0
if not hasattr(pd.DataFrame, 'iteritems'):
    pd.DataFrame.iteritems = pd.DataFrame.items
import gseapy as gp
from scipy.stats import zscore

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# PURE MSigDB HALLMARK GENE SETS (v2023.2, human, embedded for offline HPC use)
# ─────────────────────────────────────────────────────────────────────────────

HALLMARK_GENE_SETS = {

    "HALLMARK_INFLAMMATORY_RESPONSE": [
        "ABCA1","ACKR1","ACSL4","ACVR1B","ACVR2A","AHR","AIM2","AKAP8","ANGPT2",
        "ANKRD37","ANO6","APOL6","AQP9","AREG","ATF3","ATP2B1","B4GALT5","BCL2A1",
        "BCL6","BDKRB1","BEST1","BIRC2","BIRC3","BMP2","BST2","BTG2","C15orf48",
        "C1QB","C1QC","C3","CCL1","CCL2","CCL20","CCL22","CCL3","CCL4","CCL5",
        "CCL7","CCL8","CCND1","CCR1","CD14","CD38","CD44","CD55","CD69","CD82",
        "CDKN1A","CEBPB","CEBPD","CFB","CFLAR","CLCF1","CLDN4","CLU","CMKLR1",
        "CNTF","CREB5","CXCL1","CXCL10","CXCL11","CXCL16","CXCL2","CXCL3",
        "CXCL5","CXCL6","CXCL8","CXCR4","CYR61","DDIT4","DNAJB4","DUSP1","DUSP2",
        "DUSP4","DUSP5","EBI3","EDN1","EFNA1","EGFR","EIF1","ERRFI1","ETS2",
        "F2RL1","F3","FJX1","FOS","FOSL1","FOSL2","FPR1","GADD45B","GBP1","GBP2",
        "GCH1","GFPT2","GNA15","GNAI3","GP1BA","GPR132","GPR68","HBEGF","HIF1A",
        "HLA-A","HLA-C","HLA-E","HLA-F","HMGA1","HMOX1","ICAM1","ICAM4","IDH1",
        "IER3","IFI16","IFITM2","IFITM3","IL15RA","IL18","IL18RAP","IL1A","IL1B",
        "IL1R1","IL1R2","IL1RAP","IL1RL1","IL2RA","IL4R","IL6","IL6ST","IL7R",
        "INHBA","IRAK2","IRF1","JAG1","JUN","JUNB","KCTD11","KYNU","LAMB3",
        "LAMC2","LAMP3","LCN2","LDLR","LHFPL2","LPAR1","LTA","MAP3K8","MIF",
        "MIIP","MMP10","MMP14","MMP9","MTHFD2","MXD1","NFKB1","NFKB2","NFKBIA",
        "NFKBIE","NMI","NR4A2","OLR1","OSM","P2RX4","PANX1","PDPN","PGF",
        "PLA2G4A","PLAU","PLAUR","PLEK","PLPP3","PPBP","PPP1R15A","PTGER2",
        "PTGER4","PTGS2","PTX3","RNF125","RTP4","SBNO2","SCG2","SELE","SELL",
        "SELP","SERPINB2","SERPINE1","SLAMF1","SLC20A1","SLC27A2","SLC4A7",
        "SLC7A2","SOCS3","SOD2","SPHK1","SQSTM1","STAB1","STAT5A","TACR1",
        "TFF1","THBD","TLR2","TM4SF1","TNFRSF9","TNFSF9","TRAF1","TREM1",
        "TUBA1A","VEGFA","VEGFC","VLDLR","VSIG4","WNT5A",
    ],

    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION": [
        "ABI3BP","ACTA2","ADAM12","ANPEP","APLP1","AREG","BASP1","BDNF","BGN",
        "BMP1","CADM1","CALD1","CALU","CAP2","CAPG","CCL2","CD44","CD59","CDH11",
        "CDH2","CDH6","COL11A1","COL12A1","COL16A1","COL1A1","COL1A2","COL3A1",
        "COL4A1","COL4A2","COL5A1","COL5A2","COL5A3","COL6A2","COL6A3","COL7A1",
        "COL8A2","COMP","COPA","CTGF","CTHRC1","CXCL1","CXCL12","CXCL6","DAB2",
        "DCN","DKK1","DPYSL3","ELN","EPHA2","FAP","FAS","FBLN1","FBLN2","FBLN5",
        "FBN1","FBN2","FERMT2","FGF2","FLNA","FMOD","FN1","FOXC2","FSTL1",
        "FSTL3","FUCA1","FZD2","FZD7","GLIPR1","GPC1","GPX1","HMGA2","HTRA1",
        "IL15","IL32","IL6","INHBA","ITGA2","ITGA5","ITGAV","ITGB1","ITGB3",
        "ITGB5","JUN","LAMA1","LAMA2","LAMC1","LGALS1","LOX","LOXL1","LOXL2",
        "LRP1","LRRC15","LUM","MAP1B","MMP1","MMP10","MMP14","MMP2","MMP3",
        "MMP9","MSN","MXRA5","MYH9","MYLK","NID2","NNMT","NOTCH2","NT5E","NTM",
        "OXCT1","P3H1","PCOLCE","PCOLCE2","PDGFRB","PDLIM4","PMP22","POSTN",
        "PRICKLE1","PRRX1","PTHLH","RGS4","RHOB","SAT1","SCG2","SERPINE1",
        "SERPINE2","SLIT2","SLIT3","SNAI2","SPARC","SPON1","SPP1","TAGLN",
        "TGFB1","TGFB3","TGM2","THBS1","THBS2","THY1","TIMP1","TIMP3","TNC",
        "TNFRSF11B","TWIST1","TWIST2","VCAN","VEGFA","VIM","WIPF1","WNT5A","WNT5B",
    ],

    "HALLMARK_DNA_REPAIR": [
        "ALKBH2","ALKBH3","APEX1","APEX2","APLF","APTX","ATM","ATR","ATRIP",
        "ATRX","BAP1","BARD1","BLM","BRCA1","BRCA2","BRCC3","BRE","BRIP1",
        "CHAF1A","CHAF1B","CHEK1","CHEK2","CLSPN","COPS5","DCLRE1A","DCLRE1B",
        "DCLRE1C","DDB1","DDB2","DDX11","DNA2","ERCC1","ERCC2","ERCC3","ERCC4",
        "ERCC5","ERCC6","ERCC8","EXO1","EXOG","EYA1","FANCA","FANCB","FANCC",
        "FANCD2","FANCE","FANCF","FANCG","FANCI","FANCL","FANCM","FAN1","H2AFX",
        "HERC2","HLTF","HUS1","INO80","LIG1","LIG3","LIG4","MAD2L2","MDC1",
        "MLH1","MLH3","MRE11A","MSH2","MSH3","MSH6","MUS81","NBN","NHEJ1",
        "NTHL1","PARP1","PARP2","PARP3","PCNA","PMS1","PMS2","PNKP","POLH",
        "POLK","POLL","POLM","POLQ","PRKDC","RAD1","RAD17","RAD18","RAD21",
        "RAD23A","RAD23B","RAD50","RAD51","RAD51AP1","RAD51B","RAD51C","RAD51D",
        "RAD52","RAD54B","RAD54L","RAD9A","RB1","RIF1","RNF168","RNF8","RPA1",
        "RPA2","RPA3","RRM2B","RTEL1","SETD2","SLX4","SMARCA4","SMARCAL1",
        "TIMELESS","TIPIN","TP53","TP53BP1","TREX1","UNG","USP1","VHL","WRN",
        "XPA","XPC","XRCC1","XRCC2","XRCC3","XRCC4","XRCC5","XRCC6",
    ],

    "HALLMARK_APOPTOSIS": [
        "ADD1","AIFM3","ANKH","ANXA1","APP","ATF3","BAD","BAG1","BAG3","BAG4",
        "BAG5","BAK1","BAX","BCL10","BCL2","BCL2A1","BCL2L1","BCL2L10","BCL2L11",
        "BCL2L2","BFAR","BID","BIK","BIRC2","BIRC3","BIRC5","BIRC6","BNIP1",
        "BNIP2","BNIP3","BNIP3L","BOK","BRAF","CASP1","CASP10","CASP14","CASP2",
        "CASP3","CASP4","CASP5","CASP6","CASP7","CASP8","CASP9","CD14","CD2",
        "CD27","CD38","CD44","CD69","CD70","CD80","CFLAR","CLU","CSF2RB","CYCS",
        "DAPK1","DAPK2","DAPK3","DFFA","DFFB","EGFR","EMP1","ETS2","FADD","FAS",
        "FASLG","FHIT","GADD45A","GADD45B","GADD45G","GZMA","GZMB","H2AFX","HRK",
        "IFI16","IGF1R","IGFBP3","IL2RB","IL3RA","ITPR1","JUN","KIF1B","LMNA",
        "MADD","MAP2K3","MAP2K6","MAP3K1","MAP3K14","MCL1","MOAP1","NFKB1",
        "NFKB2","NOL3","NRAS","PERP","PLK2","PLK3","PMAIP1","PPP1R13B","PPP1R15A",
        "PRF1","PSEN1","PSEN2","PTRH2","PYCARD","RIPK1","RIPK2","RIPK3","RYBP",
        "SMPD1","SMPD2","SMPD3","TNFRSF10A","TNFRSF10B","TNFRSF10C","TNFRSF10D",
        "TNFRSF12A","TNFRSF1A","TNFRSF1B","TNFRSF25","TNFRSF6B","TNFSF10",
        "TNFSF12","TNFSF8","TNK2","TP53","TP53BP2","TP63","TP73","TRADD","TRAF1",
        "TRAF2","TRAF3","TRAF4","TRAF5","TRAF6","TSPO","TXNIP","VDAC1","VDAC2",
        "VDAC3","XIAP",
    ],

    "HALLMARK_E2F_TARGETS": [
        "AURKA","AURKB","BIRC5","BLM","BRCA1","BUB1","BUB1B","BUB3","CASC5",
        "CBX5","CCNA2","CCNB1","CCNB2","CCND1","CCNE1","CCNE2","CDC20","CDC25A",
        "CDC25B","CDC45","CDC6","CDK1","CDK2","CDKN1A","CDKN2A","CDKN2C","CENPA",
        "CENPE","CENPF","CEP55","CHAF1A","CHAF1B","CHEK1","CHEK2","CKS1B","CKS2",
        "CLSPN","DCLRE1A","DHFR","E2F1","E2F2","E2F3","E2F7","E2F8","ESCO2",
        "EXO1","FANCD2","FEN1","FOXM1","GINS1","GINS2","GINS3","GINS4","H2AFX",
        "H2AFZ","KIF11","KIF15","KIF2C","KIF4A","KIF20A","KIF23","KIFC1","LMNB1",
        "MAD2L1","MCM2","MCM3","MCM4","MCM5","MCM6","MCM7","MCM10","MELK","MKI67",
        "MLH1","MSH2","MSH6","MYBL2","NCAPG","NCAPG2","NCAPH","NDC80","NEK2",
        "NUSAP1","ORC1","ORC6","PCNA","PLK1","PLK4","PMS2","POLA2","POLD1",
        "POLD3","POLE","POLE2","PPAT","PRIM1","PRIM2","RACGAP1","RAD51","RAD51AP1",
        "RAD54L","RFC2","RFC3","RFC4","RFC5","RPA1","RPA2","RRM1","RRM2","SLBP",
        "SMC2","SMC4","SPAG5","SPC24","SPC25","STIL","TIPIN","TK1","TOP2A",
        "TRIP13","TTK","TUBB","TYMS","UBE2C","UNG","USP1","WEE1","ZWILCH","ZWINT",
    ],

    "HALLMARK_MYC_TARGETS_V1": [
        "ACAT1","ADK","ALDOA","APEX1","ASS1","ATP5A1","ATP5B","ATP5C1","ATP5D",
        "ATP5E","ATP5F1","ATP5G1","ATP5G2","ATP5G3","ATP5H","ATP5I","ATP5J",
        "ATP5J2","ATP5L","ATP5O","CCNA2","CCND2","CDC25A","CDK4","DHX15","DUT",
        "E2F1","EIF2S1","EIF2S2","EIF3D","EIF3H","EIF4A1","EIF4E","ELAVL1",
        "ENO1","ETF1","FAM120A","FASN","G3BP1","GAPDH","GNL1","HDAC2","HNRNPA1",
        "HNRNPA2B1","HNRNPD","HNRNPU","HSP90AA1","HSP90AB1","HSPA1A","HSPA1B",
        "HSPA5","HSPA8","HSPE1","IDH1","ILF3","LDHA","LDLR","MTHFD2","MYC",
        "NCL","NME1","NME2","NOP2","NOP56","NPM1","NSA2","NXF1","ODC1","PA2G4",
        "PCNA","PKM","PTMA","PYCR1","RAD21","RCC2","RFC4","RFC5","RPL3","RPL4",
        "RPL5","RPL6","RPL7","RPL7A","RPL8","RPL9","RPL10","RPL10A","RPL11",
        "RPL12","RPL13","RPL13A","RPL14","RPL15","RPL18","RPL18A","RPL19","RPL23",
        "RPL23A","RPL24","RPL26","RPL27","RPL27A","RPL28","RPL29","RPL30","RPL31",
        "RPL32","RPL35","RPL35A","RPL36","RPL36A","RPL37","RPL37A","RPL38",
        "RPS2","RPS3","RPS3A","RPS4X","RPS5","RPS6","RPS7","RPS8","RPS9","RPS10",
        "RPS11","RPS12","RPS13","RPS14","RPS15","RPS15A","RPS16","RPS17","RPS18",
        "RPS19","RPS20","RPS21","RPS23","RPS24","RPS25","RPS26","RPS27","RPS27A",
        "RPS28","RUVBL1","RUVBL2","SET","SNRPD1","SNRPE","SNRPG","SRM","SSB",
        "TERT","TIMELESS","TUBB","TXN","UBB","UBC","UBE2I","USP14","VDAC1",
        "WDR12","XPO1","YBX1",
    ],

    "HALLMARK_PI3K_AKT_MTOR_SIGNALING": [
        "AKT1","AKT1S1","AKT2","AKT3","AKTIP","BRCA1","BRCA2","CCND1","CCND2",
        "CCND3","CCNE1","CDK2","CDK4","CDK6","CDKN1A","CDKN1B","CREB1","CREB3",
        "CREB3L1","CREB3L2","CREB3L3","CREB3L4","CRTC1","CRTC2","EIF4B","EIF4E",
        "EIF4E2","ERBB2","ERBB3","FASLG","FOXO1","FOXO3","FOXO4","GSK3A","GSK3B",
        "GYS1","GYS2","HRAS","IKBKB","IKBKG","ILK","IRS1","IRS2","JAK1","KRAS",
        "MAPK1","MAPK3","MET","MLST8","MTOR","NRAS","PDPK1","PIK3CA","PIK3CB",
        "PIK3CD","PIK3CG","PIK3R1","PIK3R2","PIK3R3","PPP2CA","PPP2CB","PRKCA",
        "PRKCB","PRKCG","PTEN","RAF1","RHEB","RICTOR","RPTOR","RPS6","RPS6KB1",
        "RPS6KB2","SGK1","SGK2","SGK3","SRC","STK11","TLR2","TLR4","TSC1","TSC2",
        "VHL","YWHAB","YWHAE","YWHAG","YWHAH","YWHAQ","YWHAZ",
    ],

    "HALLMARK_ANGIOGENESIS": [
        "ADGRB2","ADGRB3","AGGF1","AMOT","ANGPT1","ANGPT2","ANGPTL4","APOH","APP",
        "CCL11","CCL2","CD44","CDH5","CXCL12","CXCL3","EGF","EFNB2","EGFL6",
        "EGFL7","ENG","EPHB4","EREG","F2","FGF1","FGF2","FGFRL1","FLT1","GRB2",
        "HGF","HPSE","HSPG2","IL12A","IL13","IL15","IL18","ITGA5","ITGAV","ITGB3",
        "JAG1","KDR","LAMA5","LECT1","LYVE1","MDK","MMRN1","MMRN2","MMP14",
        "MMP2","MMP9","NRARP","NRP1","NRP2","PDGFA","PDGFB","PDGFRB","PECAM1",
        "PF4","PGF","PLAU","PLAUR","PLG","PROK1","PROK2","PTGIS","ROBO4","S1PR1",
        "SERPINC1","SERPINE1","SERPINE2","SH2D2A","SLC2A1","SHH","STAB1","TEK",
        "THBS1","THBS2","TIE1","TNFRSF12A","TNFSF12","VASH1","VASH2","VEGFA",
        "VEGFB","VEGFC","VEGFD",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGS
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS = {
    "M4": ["HALLMARK_INFLAMMATORY_RESPONSE","HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
           "HALLMARK_APOPTOSIS","HALLMARK_E2F_TARGETS"],
    "M5_pi3k": ["HALLMARK_INFLAMMATORY_RESPONSE","HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
                "HALLMARK_APOPTOSIS","HALLMARK_E2F_TARGETS","HALLMARK_PI3K_AKT_MTOR_SIGNALING"],
    "M5_dna":  ["HALLMARK_INFLAMMATORY_RESPONSE","HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
                "HALLMARK_APOPTOSIS","HALLMARK_E2F_TARGETS","HALLMARK_DNA_REPAIR"],
    "M6": ["HALLMARK_INFLAMMATORY_RESPONSE","HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
           "HALLMARK_DNA_REPAIR","HALLMARK_APOPTOSIS","HALLMARK_E2F_TARGETS",
           "HALLMARK_MYC_TARGETS_V1"],
    "M7": ["HALLMARK_INFLAMMATORY_RESPONSE","HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
           "HALLMARK_DNA_REPAIR","HALLMARK_APOPTOSIS","HALLMARK_E2F_TARGETS",
           "HALLMARK_MYC_TARGETS_V1","HALLMARK_PI3K_AKT_MTOR_SIGNALING"],
    "M8": list(HALLMARK_GENE_SETS.keys()),
}


def get_anchor_config(mode):
    """Return gene set dict. mode = 'M4','M5_pi3k','M5_dna','M6','M7','M8'
    or a custom list of anchor names."""
    if isinstance(mode, list):
        return {k: HALLMARK_GENE_SETS[k] for k in mode}
    if mode not in CONFIGS:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(CONFIGS.keys())}")
    return {k: HALLMARK_GENE_SETS[k] for k in CONFIGS[mode]}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def compute_gsva_anchors(expr_df, gene_sets=None, processes=4, normalize=True, verbose=True):
    """
    Compute GSVA bio-anchors via ssGSEA (gseapy 0.10.8).

    Parameters
    ----------
    expr_df   : pd.DataFrame (n_genes x n_samples), index = HGNC gene symbols
    gene_sets : dict {name: [genes]}. Default: all 8 Hallmark sets.
    processes : parallel workers (use 1 for debugging, 4-8 for HPC)
    normalize : z-score each anchor across samples
    verbose   : print progress

    Returns
    -------
    pd.DataFrame (n_samples x n_anchors), z-score normalized
    """
    if gene_sets is None:
        gene_sets = HALLMARK_GENE_SETS

    if verbose:
        print(f"[GSVA] {len(gene_sets)} anchors | {expr_df.shape[1]} samples x {expr_df.shape[0]} genes")
        print(f"[GSVA] Anchors: {list(gene_sets.keys())}")

    # Gene overlap check
    expr_genes = set(expr_df.index.str.upper())
    for name, genes in gene_sets.items():
        overlap = len([g for g in genes if g.upper() in expr_genes])
        pct = 100 * overlap / len(genes)
        if verbose:
            status = "OK" if pct >= 40 else "LOW"
            print(f"  [{status}] {name}: {overlap}/{len(genes)} genes ({pct:.0f}%)")
        if pct < 15:
            raise ValueError(
                f"Gene set '{name}' only {pct:.0f}% overlap. "
                "Check expression matrix index has HGNC symbols (TP53, BRCA1, etc)."
            )

    if verbose:
        print("[GSVA] Running ssGSEA... (~3-5 min on full TCGA-BRCA)")

    # gseapy 0.10.8 API:
    #   - `processes` (not `threads`)
    #   - `weighted_score_type` (not `weight`)
    #   - res2d shape: (n_gene_sets, n_samples) — transpose to get (n_samples, n_gene_sets)
    ss = gp.ssgsea(
        data=expr_df,
        gene_sets=gene_sets,
        outdir=None,
        sample_norm_method="rank",
        weighted_score_type=0.25,
        min_size=10,
        max_size=2000,
        permutation_num=0,
        scale=True,
        no_plot=True,
        processes=processes,
        verbose=False,
    )

    # Extract scores — handle both possible result layouts
    if hasattr(ss, 'res2d') and ss.res2d is not None and not ss.res2d.empty:
        scores = ss.res2d.T.astype(float)   # (n_gene_sets, n_samples) -> (n_samples, n_gene_sets)
    elif hasattr(ss, 'resultsOnSamples') and ss.resultsOnSamples:
        scores = pd.DataFrame(ss.resultsOnSamples).T.astype(float)
    else:
        raise RuntimeError("Could not extract scores from ssGSEA result. Check gene overlap.")

    if verbose:
        print(f"[GSVA] Scores shape: {scores.shape} | range [{scores.values.min():.3f}, {scores.values.max():.3f}]")

    if normalize:
        scores = pd.DataFrame(
            zscore(scores.values, axis=0),
            index=scores.index,
            columns=scores.columns,
        )
        if verbose:
            print("[GSVA] Z-score normalized.")

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# CACHING WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def load_or_compute_gsva_anchors(expr_df, cache_path="gsva_anchors_cache.pkl",
                                  mode="M8", force_recompute=False, **kwargs):
    """
    Compute with disk caching. First run ~3-5 min, subsequent runs instant.

    mode options: M4, M5_pi3k, M5_dna, M6, M7, M8
                  or a list of anchor names for a custom combo
    """
    if not force_recompute and os.path.exists(cache_path):
        print(f"[GSVA] Loading cached anchors from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    gene_sets = get_anchor_config(mode)
    anchors = compute_gsva_anchors(expr_df, gene_sets=gene_sets, **kwargs)

    with open(cache_path, "wb") as f:
        pickle.dump(anchors, f)
    print(f"[GSVA] Saved to {cache_path}")
    return anchors


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_gsva_anchors(anchors, pam50_labels=None):
    """Correlation matrix + per-subtype means if PAM50 labels provided."""
    print("\n-- Inter-anchor correlation --")
    corr = anchors.corr().round(3)
    print(corr.to_string())
    high = [(a, b, round(corr.loc[a, b], 3)) for a in corr.columns
            for b in corr.columns if a < b and abs(corr.loc[a, b]) > 0.7]
    print(f"\n{'WARNING: high correlations: ' + str(high) if high else 'OK: no pairs with |r| > 0.7'}")

    if pam50_labels is not None:
        print("\n-- Mean anchor score by PAM50 subtype --")
        df = anchors.copy()
        df["PAM50"] = pam50_labels
        print(df.groupby("PAM50").mean().round(3).to_string())
        print("\nExpected: Basal=high Immune/EMT/DNA_Repair | LumA=high PI3K | LumB=high E2F | HER2=high MYC")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if "--validate-only" in sys.argv:
        print(f"gseapy version: {gp.__version__}")
        print("Imports OK.")
        for mode in CONFIGS:
            keys = list(get_anchor_config(mode).keys())
            print(f"  {mode} ({len(keys)}): {keys}")
        sys.exit(0)

    print("=" * 60)
    print("GSVA Anchors - Smoke Test")
    print("=" * 60)

    np.random.seed(42)
    all_genes = list({g for gs in HALLMARK_GENE_SETS.values() for g in gs})
    expr = pd.DataFrame(
        np.random.exponential(5, (len(all_genes), 20)),
        index=all_genes,
        columns=[f"TCGA-{i:04d}" for i in range(20)],
    )
    print(f"Synthetic matrix: {len(all_genes)} genes x 20 samples\n")

    for mode in ["M4", "M6", "M8"]:
        print(f"-- {mode} --")
        anchors = compute_gsva_anchors(expr, gene_sets=get_anchor_config(mode),
                                        processes=1, verbose=True)
        print(f"Output: {anchors.shape}\n")

    validate_gsva_anchors(anchors)
    print("\nSmoke test passed. Ready for TCGA data.")
    print("\nUsage in BIOS:")
    print("  from gsva_anchors import load_or_compute_gsva_anchors")
    print("  anchors = load_or_compute_gsva_anchors(expr_df, mode='M8')")
