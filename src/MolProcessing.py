import numpy as np
import pandas as pd
import functools
import subprocess
import random
from multiprocessing import Pool

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*') #hiding the warning messages
import os
import torch
from rdkit.Chem import BRICS, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import math
from sklearn.preprocessing import MinMaxScaler
from .pubchemfp import GetPubChemFPs

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 只在有 GPU 的情况下设置
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_mol_max_length(all_smiles_y):
    max_len = []
    for i in all_smiles_y:
        try:
            nb_atom = len(Chem.MolFromSmiles(i[0]).GetAtoms())
        except:
            nb_atom = 0
        max_len.append(nb_atom)

#     dd = np.zeros(10) # Number of mol with different number of atoms [0, 0-25, 25-50, 50-75, 75-100, 100-150, 150-200, 200-250, >250]
#     for i in max_len:
#         if i == 0:
#             dd[0] += 1
#         elif i <= 25:
#             dd[1] += 1
#         elif i <= 50:
#             dd[2] += 1
#         elif i <= 75:
#             dd[3] += 1
#         elif i <= 100:
#             dd[4] += 1
#         elif i <= 150:
#             dd[5] += 1
#         elif i <= 200:
#             dd[6] += 1
#         elif i <= 250:
#             dd[7] += 1
#         else:
#             dd[8] += 1
            
    return max(max_len) 

def load_data(dataset, task_name=None, print_info=True):
    if task_name == None:
        if dataset == 'BACE':
            task_name = ['Class', 'pIC50']
        elif dataset == 'BBBP':
            task_name = ['p_np']
        elif dataset == 'SIDER':
            task_name = ['Hepatobiliary disorders','Metabolism and nutrition disorders','Product issues','Eye disorders','Investigations','Musculoskeletal and connective tissue disorders','Gastrointestinal disorders','Social circumstances','Immune system disorders','Reproductive system and breast disorders','Neoplasms benign, malignant and unspecified (incl cysts and polyps)','General disorders and administration site conditions','Endocrine disorders','Surgical and medical procedures','Vascular disorders','Blood and lymphatic system disorders','Skin and subcutaneous tissue disorders','Congenital, familial and genetic disorders','Infections and infestations','Respiratory, thoracic and mediastinal disorders','Psychiatric disorders','Renal and urinary disorders','Pregnancy, puerperium and perinatal conditions','Ear and labyrinth disorders','Cardiac disorders','Nervous system disorders','Injury, poisoning and procedural complications']
        elif dataset == 'ToxCast':
            task_name = ['ACEA_T47D_80hr_Negative','ACEA_T47D_80hr_Positive','APR_HepG2_CellCycleArrest_24h_dn','APR_HepG2_CellCycleArrest_24h_up','APR_HepG2_CellCycleArrest_72h_dn','APR_HepG2_CellLoss_24h_dn','APR_HepG2_CellLoss_72h_dn','APR_HepG2_MicrotubuleCSK_24h_dn','APR_HepG2_MicrotubuleCSK_24h_up','APR_HepG2_MicrotubuleCSK_72h_dn','APR_HepG2_MicrotubuleCSK_72h_up','APR_HepG2_MitoMass_24h_dn','APR_HepG2_MitoMass_24h_up','APR_HepG2_MitoMass_72h_dn','APR_HepG2_MitoMass_72h_up','APR_HepG2_MitoMembPot_1h_dn','APR_HepG2_MitoMembPot_24h_dn','APR_HepG2_MitoMembPot_72h_dn','APR_HepG2_MitoticArrest_24h_up','APR_HepG2_MitoticArrest_72h_up','APR_HepG2_NuclearSize_24h_dn','APR_HepG2_NuclearSize_72h_dn','APR_HepG2_NuclearSize_72h_up','APR_HepG2_OxidativeStress_24h_up','APR_HepG2_OxidativeStress_72h_up','APR_HepG2_StressKinase_1h_up','APR_HepG2_StressKinase_24h_up','APR_HepG2_StressKinase_72h_up','APR_HepG2_p53Act_24h_up','APR_HepG2_p53Act_72h_up','APR_Hepat_Apoptosis_24hr_up','APR_Hepat_Apoptosis_48hr_up','APR_Hepat_CellLoss_24hr_dn','APR_Hepat_CellLoss_48hr_dn','APR_Hepat_DNADamage_24hr_up','APR_Hepat_DNADamage_48hr_up','APR_Hepat_DNATexture_24hr_up','APR_Hepat_DNATexture_48hr_up','APR_Hepat_MitoFxnI_1hr_dn','APR_Hepat_MitoFxnI_24hr_dn','APR_Hepat_MitoFxnI_48hr_dn','APR_Hepat_NuclearSize_24hr_dn','APR_Hepat_NuclearSize_48hr_dn','APR_Hepat_Steatosis_24hr_up','APR_Hepat_Steatosis_48hr_up','ATG_AP_1_CIS_dn','ATG_AP_1_CIS_up','ATG_AP_2_CIS_dn','ATG_AP_2_CIS_up','ATG_AR_TRANS_dn','ATG_AR_TRANS_up','ATG_Ahr_CIS_dn','ATG_Ahr_CIS_up','ATG_BRE_CIS_dn','ATG_BRE_CIS_up','ATG_CAR_TRANS_dn','ATG_CAR_TRANS_up','ATG_CMV_CIS_dn','ATG_CMV_CIS_up','ATG_CRE_CIS_dn','ATG_CRE_CIS_up','ATG_C_EBP_CIS_dn','ATG_C_EBP_CIS_up','ATG_DR4_LXR_CIS_dn','ATG_DR4_LXR_CIS_up','ATG_DR5_CIS_dn','ATG_DR5_CIS_up','ATG_E2F_CIS_dn','ATG_E2F_CIS_up','ATG_EGR_CIS_up','ATG_ERE_CIS_dn','ATG_ERE_CIS_up','ATG_ERRa_TRANS_dn','ATG_ERRg_TRANS_dn','ATG_ERRg_TRANS_up','ATG_ERa_TRANS_up','ATG_E_Box_CIS_dn','ATG_E_Box_CIS_up','ATG_Ets_CIS_dn','ATG_Ets_CIS_up','ATG_FXR_TRANS_up','ATG_FoxA2_CIS_dn','ATG_FoxA2_CIS_up','ATG_FoxO_CIS_dn','ATG_FoxO_CIS_up','ATG_GAL4_TRANS_dn','ATG_GATA_CIS_dn','ATG_GATA_CIS_up','ATG_GLI_CIS_dn','ATG_GLI_CIS_up','ATG_GRE_CIS_dn','ATG_GRE_CIS_up','ATG_GR_TRANS_dn','ATG_GR_TRANS_up','ATG_HIF1a_CIS_dn','ATG_HIF1a_CIS_up','ATG_HNF4a_TRANS_dn','ATG_HNF4a_TRANS_up','ATG_HNF6_CIS_dn','ATG_HNF6_CIS_up','ATG_HSE_CIS_dn','ATG_HSE_CIS_up','ATG_IR1_CIS_dn','ATG_IR1_CIS_up','ATG_ISRE_CIS_dn','ATG_ISRE_CIS_up','ATG_LXRa_TRANS_dn','ATG_LXRa_TRANS_up','ATG_LXRb_TRANS_dn','ATG_LXRb_TRANS_up','ATG_MRE_CIS_up','ATG_M_06_TRANS_up','ATG_M_19_CIS_dn','ATG_M_19_TRANS_dn','ATG_M_19_TRANS_up','ATG_M_32_CIS_dn','ATG_M_32_CIS_up','ATG_M_32_TRANS_dn','ATG_M_32_TRANS_up','ATG_M_61_TRANS_up','ATG_Myb_CIS_dn','ATG_Myb_CIS_up','ATG_Myc_CIS_dn','ATG_Myc_CIS_up','ATG_NFI_CIS_dn','ATG_NFI_CIS_up','ATG_NF_kB_CIS_dn','ATG_NF_kB_CIS_up','ATG_NRF1_CIS_dn','ATG_NRF1_CIS_up','ATG_NRF2_ARE_CIS_dn','ATG_NRF2_ARE_CIS_up','ATG_NURR1_TRANS_dn','ATG_NURR1_TRANS_up','ATG_Oct_MLP_CIS_dn','ATG_Oct_MLP_CIS_up','ATG_PBREM_CIS_dn','ATG_PBREM_CIS_up','ATG_PPARa_TRANS_dn','ATG_PPARa_TRANS_up','ATG_PPARd_TRANS_up','ATG_PPARg_TRANS_up','ATG_PPRE_CIS_dn','ATG_PPRE_CIS_up','ATG_PXRE_CIS_dn','ATG_PXRE_CIS_up','ATG_PXR_TRANS_dn','ATG_PXR_TRANS_up','ATG_Pax6_CIS_up','ATG_RARa_TRANS_dn','ATG_RARa_TRANS_up','ATG_RARb_TRANS_dn','ATG_RARb_TRANS_up','ATG_RARg_TRANS_dn','ATG_RARg_TRANS_up','ATG_RORE_CIS_dn','ATG_RORE_CIS_up','ATG_RORb_TRANS_dn','ATG_RORg_TRANS_dn','ATG_RORg_TRANS_up','ATG_RXRa_TRANS_dn','ATG_RXRa_TRANS_up','ATG_RXRb_TRANS_dn','ATG_RXRb_TRANS_up','ATG_SREBP_CIS_dn','ATG_SREBP_CIS_up','ATG_STAT3_CIS_dn','ATG_STAT3_CIS_up','ATG_Sox_CIS_dn','ATG_Sox_CIS_up','ATG_Sp1_CIS_dn','ATG_Sp1_CIS_up','ATG_TAL_CIS_dn','ATG_TAL_CIS_up','ATG_TA_CIS_dn','ATG_TA_CIS_up','ATG_TCF_b_cat_CIS_dn','ATG_TCF_b_cat_CIS_up','ATG_TGFb_CIS_dn','ATG_TGFb_CIS_up','ATG_THRa1_TRANS_dn','ATG_THRa1_TRANS_up','ATG_VDRE_CIS_dn','ATG_VDRE_CIS_up','ATG_VDR_TRANS_dn','ATG_VDR_TRANS_up','ATG_XTT_Cytotoxicity_up','ATG_Xbp1_CIS_dn','ATG_Xbp1_CIS_up','ATG_p53_CIS_dn','ATG_p53_CIS_up','BSK_3C_Eselectin_down','BSK_3C_HLADR_down','BSK_3C_ICAM1_down','BSK_3C_IL8_down','BSK_3C_MCP1_down','BSK_3C_MIG_down','BSK_3C_Proliferation_down','BSK_3C_SRB_down','BSK_3C_Thrombomodulin_down','BSK_3C_Thrombomodulin_up','BSK_3C_TissueFactor_down','BSK_3C_TissueFactor_up','BSK_3C_VCAM1_down','BSK_3C_Vis_down','BSK_3C_uPAR_down','BSK_4H_Eotaxin3_down','BSK_4H_MCP1_down','BSK_4H_Pselectin_down','BSK_4H_Pselectin_up','BSK_4H_SRB_down','BSK_4H_VCAM1_down','BSK_4H_VEGFRII_down','BSK_4H_uPAR_down','BSK_4H_uPAR_up','BSK_BE3C_HLADR_down','BSK_BE3C_IL1a_down','BSK_BE3C_IP10_down','BSK_BE3C_MIG_down','BSK_BE3C_MMP1_down','BSK_BE3C_MMP1_up','BSK_BE3C_PAI1_down','BSK_BE3C_SRB_down','BSK_BE3C_TGFb1_down','BSK_BE3C_tPA_down','BSK_BE3C_uPAR_down','BSK_BE3C_uPAR_up','BSK_BE3C_uPA_down','BSK_CASM3C_HLADR_down','BSK_CASM3C_IL6_down','BSK_CASM3C_IL6_up','BSK_CASM3C_IL8_down','BSK_CASM3C_LDLR_down','BSK_CASM3C_LDLR_up','BSK_CASM3C_MCP1_down','BSK_CASM3C_MCP1_up','BSK_CASM3C_MCSF_down','BSK_CASM3C_MCSF_up','BSK_CASM3C_MIG_down','BSK_CASM3C_Proliferation_down','BSK_CASM3C_Proliferation_up','BSK_CASM3C_SAA_down','BSK_CASM3C_SAA_up','BSK_CASM3C_SRB_down','BSK_CASM3C_Thrombomodulin_down','BSK_CASM3C_Thrombomodulin_up','BSK_CASM3C_TissueFactor_down','BSK_CASM3C_VCAM1_down','BSK_CASM3C_VCAM1_up','BSK_CASM3C_uPAR_down','BSK_CASM3C_uPAR_up','BSK_KF3CT_ICAM1_down','BSK_KF3CT_IL1a_down','BSK_KF3CT_IP10_down','BSK_KF3CT_IP10_up','BSK_KF3CT_MCP1_down','BSK_KF3CT_MCP1_up','BSK_KF3CT_MMP9_down','BSK_KF3CT_SRB_down','BSK_KF3CT_TGFb1_down','BSK_KF3CT_TIMP2_down','BSK_KF3CT_uPA_down','BSK_LPS_CD40_down','BSK_LPS_Eselectin_down','BSK_LPS_Eselectin_up','BSK_LPS_IL1a_down','BSK_LPS_IL1a_up','BSK_LPS_IL8_down','BSK_LPS_IL8_up','BSK_LPS_MCP1_down','BSK_LPS_MCSF_down','BSK_LPS_PGE2_down','BSK_LPS_PGE2_up','BSK_LPS_SRB_down','BSK_LPS_TNFa_down','BSK_LPS_TNFa_up','BSK_LPS_TissueFactor_down','BSK_LPS_TissueFactor_up','BSK_LPS_VCAM1_down','BSK_SAg_CD38_down','BSK_SAg_CD40_down','BSK_SAg_CD69_down','BSK_SAg_Eselectin_down','BSK_SAg_Eselectin_up','BSK_SAg_IL8_down','BSK_SAg_IL8_up','BSK_SAg_MCP1_down','BSK_SAg_MIG_down','BSK_SAg_PBMCCytotoxicity_down','BSK_SAg_PBMCCytotoxicity_up','BSK_SAg_Proliferation_down','BSK_SAg_SRB_down','BSK_hDFCGF_CollagenIII_down','BSK_hDFCGF_EGFR_down','BSK_hDFCGF_EGFR_up','BSK_hDFCGF_IL8_down','BSK_hDFCGF_IP10_down','BSK_hDFCGF_MCSF_down','BSK_hDFCGF_MIG_down','BSK_hDFCGF_MMP1_down','BSK_hDFCGF_MMP1_up','BSK_hDFCGF_PAI1_down','BSK_hDFCGF_Proliferation_down','BSK_hDFCGF_SRB_down','BSK_hDFCGF_TIMP1_down','BSK_hDFCGF_VCAM1_down','CEETOX_H295R_11DCORT_dn','CEETOX_H295R_ANDR_dn','CEETOX_H295R_CORTISOL_dn','CEETOX_H295R_DOC_dn','CEETOX_H295R_DOC_up','CEETOX_H295R_ESTRADIOL_dn','CEETOX_H295R_ESTRADIOL_up','CEETOX_H295R_ESTRONE_dn','CEETOX_H295R_ESTRONE_up','CEETOX_H295R_OHPREG_up','CEETOX_H295R_OHPROG_dn','CEETOX_H295R_OHPROG_up','CEETOX_H295R_PROG_up','CEETOX_H295R_TESTO_dn','CLD_ABCB1_48hr','CLD_ABCG2_48hr','CLD_CYP1A1_24hr','CLD_CYP1A1_48hr','CLD_CYP1A1_6hr','CLD_CYP1A2_24hr','CLD_CYP1A2_48hr','CLD_CYP1A2_6hr','CLD_CYP2B6_24hr','CLD_CYP2B6_48hr','CLD_CYP2B6_6hr','CLD_CYP3A4_24hr','CLD_CYP3A4_48hr','CLD_CYP3A4_6hr','CLD_GSTA2_48hr','CLD_SULT2A_24hr','CLD_SULT2A_48hr','CLD_UGT1A1_24hr','CLD_UGT1A1_48hr','NCCT_HEK293T_CellTiterGLO','NCCT_QuantiLum_inhib_2_dn','NCCT_QuantiLum_inhib_dn','NCCT_TPO_AUR_dn','NCCT_TPO_GUA_dn','NHEERL_ZF_144hpf_TERATOSCORE_up','NVS_ADME_hCYP19A1','NVS_ADME_hCYP1A1','NVS_ADME_hCYP1A2','NVS_ADME_hCYP2A6','NVS_ADME_hCYP2B6','NVS_ADME_hCYP2C19','NVS_ADME_hCYP2C9','NVS_ADME_hCYP2D6','NVS_ADME_hCYP3A4','NVS_ADME_hCYP4F12','NVS_ADME_rCYP2C12','NVS_ENZ_hAChE','NVS_ENZ_hAMPKa1','NVS_ENZ_hAurA','NVS_ENZ_hBACE','NVS_ENZ_hCASP5','NVS_ENZ_hCK1D','NVS_ENZ_hDUSP3','NVS_ENZ_hES','NVS_ENZ_hElastase','NVS_ENZ_hFGFR1','NVS_ENZ_hGSK3b','NVS_ENZ_hMMP1','NVS_ENZ_hMMP13','NVS_ENZ_hMMP2','NVS_ENZ_hMMP3','NVS_ENZ_hMMP7','NVS_ENZ_hMMP9','NVS_ENZ_hPDE10','NVS_ENZ_hPDE4A1','NVS_ENZ_hPDE5','NVS_ENZ_hPI3Ka','NVS_ENZ_hPTEN','NVS_ENZ_hPTPN11','NVS_ENZ_hPTPN12','NVS_ENZ_hPTPN13','NVS_ENZ_hPTPN9','NVS_ENZ_hPTPRC','NVS_ENZ_hSIRT1','NVS_ENZ_hSIRT2','NVS_ENZ_hTrkA','NVS_ENZ_hVEGFR2','NVS_ENZ_oCOX1','NVS_ENZ_oCOX2','NVS_ENZ_rAChE','NVS_ENZ_rCNOS','NVS_ENZ_rMAOAC','NVS_ENZ_rMAOAP','NVS_ENZ_rMAOBC','NVS_ENZ_rMAOBP','NVS_ENZ_rabI2C','NVS_GPCR_bAdoR_NonSelective','NVS_GPCR_bDR_NonSelective','NVS_GPCR_g5HT4','NVS_GPCR_gH2','NVS_GPCR_gLTB4','NVS_GPCR_gLTD4','NVS_GPCR_gMPeripheral_NonSelective','NVS_GPCR_gOpiateK','NVS_GPCR_h5HT2A','NVS_GPCR_h5HT5A','NVS_GPCR_h5HT6','NVS_GPCR_h5HT7','NVS_GPCR_hAT1','NVS_GPCR_hAdoRA1','NVS_GPCR_hAdoRA2a','NVS_GPCR_hAdra2A','NVS_GPCR_hAdra2C','NVS_GPCR_hAdrb1','NVS_GPCR_hAdrb2','NVS_GPCR_hAdrb3','NVS_GPCR_hDRD1','NVS_GPCR_hDRD2s','NVS_GPCR_hDRD4.4','NVS_GPCR_hH1','NVS_GPCR_hLTB4_BLT1','NVS_GPCR_hM1','NVS_GPCR_hM2','NVS_GPCR_hM3','NVS_GPCR_hM4','NVS_GPCR_hNK2','NVS_GPCR_hOpiate_D1','NVS_GPCR_hOpiate_mu','NVS_GPCR_hTXA2','NVS_GPCR_p5HT2C','NVS_GPCR_r5HT1_NonSelective','NVS_GPCR_r5HT_NonSelective','NVS_GPCR_rAdra1B','NVS_GPCR_rAdra1_NonSelective','NVS_GPCR_rAdra2_NonSelective','NVS_GPCR_rAdrb_NonSelective','NVS_GPCR_rNK1','NVS_GPCR_rNK3','NVS_GPCR_rOpiate_NonSelective','NVS_GPCR_rOpiate_NonSelectiveNa','NVS_GPCR_rSST','NVS_GPCR_rTRH','NVS_GPCR_rV1','NVS_GPCR_rabPAF','NVS_GPCR_rmAdra2B','NVS_IC_hKhERGCh','NVS_IC_rCaBTZCHL','NVS_IC_rCaDHPRCh_L','NVS_IC_rNaCh_site2','NVS_LGIC_bGABARa1','NVS_LGIC_h5HT3','NVS_LGIC_hNNR_NBungSens','NVS_LGIC_rGABAR_NonSelective','NVS_LGIC_rNNR_BungSens','NVS_MP_hPBR','NVS_MP_rPBR','NVS_NR_bER','NVS_NR_bPR','NVS_NR_cAR','NVS_NR_hAR','NVS_NR_hCAR_Antagonist','NVS_NR_hER','NVS_NR_hFXR_Agonist','NVS_NR_hFXR_Antagonist','NVS_NR_hGR','NVS_NR_hPPARa','NVS_NR_hPPARg','NVS_NR_hPR','NVS_NR_hPXR','NVS_NR_hRAR_Antagonist','NVS_NR_hRARa_Agonist','NVS_NR_hTRa_Antagonist','NVS_NR_mERa','NVS_NR_rAR','NVS_NR_rMR','NVS_OR_gSIGMA_NonSelective','NVS_TR_gDAT','NVS_TR_hAdoT','NVS_TR_hDAT','NVS_TR_hNET','NVS_TR_hSERT','NVS_TR_rNET','NVS_TR_rSERT','NVS_TR_rVMAT2','OT_AR_ARELUC_AG_1440','OT_AR_ARSRC1_0480','OT_AR_ARSRC1_0960','OT_ER_ERaERa_0480','OT_ER_ERaERa_1440','OT_ER_ERaERb_0480','OT_ER_ERaERb_1440','OT_ER_ERbERb_0480','OT_ER_ERbERb_1440','OT_ERa_EREGFP_0120','OT_ERa_EREGFP_0480','OT_FXR_FXRSRC1_0480','OT_FXR_FXRSRC1_1440','OT_NURR1_NURR1RXRa_0480','OT_NURR1_NURR1RXRa_1440','TOX21_ARE_BLA_Agonist_ch1','TOX21_ARE_BLA_Agonist_ch2','TOX21_ARE_BLA_agonist_ratio','TOX21_ARE_BLA_agonist_viability','TOX21_AR_BLA_Agonist_ch1','TOX21_AR_BLA_Agonist_ch2','TOX21_AR_BLA_Agonist_ratio','TOX21_AR_BLA_Antagonist_ch1','TOX21_AR_BLA_Antagonist_ch2','TOX21_AR_BLA_Antagonist_ratio','TOX21_AR_BLA_Antagonist_viability','TOX21_AR_LUC_MDAKB2_Agonist','TOX21_AR_LUC_MDAKB2_Antagonist','TOX21_AR_LUC_MDAKB2_Antagonist2','TOX21_AhR_LUC_Agonist','TOX21_Aromatase_Inhibition','TOX21_AutoFluor_HEK293_Cell_blue','TOX21_AutoFluor_HEK293_Media_blue','TOX21_AutoFluor_HEPG2_Cell_blue','TOX21_AutoFluor_HEPG2_Cell_green','TOX21_AutoFluor_HEPG2_Media_blue','TOX21_AutoFluor_HEPG2_Media_green','TOX21_ELG1_LUC_Agonist','TOX21_ERa_BLA_Agonist_ch1','TOX21_ERa_BLA_Agonist_ch2','TOX21_ERa_BLA_Agonist_ratio','TOX21_ERa_BLA_Antagonist_ch1','TOX21_ERa_BLA_Antagonist_ch2','TOX21_ERa_BLA_Antagonist_ratio','TOX21_ERa_BLA_Antagonist_viability','TOX21_ERa_LUC_BG1_Agonist','TOX21_ERa_LUC_BG1_Antagonist','TOX21_ESRE_BLA_ch1','TOX21_ESRE_BLA_ch2','TOX21_ESRE_BLA_ratio','TOX21_ESRE_BLA_viability','TOX21_FXR_BLA_Antagonist_ch1','TOX21_FXR_BLA_Antagonist_ch2','TOX21_FXR_BLA_agonist_ch2','TOX21_FXR_BLA_agonist_ratio','TOX21_FXR_BLA_antagonist_ratio','TOX21_FXR_BLA_antagonist_viability','TOX21_GR_BLA_Agonist_ch1','TOX21_GR_BLA_Agonist_ch2','TOX21_GR_BLA_Agonist_ratio','TOX21_GR_BLA_Antagonist_ch2','TOX21_GR_BLA_Antagonist_ratio','TOX21_GR_BLA_Antagonist_viability','TOX21_HSE_BLA_agonist_ch1','TOX21_HSE_BLA_agonist_ch2','TOX21_HSE_BLA_agonist_ratio','TOX21_HSE_BLA_agonist_viability','TOX21_MMP_ratio_down','TOX21_MMP_ratio_up','TOX21_MMP_viability','TOX21_NFkB_BLA_agonist_ch1','TOX21_NFkB_BLA_agonist_ch2','TOX21_NFkB_BLA_agonist_ratio','TOX21_NFkB_BLA_agonist_viability','TOX21_PPARd_BLA_Agonist_viability','TOX21_PPARd_BLA_Antagonist_ch1','TOX21_PPARd_BLA_agonist_ch1','TOX21_PPARd_BLA_agonist_ch2','TOX21_PPARd_BLA_agonist_ratio','TOX21_PPARd_BLA_antagonist_ratio','TOX21_PPARd_BLA_antagonist_viability','TOX21_PPARg_BLA_Agonist_ch1','TOX21_PPARg_BLA_Agonist_ch2','TOX21_PPARg_BLA_Agonist_ratio','TOX21_PPARg_BLA_Antagonist_ch1','TOX21_PPARg_BLA_antagonist_ratio','TOX21_PPARg_BLA_antagonist_viability','TOX21_TR_LUC_GH3_Agonist','TOX21_TR_LUC_GH3_Antagonist','TOX21_VDR_BLA_Agonist_viability','TOX21_VDR_BLA_Antagonist_ch1','TOX21_VDR_BLA_agonist_ch2','TOX21_VDR_BLA_agonist_ratio','TOX21_VDR_BLA_antagonist_ratio','TOX21_VDR_BLA_antagonist_viability','TOX21_p53_BLA_p1_ch1','TOX21_p53_BLA_p1_ch2','TOX21_p53_BLA_p1_ratio','TOX21_p53_BLA_p1_viability','TOX21_p53_BLA_p2_ch1','TOX21_p53_BLA_p2_ch2','TOX21_p53_BLA_p2_ratio','TOX21_p53_BLA_p2_viability','TOX21_p53_BLA_p3_ch1','TOX21_p53_BLA_p3_ch2','TOX21_p53_BLA_p3_ratio','TOX21_p53_BLA_p3_viability','TOX21_p53_BLA_p4_ch1','TOX21_p53_BLA_p4_ch2','TOX21_p53_BLA_p4_ratio','TOX21_p53_BLA_p4_viability','TOX21_p53_BLA_p5_ch1','TOX21_p53_BLA_p5_ch2','TOX21_p53_BLA_p5_ratio','TOX21_p53_BLA_p5_viability','Tanguay_ZF_120hpf_AXIS_up','Tanguay_ZF_120hpf_ActivityScore','Tanguay_ZF_120hpf_BRAI_up','Tanguay_ZF_120hpf_CFIN_up','Tanguay_ZF_120hpf_CIRC_up','Tanguay_ZF_120hpf_EYE_up','Tanguay_ZF_120hpf_JAW_up','Tanguay_ZF_120hpf_MORT_up','Tanguay_ZF_120hpf_OTIC_up','Tanguay_ZF_120hpf_PE_up','Tanguay_ZF_120hpf_PFIN_up','Tanguay_ZF_120hpf_PIG_up','Tanguay_ZF_120hpf_SNOU_up','Tanguay_ZF_120hpf_SOMI_up','Tanguay_ZF_120hpf_SWIM_up','Tanguay_ZF_120hpf_TRUN_up','Tanguay_ZF_120hpf_TR_up','Tanguay_ZF_120hpf_YSE_up']
        elif dataset == 'ClinTox':
            task_name = ['FDA_APPROVED','CT_TOX']
        elif dataset == 'Tox21':
            task_name = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD','NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
        elif dataset == 'HIV':
            task_name = ['HIV_active']
        elif dataset == 'MUV':
            task_name = ["MUV-466","MUV-548","MUV-600","MUV-644","MUV-652","MUV-689","MUV-692","MUV-712","MUV-713","MUV-733","MUV-737","MUV-810","MUV-832","MUV-846","MUV-852","MUV-858","MUV-859"]
        elif dataset == 'PCBA':
            task_name = ['PCBA-1030', 'PCBA-1379', 'PCBA-1452', 'PCBA-1454', 'PCBA-1457', 'PCBA-1458', 'PCBA-1460', 'PCBA-1461', 'PCBA-1468', 'PCBA-1469', 'PCBA-1471', 'PCBA-1479', 'PCBA-1631', 'PCBA-1634', 'PCBA-1688', 'PCBA-1721', 'PCBA-2100', 'PCBA-2101', 'PCBA-2147', 'PCBA-2242', 'PCBA-2326', 'PCBA-2451', 'PCBA-2517', 'PCBA-2528', 'PCBA-2546', 'PCBA-2549', 'PCBA-2551', 'PCBA-2662', 'PCBA-2675', 'PCBA-2676', 'PCBA-411', 'PCBA-463254', 'PCBA-485281', 'PCBA-485290', 'PCBA-485294', 'PCBA-485297', 'PCBA-485313', 'PCBA-485314', 'PCBA-485341', 'PCBA-485349', 'PCBA-485353', 'PCBA-485360', 'PCBA-485364', 'PCBA-485367', 'PCBA-492947', 'PCBA-493208', 'PCBA-504327', 'PCBA-504332', 'PCBA-504333', 'PCBA-504339', 'PCBA-504444', 'PCBA-504466', 'PCBA-504467', 'PCBA-504706', 'PCBA-504842', 'PCBA-504845', 'PCBA-504847', 'PCBA-504891', 'PCBA-540276', 'PCBA-540317', 'PCBA-588342', 'PCBA-588453', 'PCBA-588456', 'PCBA-588579', 'PCBA-588590', 'PCBA-588591', 'PCBA-588795', 'PCBA-588855', 'PCBA-602179', 'PCBA-602233', 'PCBA-602310', 'PCBA-602313', 'PCBA-602332', 'PCBA-624170', 'PCBA-624171', 'PCBA-624173', 'PCBA-624202', 'PCBA-624246', 'PCBA-624287', 'PCBA-624288', 'PCBA-624291', 'PCBA-624296', 'PCBA-624297', 'PCBA-624417', 'PCBA-651635', 'PCBA-651644', 'PCBA-651768', 'PCBA-651965', 'PCBA-652025', 'PCBA-652104', 'PCBA-652105', 'PCBA-652106', 'PCBA-686970', 'PCBA-686978', 'PCBA-686979', 'PCBA-720504', 'PCBA-720532', 'PCBA-720542', 'PCBA-720551', 'PCBA-720553', 'PCBA-720579', 'PCBA-720580', 'PCBA-720707', 'PCBA-720708', 'PCBA-720709', 'PCBA-720711', 'PCBA-743255', 'PCBA-743266', 'PCBA-875', 'PCBA-881', 'PCBA-883', 'PCBA-884', 'PCBA-885', 'PCBA-887', 'PCBA-891', 'PCBA-899', 'PCBA-902', 'PCBA-903', 'PCBA-904', 'PCBA-912', 'PCBA-914', 'PCBA-915', 'PCBA-924', 'PCBA-925', 'PCBA-926', 'PCBA-927', 'PCBA-938', 'PCBA-995']
        elif dataset == 'FreeSolv':
            task_name = ['expt']
        elif dataset == 'Lipo':
            task_name = ['exp']
        elif dataset == 'ESOL':
            task_name = ['measured log solubility in mols per litre']
        elif dataset == 'QM7':
            task_name = ["u0_atom"]
        elif dataset == 'QM8':
            task_name = ["E1-CC2","E2-CC2","f1-CC2","f2-CC2","E1-PBE0","E2-PBE0","f1-PBE0","f2-PBE0","E1-PBE0.1","E1-CAM","E2-CAM","f1-CAM","f2-CAM"]
        elif dataset == 'QM9':
            task_name = ["mu","alpha","homo","lumo","gap","r2","zpve","u0","u298","h298","g298","cv"]
        elif dataset == 'corrosion':
            task_name = ['IE']
        
        
    print('Read and process the collected data...')
    #file = pd.read_csv('./data/' + dataset + 'Scaffold.csv', header=0)
    scaffold_file = './data/' + dataset + 'Scaffold.csv'
    default_file = './data/' + dataset + '.csv'
    if os.path.exists(scaffold_file):
        file = pd.read_csv(scaffold_file, header=0)
    else:
        # 如果不存在，则读取默认的 CSV 文件
        file = pd.read_csv(default_file, header=0)

    if 'smiles' in file:
        smi_name = 'smiles'
    else:
        smi_name = 'mol'
        
    if print_info:
        print('----------------------------------------')
        print('Dataset: ', dataset)
        print('Example: ')
        print(file.iloc[0])
        print('Number of molecules:', file.shape[0])
        
    all_smiles_y = []
    file = file.where(pd.notnull(file), -1)
    for i in range(file.shape[0]):
        target = []
        for j in task_name:
            target.append(file[j][i])
        all_smiles_y.append([file[smi_name][i],target])
    return all_smiles_y

class Molecule():
    def __init__(self, smiles_y, dataset,seed, bool_random=True, max_len=100,max_motif=100, max_ring=15, print_info=False):
        self.max_len = max_len # maximum number of atoms in a molecule
        self.max_motif = max_motif
        self.max_ring = max_ring
        self.smiles, self.targets = smiles_y
        self.nb_MP = len(self.targets)
        self.exist = True
        self.dataset = dataset # name of dataset
        self.seed = seed
        self.process_mol_with_RDKit()
       
    def model_needed_info(self):
        return self.nb_node_features, self.nb_edge_features, self.nb_MP
    
    def process_mol_with_RDKit(self, bool_random=True):
        random.seed(self.seed)
        np.random.seed(self.seed)
        mol = Chem.MolFromSmiles(self.smiles, sanitize=True)
        if mol is None:
            self.exist = False
            print('Bad smiles to generate mol!!!', self.smiles)
            return None#生成分子对象
        nodes_features = []
        edges = []
        edges_features = []
        
        nb_atoms = len(mol.GetAtoms())
        if self.dataset in ['PCBA', 'SIDER', 'HIV'] and nb_atoms <= self.max_len:
            special_max_len = [24, 50, 75, 100, 150, 200, 250, 300, 350, 500]
            for i in special_max_len:
                if nb_atoms <= i:
                    self.max_len = min(i, self.max_len)
                    break
                    
        node_len = self.max_len
        edge_len = self.max_len + self.max_ring
        all_ele = ['PAD', 'MASK', 'UNK', 'H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn',\
                   'Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I',\
                   'Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl',\
                   'Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm']
        ele2index = {j : i for i,j in enumerate(all_ele)}#原子类型到索引的映射 103（100元素，PAD,MASK，UNK）
        num_ele = len(all_ele)
                
        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()
        adjacency_matrix = np.zeros((node_len, node_len), dtype=np.float32)
        mapping = np.arange(len(mol.GetAtoms()))
        if bool_random:
            np.random.shuffle(mapping)#随机打乱mapping，可以使原子顺序在训练过程中变得不可预测
        reverse_mapping = np.zeros(len(mapping)).astype(int)
        for i in range(len(mapping)):
            reverse_mapping[mapping[i]] = i#打乱后的索引找到原始的原子位置
        mapping = mapping.tolist()
        reverse_mapping = reverse_mapping.tolist()
        if len(atoms) <= node_len:
            for i in range(len(atoms)): 
                atom = atoms[reverse_mapping[i]]#顺序访问原子
                node_features = [0] * (num_ele+28)#103+28=131
                node_features[ele2index[atom.GetSymbol() if atom.GetSymbol() in all_ele else 'UNK']] = 1 # 103 atomic numebr [all_ele]node_features 中对应于元素符号的索引位置的值设置为 1
                node_features[num_ele+(atom.GetDegree() if atom.GetDegree()<=5 else 6)] = 1  # 7 degree of atom [0~5, 6=other]
                node_features[int(atom.GetHybridization())+num_ele+7] = 1   # 8 hybridization type [unspecified,s,sp,sp2,sp3,sp3d,sp3d2,other]
                node_features[int(atom.GetChiralTag())+num_ele+15] = 1      #4 chirality [CHI_UNSPECIFIED,CHI_TETRAHEDRAL_CW,CHI_TETRAHEDRAL_CCW,CHI_OTHER]
                num_H = atom.GetTotalNumHs() if atom.GetTotalNumHs() < 4 else 4
                node_features[num_H+num_ele+19] = 1                         #5 number of H atoms [0,1,2,3,4]
                node_features[num_ele+24] = int(atom.IsInRing())             #1 whether in ring
                node_features[num_ele+25] = int(atom.GetIsAromatic())        #1 whether aromatic
                node_features[num_ele+26] = atom.GetFormalCharge()          #1 formal charge
                node_features[num_ele+27] = atom.GetNumRadicalElectrons()   #1 radical electrons
                nodes_features.append(node_features)
            node_features = [0] * (num_ele+28)
            node_features[ele2index['PAD']] = 1
            for _ in range(node_len - len(atoms)):
                nodes_features.append(node_features)#实际原子数量 len(atoms) 少于 node_len，则使用填充特征来补充

            for bond in bonds:
                edge = [mapping[bond.GetBeginAtomIdx()], mapping[bond.GetEndAtomIdx()]]
                edges.append(edge)
                adjacency_matrix[edge[0], edge[1]] = 1
                adjacency_matrix[edge[1], edge[0]] = 1
                edge_features = [0] * 17                        # First two places are indices of connected nodes, third place is used as [MASKED EDGE]
                edge_features[0] = edge[0]
                edge_features[1] = edge[1]# 存储键的两个端点原子的索引
                bond_type = int(bond.GetBondType()) if (int(bond.GetBondType())<=3 or int(bond.GetBondType())==12) else 0
                if bond_type == 12:
                    bond_type = 4#键的类型是 12，则被转换为 4（芳香键）
                edge_features[bond_type+2] = 1                  # bond type [OTHERS,SINGLE,DOUBLE,TRIPLE,AROMATIC]
                edge_features[7] = int(bond.GetIsConjugated())  # whether conjugation
                edge_features[8] = int(bond.IsInRing())         # whether in the ring
                edge_features[int(bond.GetStereo())+9] = 1      #6 stereo type [STEREONONE,STEREOANY,STEREOZ,STEREOE,STEREOCIS,STEREOTRANS]

                edges_features.append(edge_features)
            edge_features = [0] * 17
            edge_features[-1] = 1
            if edge_len < len(bonds):
                print('Too Much Bonds!!! ', self.smiles)
                self.exist = False
                return None
            for _ in range(edge_len - len(bonds)):
                edges_features.append(edge_features)

            np.fill_diagonal(adjacency_matrix, 1)

            max_motif = self.max_motif
            fragments_atom_indices, edges = motif_decomp(mol, mapping)#拆分分子，模体包含的原子，模体之间的连接
            frag_connection_matrix = create_adjacency_matrix(fragments_atom_indices, edges, max_motif)#碎片之间连接状态
            # 碎片连接原子情况
            atom_motif_matrix = np.zeros((max_motif, node_len))
            nb_atom = len(mol.GetAtoms())
            for frag_idx, atom_indices in enumerate(fragments_atom_indices):
                for atom_idx in atom_indices:
                    if frag_idx < max_motif and atom_idx < nb_atom:
                        atom_motif_matrix[frag_idx, atom_idx] = 1

            # descriptor_names = [desc_name[0] for desc_name in Descriptors._descList]
            # calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
            # try:
            #     descriptors = calculator.CalcDescriptors(mol)
            # except:
            #     # 如果无法计算分子描述符，返回一个空矩阵
            #     descriptors = [0] * len(descriptor_names)
            # descriptors = [0 if isinstance(d, float) and math.isnan(d) else d for d in descriptors]
            # descriptors = np.array(descriptors).reshape(1, -1)
            # scaler = MinMaxScaler()
            # descriptors = scaler.fit_transform(descriptors)  # shape: (32, 210)
            # descriptors = descriptors.flatten()
            descriptor_names = [desc_name[0] for desc_name in Descriptors._descList]
            calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
            # 计算分子描述符
            try:
                descriptors = calculator.CalcDescriptors(mol)
            except:
                # 如果无法计算分子描述符，返回一个空矩阵
                descriptors = [0] * len(descriptor_names)
            # 处理 NaN 和 Inf 值
            descriptors = [
                0 if isinstance(d, float) and (math.isnan(d) or math.isinf(d)) else d
                for d in descriptors
            ]
            # 确保描述符是数值类型并转化为 NumPy 数组
            descriptors = np.array(descriptors, dtype=np.float64)
            # 扩展为 (1, -1) 形状以适应 MinMaxScaler
            descriptors = descriptors.reshape(1, -1)
            # 标准化描述符
            scaler = MinMaxScaler()
            descriptors = scaler.fit_transform(descriptors)  # 现在 descriptors 形状为 (1, 210)
            # 将描述符拉平
            descriptors = descriptors.flatten()

            fp = []
            fp_list = []
            #if self.fp_type == 'mixed':
            fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
            fp_phaErGfp = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)
            fp_pubcfp = GetPubChemFPs(mol)
            fp.extend(fp_maccs)
            fp.extend(fp_phaErGfp)
            fp.extend(fp_pubcfp)
            # else:
            #     fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            #     fp.extend(fp_morgan)
            #fp_list.append(fp)

        else:
            self.exist = False
#             print('Bad molecule to generate features!!!', self.smiles)
            return None
#             raise RuntimeError('Bad molecule to generate features!!!')

        self.mol = mol
        self.nb_atoms = len(atoms)
        self.nodes_features = np.array(nodes_features)
        self.edges_features = np.array(edges_features)
        self.frag_connection_matrix = np.array(frag_connection_matrix)
        self.atom_motif_matrix = np.array(atom_motif_matrix)
        self.adjacency_matrix = np.array(adjacency_matrix)
        self.descriptors = np.array(descriptors)
        self.fp = np.array(fp)
        self.nb_node_features = len(node_features)
        self.nb_edge_features = len(edge_features)
        self.edges = edges
        
    def get_inputs_features(self):
        return self.nodes_features, self.edges_features, self.frag_connection_matrix, self.atom_motif_matrix, self.adjacency_matrix, self.descriptors,self.fp
    
    def get_edges(self):
        return self.edges
    
def para_process_mol(i, all_smiles_y, dataset, max_len,max_motif,seed, nb_time_processing=10):
    mol = []
    for j in range(i*nb_time_processing, min((i+1)*nb_time_processing,len(all_smiles_y))):
        one_mol = Molecule(all_smiles_y[j], dataset, seed,False, max_len,max_motif)
        if one_mol.exist:
            mol.append(one_mol)
    return i, mol

def Read_mol_data(dataset, log, task_name=None, target_type='classification', seed=0, split="scaffold"):
    assert target_type in ['classification', 'regression']
    all_smiles_y = load_data(dataset, task_name)
    smiles_list = [entry[0] for entry in all_smiles_y]
    features_path = './data/' + dataset + 'Scaffold.npy'
    smiles_features = np.load(features_path)
    smiles_to_features = {smiles: smiles_features[i] for i, smiles in enumerate(smiles_list)}
    set_seed(seed)
   #加载数据与设置分子长度
    all_smiles_y = load_data(dataset, task_name)
    #加载数据并根据数据集设置最大分子长度。
    # if dataset in max_len:
    #     train_max_len, val_max_len, test_max_len = max_len[dataset]
    # else:
    max_len = get_mol_max_length(all_smiles_y)
    train_max_len = val_max_len = test_max_len = max_len
    fra_train_max_len = fra_val_max_len = fra_test_max_len = get_motif_max_length(all_smiles_y)
    #计算回归任务的均值和标准差
    if target_type == 'classification':
        mean = std = None
    else:
        all_targets = []
        for i in all_smiles_y:
            all_targets.append(i[1])
        mean = torch.Tensor(np.mean(all_targets, axis=0))
        std = torch.Tensor(np.std(all_targets, axis=0))
            
    pos_times_HIV = 20#正样本复制次数
    pos_times_ClinTox = 12 
    pos_times_MUV = 2

    #数据集划分
    # num_train = int(len(all_smiles_y) * 0.8)
    # num_val= int(len(all_smiles_y) * 0.1)
    # train_smiles_y = all_smiles_y[:num_train]
    # val_smiles_y = all_smiles_y[num_train:num_train+num_val]
    # test_smiles_y = all_smiles_y[num_train+num_val:]
    split_ratio = [0.8, 0.1, 0.1]
    if split == "scaffold":
        train_smiles_y, val_smiles_y, test_smiles_y = scaffold_split(all_smiles_y, size=split_ratio, seed=seed, log=log)
        print("scaffold")
    elif split == "random":
        train_smiles_y, val_smiles_y, test_smiles_y = random_split(all_smiles_y, size=split_ratio, seed=seed, log=log)
        print("random")
    else:
        raise ValueError("Invalid split option.")

    #正样本增强（标签为1复制pos_times），平衡样本比例
    new_train_smiles_y = []
    mol_train = []
    mol_val = []
    mol_test = []
    if dataset in ['HIV']:
        for i in range(len(train_smiles_y)):
            if train_smiles_y[i][1][0] == 0:
                new_train_smiles_y.append(train_smiles_y[i])
            else:
                for _ in range(pos_times_HIV):
                    new_train_smiles_y.append(train_smiles_y[i])
    # elif dataset in ['ClinTox']:
    #     for i in range(len(train_smiles_y)):
    #         if train_smiles_y[i][1][0] == 1:
    #             new_train_smiles_y.append(train_smiles_y[i])
    #         else:
    #             for _ in range(pos_times_ClinTox):
    #                 new_train_smiles_y.append(train_smiles_y[i])
    elif dataset in ['MUV']:
        for i in range(len(train_smiles_y)):
            if 1 in train_smiles_y[i][1]:
                for _ in range(pos_times_MUV):
                    new_train_smiles_y.append(train_smiles_y[i])
            else:
                new_train_smiles_y.append(train_smiles_y[i])
    else:
        for i in range(len(train_smiles_y)):
            new_train_smiles_y.append(train_smiles_y[i])

    random.shuffle(new_train_smiles_y)
    nb_cpu = 4
    nb_time_processing = 20
    nb = 1
    #并行处理训练集
    nb_part = len(new_train_smiles_y)
    for i in range(int(np.ceil(nb_part / nb_time_processing))):
        _, mol_part = para_process_mol(i, all_smiles_y=new_train_smiles_y, dataset=dataset, max_len=train_max_len,
                                       max_motif=fra_train_max_len, seed=seed, nb_time_processing=nb_time_processing)
        mol_train += mol_part
        if nb == 1 and (i + 1) % (nb_cpu * 50) == 0:
            print(i + 1, '/', int(np.ceil(nb_part / nb_time_processing)), ' finished')
    print('Training dataset finished')
    random.shuffle(mol_train)  

    #并行处理验证集
    nb_part = len(val_smiles_y)
    for i in range(int(np.ceil(nb_part / nb_time_processing))):
        _, mol_part = para_process_mol(i, all_smiles_y=val_smiles_y, dataset=dataset, max_len=val_max_len,
                                       max_motif=fra_val_max_len,seed=seed,nb_time_processing=nb_time_processing)
        mol_val += mol_part
        if nb == 1 and (i + 1) % (nb_cpu * 25) == 0:
            print(i + 1, '/', int(np.ceil(nb_part / nb_time_processing)), ' finished')
    print('Val dataset finished')
    #并行处理验证集
    nb_part = len(test_smiles_y)
    for i in range(int(np.ceil(nb_part / nb_time_processing))):
        _, mol_part = para_process_mol(i, all_smiles_y=test_smiles_y, dataset=dataset, max_len=test_max_len,
                                       max_motif=fra_test_max_len,seed=seed,nb_time_processing=nb_time_processing)
        mol_test += mol_part
        if nb == 1 and (i + 1) % (25) == 0:
            print(i + 1, '/', int(np.ceil(nb_part/ nb_time_processing)), ' finished')
    print('Test dataset finished')
    return mol_train, mol_val, mol_test, mean, std, smiles_to_features

def PreProcess(mol,smiles_features):
    targets = []
    inputs = {'nodes_features':[], 'edges_features':[],'frag_connection_matrix':[], 'atom_motif_matrix':[],'adjacency_matrix':[],'descriptors':[],'fp':[],'smiles_features':[]}
    for i in range(len(mol)):
        # for targets
        targets.append(mol[i].targets)
        
        # for inputs
        nodes_features, edges_features,frag_connection_matrix,atom_motif_matrix,adjacency_matrix, descriptors,fp = mol[i].get_inputs_features()
        # node_len = nodes_features.shape[0]
        # edge_len = edges_features.shape[0]
        inputs['nodes_features'] += [nodes_features.tolist()]
        inputs['edges_features'] += [edges_features.tolist()]
        inputs['frag_connection_matrix'] += [frag_connection_matrix.tolist()]
        inputs['atom_motif_matrix'] += [atom_motif_matrix.tolist()]
        inputs['adjacency_matrix'] += [adjacency_matrix.tolist()]
        inputs['descriptors'] += [descriptors.tolist()]
        inputs['fp'] += [fp.tolist()]

        smiles = mol[i].smiles  # 假设 mol[i] 对象有 smiles 属性
        smiles_feature = smiles_features.get(smiles, None)  # 查找对应的特征
        if smiles_feature is not None:
            inputs['smiles_features'] += [smiles_feature.tolist()]  # 将特征添加到 inputs 字典
        else:
            # 如果找不到对应的 corrosion 特征，添加一个默认值
            inputs['smiles_features'] += [np.zeros(512).tolist()]

        
    targets = torch.Tensor(targets)
    for name in inputs:
        inputs[name] = torch.Tensor(inputs[name])
    
    return inputs, targets

def Generate_dataloader(dataset, mol_train, mol_val, mol_test, smiles_features,bsz):
    train_dataloader = []
    val_dataloader = []
    test_dataloader = []
    if dataset in ['SIDER', 'PCBA', 'HIV']:
        special_max_len = [24, 50, 75, 100, 150, 200, 250, 300, 350, 500]

        for i in special_max_len:
            new_mol_train = []
            for j in mol_train:
                if j.max_len == i:
                    new_mol_train.append(j)
            new_mol_val = []
            for j in mol_val:
                if j.max_len == i:
                    new_mol_val.append(j)
            new_mol_test = []
            for j in mol_test:
                if j.max_len == i:
                    new_mol_test.append(j)

            # if i <= 350:
            #     bsz = 32
            # else:
            #     bsz = 8
            num_train_dataloader = int(np.ceil(len(new_mol_train)/bsz))
            num_val_dataloader = int(np.ceil(len(new_mol_val)/bsz))
            num_test_dataloader = int(np.ceil(len(new_mol_test)/bsz))

            for j in range(num_train_dataloader):
                inputs, targets = PreProcess(new_mol_train[j * bsz:(j+1)*bsz], smiles_features)
                train_dataloader.append([inputs, targets])

            for j in range(num_val_dataloader):
                inputs, targets = PreProcess(new_mol_val[j * bsz: (j+1) * bsz], smiles_features)
                val_dataloader.append([inputs, targets])

            for j in range(num_test_dataloader):
                inputs, targets = PreProcess(new_mol_test[j * bsz: (j+1) * bsz], smiles_features)
                test_dataloader.append([inputs, targets])
        random.shuffle(train_dataloader)
    else:
        # bsz = 32
        num_train_dataloader = int(np.ceil(len(mol_train)/bsz))
        num_val_dataloader = int(np.ceil(len(mol_val)/bsz))
        num_test_dataloader = int(np.ceil(len(mol_test)/bsz))

        for i in range(num_train_dataloader):
            inputs, targets = PreProcess(mol_train[i * bsz:(i+1)*bsz], smiles_features)
            train_dataloader.append([inputs, targets])

        for i in range(num_val_dataloader):
            inputs, targets = PreProcess(mol_val[i * bsz: (i+1) * bsz], smiles_features)
            val_dataloader.append([inputs, targets])

        for i in range(num_test_dataloader):
            inputs, targets = PreProcess(mol_test[i * bsz: (i+1) * bsz], smiles_features)
            test_dataloader.append([inputs, targets])
    return train_dataloader, val_dataloader,test_dataloader

def generate_scaffold(smiles, include_chirality=False):
    #从给定的 SMILES 字符串中生成 Bemis-Murcko scaffold（分子骨架，去掉了分子上的侧链和其他修饰基团，仅保留核心结构）
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold#生成的 scaffold 的 SMILES 字符串

def scaffold_split(all_smiles_y, size, seed, log):
    #all_smiles_y, size = split_ratio, seed = seed,log = log
    train_size, val_size, test_size = size[0] * len(all_smiles_y), size[1] * len(all_smiles_y), size[2] * len(all_smiles_y)
    train, val, test = [], [], []
    np.testing.assert_almost_equal(size[0] + size[1] + size[2], 1.0)

    all_scaffolds = {}
    for i, data in enumerate(all_smiles_y):
        smiles = data[0]
        scaffold = generate_scaffold(smiles, include_chirality=True)
        # scaffold = generate_scaffold(smiles)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    index_sets = list(all_scaffolds.values())

    big_index_sets = []
    small_index_sets = []

    for index_set in index_sets:
        if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
            big_index_sets.append(index_set)
        else:
            small_index_sets.append(index_set)

    random.seed(seed)
    random.shuffle(big_index_sets)
    random.shuffle(small_index_sets)
    index_sets = big_index_sets + small_index_sets
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0
    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        elif len(val) + len(index_set) <= val_size:
            val += index_set
            val_scaffold_count += 1
        else:
            test += index_set
            test_scaffold_count += 1
    assert len(set(train).intersection(set(val))) == 0  # 确保训练集和验证集之间没有重叠的样本。
    assert len(set(test).intersection(set(val))) == 0  # 确保验证集和测试集之间没有重叠的样本。
    log.debug(f'Total scaffolds = {len(all_scaffolds):,} | '
              f'train scaffolds = {train_scaffold_count:,} | '
              f'val scaffolds = {val_scaffold_count:,} | '
              f'test scaffolds = {test_scaffold_count:,}')

    train_dataset = [all_smiles_y[i] for i in train]
    valid_dataset = [all_smiles_y[i] for i in val]
    test_dataset = [all_smiles_y[i] for i in test]

    return train_dataset, valid_dataset, test_dataset

def random_split(all_smiles_y, size, seed, log):
    #all_smiles_y, size = split_ratio, seed = seed,log = log
    np.testing.assert_almost_equal(size[0] + size[1] + size[2], 1.0)
    num_mols = len(all_smiles_y)
    random.seed(seed)
    all_idx = list(range(num_mols))
    random.shuffle(all_idx)

    train = all_idx[:int(size[0] * num_mols)]
    valid = all_idx[int(size[0] * num_mols):int(size[0] * num_mols) + int(size[1] * num_mols)]
    test = all_idx[int(size[0] * num_mols) + int(size[1] * num_mols):]

    assert len(set(train).intersection(set(valid))) == 0
    assert len(set(valid).intersection(set(test))) == 0
    assert len(train) + len(valid) + len(test) == num_mols

    train_dataset = [all_smiles_y[i] for i in train]
    valid_dataset = [all_smiles_y[i] for i in valid]
    test_dataset = [all_smiles_y[i] for i in test]

    return train_dataset, valid_dataset, test_dataset

def get_motif_max_length(all_smiles_y):
    max_lengths = []

    for smiles, _ in all_smiles_y:
        try:
            # 将SMILES字符串转换为RDKit分子对象
            mol = Chem.MolFromSmiles(smiles)

            if mol is not None:
                cliques = motif_decomp_num(mol)
                # if len(edges) <= 1:
                #     cliques, edges = tree_decomp(mol)
                num_fragments = len(cliques)

            else:
                num_fragments = 0
        except Exception as e:
            # 如果发生错误，碎片数量设置为0
            print(f"Error processing SMILES {smiles}: {e}")
            num_fragments = 0

        # 将碎片数量添加到列表中
        max_lengths.append(num_fragments)

    # 返回最大碎片数量
    return max(max_lengths)

def motif_decomp_num(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]]

    # 初始化全部键的原子索引
    cliques = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1, a2])

    # 使用 BRICS 查找需要断裂的键
    res = list(BRICS.FindBRICSBonds(mol))  # ((5, 4), ('8', '9')) 原子索引的对 (5, 4)
    if len(res) == 0:
        return [list(range(n_atoms))]

    # 根据需要断裂的键调整 cliques
    for bond in res:
        a1, a2 = bond[0][0], bond[0][1]
        if [a1, a2] in cliques:
            cliques.remove([a1, a2])
        else:
            cliques.remove([a2, a1])
        cliques.append([a1])
        cliques.append([a2])

    #合并重叠的碎片
    for c in range(len(cliques) - 1):
        if c >= len(cliques):
            break
        for k in range(c + 1, len(cliques)):
            if k >= len(cliques):
                break
            if len(set(cliques[c]) & set(cliques[k])) > 0:
                cliques[c] = list(set(cliques[c]) | set(cliques[k]))
                cliques[k] = []

    # 去除空的碎片
    cliques = [c for c in cliques if len(c) > 0]

    return cliques

def motif_decomp(mol, mapping):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]],[]

    cliques = []
    breaks = []
    for bond in mol.GetBonds():
        a1 = mapping[bond.GetBeginAtom().GetIdx()]
        a2 = mapping[bond.GetEndAtom().GetIdx()]
        cliques.append([a1, a2])  #初始化全部键的原子索引

    res = list(BRICS.FindBRICSBonds(mol))  #((5, 4), ('8', '9'))原子索引的对 (5, 4)
    if len(res) == 0:
        return [list(range(n_atoms))],[]
    else:
        for bond in res:
            a1, a2 = mapping[bond[0][0]], mapping[bond[0][1]]
            if [a1, a2] in cliques:
                cliques.remove([a1, a2])
            else:
                cliques.remove([a2, a1])#需要断掉的键在初始cliques中，将需要断掉的键删去
            cliques.append([a1])
            cliques.append([a2]) #将删去的键的原子索引单独作为list


    # merge cliques
    for c in range(len(cliques) - 1):
        if c >= len(cliques):
            break
        for k in range(c + 1, len(cliques)):
            if k >= len(cliques):
                break
            if len(set(cliques[c]) & set(cliques[k])) > 0:  # &交集
                cliques[c] = list(set(cliques[c]) | set(cliques[k]))  # ｜并集
                cliques[k] = []
        cliques = [c for c in cliques if len(c) > 0]
    cliques = [c for c in cliques if len(c) > 0]

    edges = []  # 存储基元之间的连接关系
    for bond in res:
        a1, a2 = mapping[bond[0][0]], mapping[bond[0][1]]
        c1, c2 = -1, -1
        for c in range(len(cliques)):
            if a1 in cliques[c]:
                c1 = c
            if a2 in cliques[c]:
                c2 = c
        edges.append((c1, c2))
    for bond in breaks:#+
        for c in range(len(cliques)):
            if bond[0] in cliques[c]:
                c1 = c
            if bond[1] in cliques[c]:
                c2 = c
        edges.append((c1, c2))

    return cliques, edges


def create_adjacency_matrix(cliques, edges,max_motif):
    num_motifs = len(cliques)
    adjacency_matrix = np.zeros((max_motif, max_motif), dtype=int)  # 初始化邻接矩阵为零矩阵

    for c1, c2 in edges:
        adjacency_matrix[c1, c2] = 1  # 设置连接关系
        adjacency_matrix[c2, c1] = 1  # 确保对称性

    # for frag_idx in range(num_motifs):
    #     if frag_idx < max_motif:  # 确保碎片索引在矩阵范围内
    #         adjacency_matrix[frag_idx, frag_idx] = 1
    for frag_idx in range(max_motif):
            adjacency_matrix[frag_idx, frag_idx] = 1

    return adjacency_matrix