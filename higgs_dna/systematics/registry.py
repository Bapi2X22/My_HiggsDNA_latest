from .photon_systematics import (
    photon_pt_scale_dummy,
    Photon_Scale_EGM,
    Photon_Smearing_EGM,
    Scale,
    Smearing,
    energyErrShift,
    FNUF,
    ShowerShape,
    Material,
    PhotonIDMVAShape,
)
from .event_weight_systematics import (
    Pileup,
    L1PreFiring,
    SF_photon_ID,
    LoosePhoIdSF,
    ElectronVetoSF,
    PreselSF,
    TriggerSF,
    leptonTriggerDataEff,
    NNLOPS,
    AlphaS,
    PartonShower,
    cTagSF,
    cTagSF_WPs,
    bTagShapeSF,
    bTagFixedWP,
    bTagFixedWP_bbgg,
    bTagMultiFixedWP,
    Zpt,
    muonSFs,
    electronSFs,
    electron_reco_sf_for_Zee_val_photons,
    atLeast1LeptonIdSF,
    Higgs_plus_HF_syst,
    SF_photon_ID_LM,
    ElectronVetoSF_LM,
    ElectronIndexSF_LM,
    PreselSF_LM,
    TriggerSF_LM,
    Tau_ID,
)
from .jet_systematics import (
    jet_pt_scale_dummy,
    JERC_jet,
)

from .electron_systematics import (
    Electron_Scale_EGM,
    Electron_Smearing_EGM,
)

from .muon_systematics import (
    muon_pt_scare
)

from .tau_systematics import (
    Tau_EnergyScale
)

from .MET_systematics import (
    MET_syst_Unclustered
)

from functools import partial
import logging

logger = logging.getLogger(__name__)

# using add_systematic function of coffea.nanoevents.methods.nanoaod objects as Photon to store object systematics in addition to nominal objects
object_systematics = {
    "PhotonPtScale_dummy": {
        "object": "Photon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": photon_pt_scale_dummy,
        },
    },
    # Traditional EGM scale and smearing
    "Scale_EGM": {
        "object": "Photon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(Photon_Scale_EGM, is_correction=False),
        },
    },
    "ScaleEB_EGM": {
        "object": "Photon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(Photon_Scale_EGM, is_correction=False, restriction="EB"),
        },
    },
    "ScaleEE_EGM": {
        "object": "Photon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(Photon_Scale_EGM, is_correction=False, restriction="EE"),
        },
    },
    "Smearing_EGM": {
        "object": "Photon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(Photon_Smearing_EGM, is_correction=False),
        },
    },
    # IJazZ (Fabrice et al Saclay) scale and smearing
    "Scale_Zee": {
        "object": "Photon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(Scale, is_correction=False, gaussians="2G", restriction=None, is_Zee=True),
        },
    },
    "ScaleEB_Zee": {
        "object": "Photon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(Scale, is_correction=False, gaussians="2G", restriction="EB", is_Zee=True),
        },
    },
    "ScaleEE_Zee": {
        "object": "Photon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(Scale, is_correction=False, gaussians="2G", restriction="EE", is_Zee=True),
        },
    },
    "Scale_Zmmg": {
        "object": "Photon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(Scale, is_correction=False, gaussians="2G", restriction=None, is_Zee=False),
        },
    },
    "ScaleEB_Zmmg": {
        "object": "Photon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(Scale, is_correction=False, gaussians="2G", restriction="EB", is_Zee=False),
        },
    },
    "ScaleEE_Zmmg": {
        "object": "Photon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(Scale, is_correction=False, gaussians="2G", restriction="EE", is_Zee=False),
        },
    },
    "Smearing": {
        "object": "Photon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(Smearing, is_correction=False, gaussians="2G"),
        },
    },
    # Central EGM corrections for Electrons
    "Electron_Scale_EGM": {
        "object": "Electron",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(Electron_Scale_EGM, is_correction=False),
        },
    },
    "Electron_Smearing_EGM": {
        "object": "Electron",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(Electron_Smearing_EGM, is_correction=False),
        },
    },
    "MuonScale": {
        "object": "Muon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(muon_pt_scare, is_correction=False, unc_type="Scale"),
        },
    },
    "MuonResolution": {
        "object": "Muon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(muon_pt_scare, is_correction=False, unc_type="Resolution"),
        },
    },
    "energyErrShift": {
        "object": "Photon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "energyErr",
            "varying_function": partial(energyErrShift, is_correction=False),
        },
    },
    "FNUF": {
        "object": "Photon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(FNUF, year="2017", is_correction=False),
        },
    },
    "ShowerShape": {
        "object": "Photon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(ShowerShape, year="2017", is_correction=False),
        },
    },
    "Material": {
        "object": "Photon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(Material, year="2017", is_correction=False),
        },
    },
    "PhotonIDMVAShape": {
        "object": "Photon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "mvaID",
            "varying_function": partial(PhotonIDMVAShape, is_correction=False),
        },
    },
    "JetPtScale_dummy": {
        "object": "Jet",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(
                jet_pt_scale_dummy, year=None, is_correction=False
            ),
        },
    },
    "JES": {
        "object": "Jet",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(
                JERC_jet,
                year="2022postEE",
                skip_JER=False,
                skip_JEC=False,
                is_correction=False,
            ),
        },
    },
    "MET_unclusteredEnergy": {
        "object": "MET",
        "args": {
            "kind": "UpDownMultiSystematic",
            "what": ["pt", "phi"],
            "varying_function": MET_syst_Unclustered,
        },
    },
    "energyErrShift_LM": {
        "object": "Photon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "energyErr",
            "varying_function": partial(energyErrShift, is_correction=False, workflow="lowmass"),
        },
    },
    "photonIDMVAShift_LM": {
        "object": "Photon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "mvaID",
            "varying_function": partial(PhotonIDMVAShape, is_correction=False, workflow="lowmass"),
        },
    },
    "ShowerShape_LM": {
        "object": "Photon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "mvaID",
            "varying_function": partial(ShowerShape, is_correction=False, workflow="lowmass"),
        },
    },
    "Tau_EnergyScale": {
        "object": "Tau",
        "args": {
            "kind": "UpDownMultiSystematic",
            "what": ["pt", "mass"],
            "varying_function": partial(Tau_EnergyScale, is_correction=False),
        }
    },
}

# functions correcting nominal object quantities to be placed here
# dict containing "name": varying_function
object_corrections = {
    "Scale_EGM": partial(Photon_Scale_EGM, pt=None, is_correction=True),
    "Smearing_EGM": partial(Photon_Smearing_EGM, pt=None, is_correction=True),
    "Electron_Scale_EGM": partial(Electron_Scale_EGM, pt=None, is_correction=True),
    "Electron_Smearing_EGM": partial(Electron_Smearing_EGM, pt=None, is_correction=True),
    "Scale": partial(Scale, pt=None, is_correction=True, gaussians="2G", is_Zee=False),  # Zee and Zmmg scale corrections are applied
    "Scale_Zee": partial(Scale, pt=None, is_correction=True, gaussians="2G", is_Zee=True),  # Only Zee scale corrections is applied
    "Smearing": partial(Smearing, pt=None, is_correction=True, gaussians="2G"),  # The smearing is the same for Zee and Hgg.
    "MuonScaRe": partial(muon_pt_scare, pt=None, is_correction=True),
    "MuonScaRe_PT10": partial(muon_pt_scare, pt=None, is_correction=True, low_pt_threshold=10),
    "energyErrShift": partial(energyErrShift, energyErr=None, is_correction=True),
    "FNUF": partial(FNUF, pt=None, is_correction=True),
    "ShowerShape": partial(ShowerShape, pt=None, is_correction=True),
    "Material": partial(Material, pt=None, is_correction=True),
    "JetPtScale_dummy": partial(
        jet_pt_scale_dummy, pt=None, year=None, is_correction=True
    ),
    "JES": partial(
        JERC_jet,
        pt=None,
        year="2022postEE",
        skip_JER=False,
        skip_JEC=False,
        is_correction=True,
    ),
    "energyErrShift_LM": partial(energyErrShift, energyErr=None, is_correction=True, workflow="lowmass"),
    "ShowerShape_LM": partial(ShowerShape, pt=None, is_correction=True, workflow="lowmass"),
    "Tau_EnergyScale": partial(Tau_EnergyScale, ptmass=None, year=None, is_correction=True),
}

# functions adding systematic variations to event weights to be placed here
# dict containing "name": varying_function
weight_systematics = {
    "Pileup": partial(Pileup, is_correction=False),
    "L1PreFiring": partial(L1PreFiring, is_correction=False),
    "SF_photon_ID": partial(SF_photon_ID, is_correction=False),
    "LoosePhoIDSF": partial(LoosePhoIdSF, is_correction=False),
    "ElectronVetoSF": partial(ElectronVetoSF, is_correction=False),
    "PreselSF": partial(PreselSF, is_correction=False),
    "TriggerSF": partial(TriggerSF, is_correction=False),
    "cTagSF": partial(cTagSF, is_correction=False),
    "cTagSF_WPs": partial(cTagSF_WPs, is_correction=False),
    "deepJet_bTagShapeSF": partial(bTagShapeSF, ShapeSF_name="deepJet_shape", is_correction=False),
    "PNet_bTagShapeSF": partial(bTagShapeSF, ShapeSF_name="particleNet_shape", is_correction=False),
    "bTagFixedWP_PNetLoose": partial(bTagFixedWP, mva_name="particleNet", wp="L", is_correction=False),
    "bTagFixedWP_PNetMedium": partial(bTagFixedWP, mva_name="particleNet", wp="M", is_correction=False),
    "bTagFixedWP_PNetTight": partial(bTagFixedWP, mva_name="particleNet", wp="T", is_correction=False),
    "bTagFixedWP_PNetExtraTight": partial(bTagFixedWP, mva_name="particleNet", wp="XT", is_correction=False),
    "bTagFixedWP_PNetExtraExtraTight": partial(bTagFixedWP, mva_name="particleNet", wp="XXT", is_correction=False),
    "bTagFixedWP_deepJetLoose": partial(bTagFixedWP, mva_name="deepJet", wp="L", is_correction=False),
    "bTagFixedWP_deepJetMedium": partial(bTagFixedWP, mva_name="deepJet", wp="M", is_correction=False),
    "bTagFixedWP_deepJetTight": partial(bTagFixedWP, mva_name="deepJet", wp="T", is_correction=False),
    "bTagFixedWP_deepJetExtraTight": partial(bTagFixedWP, mva_name="deepJet", wp="XT", is_correction=False),
    "bTagFixedWP_deepJetExtraExtraTight": partial(bTagFixedWP, mva_name="deepJet", wp="XXT", is_correction=False),
    "bTagFixedWP_robustParticleTransformerLoose": partial(bTagFixedWP, mva_name="robustParticleTransformer", wp="L", is_correction=False),
    "bTagFixedWP_robustParticleTransformerMedium": partial(bTagFixedWP, mva_name="robustParticleTransformer", wp="M", is_correction=False),
    "bTagFixedWP_robustParticleTransformerTight": partial(bTagFixedWP, mva_name="robustParticleTransformer", wp="T", is_correction=False),
    "bTagFixedWP_robustParticleTransformerExtraTight": partial(bTagFixedWP, mva_name="robustParticleTransformer", wp="XT", is_correction=False),
    "bTagFixedWP_robustParticleTransformerExtraExtraTight": partial(bTagFixedWP, mva_name="robustParticleTransformer", wp="XXT", is_correction=False),
    "bTagFixedWP_UParTAK4Loose": partial(bTagFixedWP, mva_name="UParTAK4", wp="L", is_correction=False),
    "bTagFixedWP_UParTAK4Loose_bbgg": partial(bTagFixedWP_bbgg, mva_name="UParTAK4", wp="L", bTagEffFileName="HtoAAtobbgg", is_correction=False),
    "bTagFixedWP_UParTAK4Medium": partial(bTagFixedWP, mva_name="UParTAK4", wp="M", is_correction=False),
    "bTagFixedWP_UParTAK4Medium_bbgg": partial(bTagFixedWP_bbgg, mva_name="UParTAK4", wp="M", bTagEffFileName="HtoAAtobbgg", is_correction=False),
    "bTagFixedWP_UParTAK4Tight": partial(bTagFixedWP, mva_name="UParTAK4", wp="T", is_correction=False),
    "bTagFixedWP_UParTAK4ExtraTight": partial(bTagFixedWP, mva_name="UParTAK4", wp="XT", is_correction=False),
    "bTagFixedWP_UParTAK4ExtraExtraTight": partial(bTagFixedWP, mva_name="UParTAK4", wp="XXT", is_correction=False),
    "bTagFixedWP_UParTAK4Loose_Run2_v15": partial(bTagFixedWP, mva_name="UParTAK4", wp="L", is_correction=False, is_Run2_v15=True),
    "bTagFixedWP_UParTAK4Medium_Run2_v15": partial(bTagFixedWP, mva_name="UParTAK4", wp="M", is_correction=False, is_Run2_v15=True),
    "bTagFixedWP_UParTAK4Tight_Run2_v15": partial(bTagFixedWP, mva_name="UParTAK4", wp="T", is_correction=False, is_Run2_v15=True),
    "bTagFixedWP_UParTAK4ExtraTight_Run2_v15": partial(bTagFixedWP, mva_name="UParTAK4", wp="XT", is_correction=False, is_Run2_v15=True),
    "bTagFixedWP_UParTAK4ExtraExtraTight_Run2_v15": partial(bTagFixedWP, mva_name="UParTAK4", wp="XXT", is_correction=False, is_Run2_v15=True),
    "bTagMultiFixedWP_UParTAK4LMTXTXXT": partial(bTagMultiFixedWP, mva_name="UParTAK4", wps=["L", "M", "T", "XT", "XXT"], is_correction=False),
    "bTagMultiFixedWP_UParTAK4LMTXTXXT_Run2_v15": partial(bTagMultiFixedWP, mva_name="UParTAK4", wps=["L", "M", "T", "XT", "XXT"], is_correction=False, is_Run2_v15=True),
    "ParT_bTagShapeSF": partial(bTagShapeSF, ShapeSF_name="robustParticleTransformer_shape", is_correction=False),
    "AlphaS": partial(AlphaS),
    "PartonShower": partial(PartonShower),
    # LHEScale and LHEPdf are implemented in the processors themselves (special behavior, taking variations directly from nanoAOD)
    "LHEScale": None,
    "LHEPdf": None,
    "Zpt": partial(Zpt),
    "MuonIdMediumSF": partial(muonSFs, SF_name="NUM_MediumID_DEN_TrackerMuons", is_correction=False),
    "MuonIsoTightSF_IdMedium": partial(muonSFs, SF_name="NUM_TightPFIso_DEN_MediumID", is_correction=False),
    "MuonIsoLooseSF_IdMedium": partial(muonSFs, SF_name="NUM_LoosePFIso_DEN_MediumID", is_correction=False),
    "MuonIsoMediumSF_IdMedium": partial(muonSFs, SF_name="NUM_MediumPFIso_DEN_MediumID", is_correction=False),
    "ElectronIdSFWP90iso": partial(electronSFs, sf_key="wp90iso", is_correction=False),
    "ElectronIdSFWP80iso": partial(electronSFs, sf_key="wp80iso", is_correction=False),
    "ElectronIdSFLoose": partial(electronSFs, sf_key="Loose", is_correction=False),
    "ElectronIdSFMedium": partial(electronSFs, sf_key="Medium", is_correction=False),
    "ElectronIdSFTight": partial(electronSFs, sf_key="Tight", is_correction=False),
    "ElectronRecoSF": partial(electronSFs, sf_key="Reco", is_correction=False),
    "ElectronRecoSF_Zee_val": partial(electron_reco_sf_for_Zee_val_photons, is_correction=False),
    # SF for analyses requiring at least one lepton which can be e or mu
    # Choose combination of WPs for e and mu by name as in example below
    "atLeast1LeptonSF_eleRecoWP90iso_muIDMediumIsoTight": partial(
        atLeast1LeptonIdSF,
        ele_SF_names=("Reco", "wp90iso"),
        mu_SF_names=("NUM_MediumID_DEN_TrackerMuons", "NUM_TightPFIso_DEN_MediumID"),
        is_correction=False,
    ),
    "Higgs_plus_b_pt20_syst50": partial(Higgs_plus_HF_syst, min_pt=20, flav="b", rel_unc=0.5),
    "Higgs_plus_b_pt25_syst50": partial(Higgs_plus_HF_syst, min_pt=25, flav="b", rel_unc=0.5),
    "Higgs_plus_b_pt20_syst100": partial(Higgs_plus_HF_syst, min_pt=20, flav="b", rel_unc=1.0),
    "Higgs_plus_b_pt25_syst100": partial(Higgs_plus_HF_syst, min_pt=25, flav="b", rel_unc=1.0),
    "Higgs_plus_c_pt20_syst50": partial(Higgs_plus_HF_syst, min_pt=20, flav="c", rel_unc=0.5),
    "Higgs_plus_c_pt25_syst50": partial(Higgs_plus_HF_syst, min_pt=25, flav="c", rel_unc=0.5),
    "Higgs_plus_c_pt20_syst100": partial(Higgs_plus_HF_syst, min_pt=20, flav="c", rel_unc=1.0),
    "Higgs_plus_c_pt25_syst100": partial(Higgs_plus_HF_syst, min_pt=25, flav="c", rel_unc=1.0),
    "SF_photon_ID_LM": partial(SF_photon_ID_LM, is_correction=False),
    "ElectronVetoSF_LM": partial(ElectronVetoSF_LM, is_correction=False),
    "ElectronIndexSF_LM": partial(ElectronIndexSF_LM, is_correction=False),
    "PreselSF_LM": partial(PreselSF_LM, is_correction=False),
    "TriggerSF_LM": partial(TriggerSF_LM, is_correction=False),
    "TriggerSF_singleLep": partial(leptonTriggerDataEff, is_correction=False),
    "Tau_ID": partial(Tau_ID, is_correction=False),
}

# functions correcting nominal event weights to be placed here
# dict containing "name": varying_function
weight_corrections = {
    "Pileup": partial(Pileup, is_correction=True),
    "L1PreFiring": partial(L1PreFiring, is_correction=True),
    "SF_photon_ID": partial(SF_photon_ID, is_correction=True),
    "LoosePhoIDSF": partial(LoosePhoIdSF, is_correction=True),
    "ElectronVetoSF": partial(ElectronVetoSF, is_correction=True),
    "PreselSF": partial(PreselSF, is_correction=True),
    "TriggerSF": partial(TriggerSF, is_correction=True),
    "cTagSF": partial(cTagSF, is_correction=True),
    "cTagSF_WPs": partial(cTagSF_WPs, is_correction=True),
    "deepJet_bTagShapeSF": partial(bTagShapeSF, ShapeSF_name="deepJet_shape", is_correction=True),
    "PNet_bTagShapeSF": partial(bTagShapeSF, ShapeSF_name="particleNet_shape", is_correction=True),
    "bTagFixedWP_PNetLoose": partial(bTagFixedWP, mva_name="particleNet", wp="L", is_correction=True),
    "bTagFixedWP_PNetMedium": partial(bTagFixedWP, mva_name="particleNet", wp="M", is_correction=True),
    "bTagFixedWP_PNetTight": partial(bTagFixedWP, mva_name="particleNet", wp="T", is_correction=True),
    "bTagFixedWP_PNetExtraTight": partial(bTagFixedWP, mva_name="particleNet", wp="XT", is_correction=True),
    "bTagFixedWP_PNetExtraExtraTight": partial(bTagFixedWP, mva_name="particleNet", wp="XXT", is_correction=True),
    "bTagFixedWP_deepJetLoose": partial(bTagFixedWP, mva_name="deepJet", wp="L", is_correction=True),
    "bTagFixedWP_deepJetMedium": partial(bTagFixedWP, mva_name="deepJet", wp="M", is_correction=True),
    "bTagFixedWP_deepJetTight": partial(bTagFixedWP, mva_name="deepJet", wp="T", is_correction=True),
    "bTagFixedWP_deepJetExtraTight": partial(bTagFixedWP, mva_name="deepJet", wp="XT", is_correction=True),
    "bTagFixedWP_deepJetExtraExtraTight": partial(bTagFixedWP, mva_name="deepJet", wp="XXT", is_correction=True),
    "bTagFixedWP_robustParticleTransformerLoose": partial(bTagFixedWP, mva_name="robustParticleTransformer", wp="L", is_correction=True),
    "bTagFixedWP_robustParticleTransformerMedium": partial(bTagFixedWP, mva_name="robustParticleTransformer", wp="M", is_correction=True),
    "bTagFixedWP_robustParticleTransformerTight": partial(bTagFixedWP, mva_name="robustParticleTransformer", wp="T", is_correction=True),
    "bTagFixedWP_robustParticleTransformerExtraTight": partial(bTagFixedWP, mva_name="robustParticleTransformer", wp="XT", is_correction=True),
    "bTagFixedWP_robustParticleTransformerExtraExtraTight": partial(bTagFixedWP, mva_name="robustParticleTransformer", wp="XXT", is_correction=True),
    "bTagFixedWP_UParTAK4Loose": partial(bTagFixedWP, mva_name="UParTAK4", wp="L", is_correction=True),
    "bTagFixedWP_UParTAK4Loose_bbgg": partial(bTagFixedWP_bbgg, mva_name="UParTAK4", wp="L", bTagEffFileName="HtoAAtobbgg", is_correction=True),
    "bTagFixedWP_UParTAK4Medium": partial(bTagFixedWP, mva_name="UParTAK4", wp="M", is_correction=True),
    "bTagFixedWP_UParTAK4Medium_bbgg": partial(bTagFixedWP_bbgg, mva_name="UParTAK4", wp="M", bTagEffFileName="HtoAAtobbgg", is_correction=True),
    "bTagFixedWP_UParTAK4Tight": partial(bTagFixedWP, mva_name="UParTAK4", wp="T", is_correction=True),
    "bTagFixedWP_UParTAK4ExtraTight": partial(bTagFixedWP, mva_name="UParTAK4", wp="XT", is_correction=True),
    "bTagFixedWP_UParTAK4ExtraExtraTight": partial(bTagFixedWP, mva_name="UParTAK4", wp="XXT", is_correction=True),
    "bTagFixedWP_UParTAK4Loose_Run2_v15": partial(bTagFixedWP, mva_name="UParTAK4", wp="L", is_correction=True, is_Run2_v15=True),
    "bTagFixedWP_UParTAK4Medium_Run2_v15": partial(bTagFixedWP, mva_name="UParTAK4", wp="M", is_correction=True, is_Run2_v15=True),
    "bTagFixedWP_UParTAK4Tight_Run2_v15": partial(bTagFixedWP, mva_name="UParTAK4", wp="T", is_correction=True, is_Run2_v15=True),
    "bTagFixedWP_UParTAK4ExtraTight_Run2_v15": partial(bTagFixedWP, mva_name="UParTAK4", wp="XT", is_correction=True, is_Run2_v15=True),
    "bTagFixedWP_UParTAK4ExtraExtraTight_Run2_v15": partial(bTagFixedWP, mva_name="UParTAK4", wp="XXT", is_correction=True, is_Run2_v15=True),
    "bTagMultiFixedWP_UParTAK4LMTXTXXT": partial(bTagMultiFixedWP, mva_name="UParTAK4", wps=["L", "M", "T", "XT", "XXT"], is_correction=True),
    "bTagMultiFixedWP_UParTAK4LMTXTXXT_Run2_v15": partial(bTagMultiFixedWP, mva_name="UParTAK4", wps=["L", "M", "T", "XT", "XXT"], is_correction=True, is_Run2_v15=True),
    "ParT_bTagShapeSF": partial(bTagShapeSF, ShapeSF_name="robustParticleTransformer_shape", is_correction=True),
    "NNLOPS": partial(NNLOPS, is_correction=True),  # backwards compatible
    "NNLOPS_amcatnlo": partial(NNLOPS, generator="mcatnlo", is_correction=True),
    "NNLOPS_powheg": partial(NNLOPS, generator="powheg", is_correction=True),
    "Zpt": partial(Zpt, is_correction=True),
    "MuonIdMediumSF": partial(muonSFs, SF_name="NUM_MediumID_DEN_TrackerMuons", is_correction=True),
    "MuonIsoTightSF_IdMedium": partial(muonSFs, SF_name="NUM_TightPFIso_DEN_MediumID", is_correction=True),
    "MuonIsoLooseSF_IdMedium": partial(muonSFs, SF_name="NUM_LoosePFIso_DEN_MediumID", is_correction=True),
    "MuonIsoMediumSF_IdMedium": partial(muonSFs, SF_name="NUM_MediumPFIso_DEN_MediumID", is_correction=True),
    "ElectronIdSFWP90iso": partial(electronSFs, sf_key="wp90iso", is_correction=True),
    "ElectronIdSFWP80iso": partial(electronSFs, sf_key="wp80iso", is_correction=True),
    "ElectronIdSFLoose": partial(electronSFs, sf_key="Loose", is_correction=True),
    "ElectronIdSFMedium": partial(electronSFs, sf_key="Medium", is_correction=True),
    "ElectronIdSFTight": partial(electronSFs, sf_key="Tight", is_correction=True),
    "ElectronRecoSF": partial(electronSFs, sf_key="Reco", is_correction=True),
    "ElectronRecoSF_Zee_val": partial(electron_reco_sf_for_Zee_val_photons, is_correction=True),
    # SF for analyses requiring at least one lepton which can be e or mu
    # Choose combination of WPs for e and mu by name as in example below
    "atLeast1LeptonSF_eleRecoWP90iso_muIDMediumIsoTight": partial(
        atLeast1LeptonIdSF,
        ele_SF_names=("Reco", "wp90iso"),
        mu_SF_names=("NUM_MediumID_DEN_TrackerMuons", "NUM_TightPFIso_DEN_MediumID"),
        is_correction=True,
    ),
    "SF_photon_ID_LM": partial(SF_photon_ID_LM, is_correction=True),
    "ElectronVetoSF_LM": partial(ElectronVetoSF_LM, is_correction=True),
    "ElectronIndexSF_LM": partial(ElectronIndexSF_LM, is_correction=True),
    "PreselSF_LM": partial(PreselSF_LM, is_correction=True),
    "TriggerSF_LM": partial(TriggerSF_LM, is_correction=True),
    "TriggerSF_singleLep": partial(leptonTriggerDataEff, is_correction=True),
    "Tau_ID": partial(Tau_ID, is_correction=True),
}
