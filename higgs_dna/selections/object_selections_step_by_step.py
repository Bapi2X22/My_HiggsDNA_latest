import awkward as ak
import numpy
from higgs_dna.selections.object_selections import delta_r_mask
from higgs_dna.tools.jetID import add_jetId
import awkward as ak
import correctionlib
import os
from coffea.analysis_tools import PackedSelection
import numpy as np
from correctionlib.highlevel import model_auto, open_auto
import json
import logging

logger = logging.getLogger(__name__)



def enabled_mask(mask, objects, enabled):
    if enabled:
        return mask
    return ak.ones_like(objects.pt, dtype=bool)



def photon_preselection_bbgg_mask(
    self,
    photons: ak.Array,
    electrons: ak.Array,
    muons: ak.Array,
    electron_veto=True,
    revert_electron_veto=False,
    year="2023",
    IsFlag=False
) -> ak.Array:
    """
    Apply preselection cuts to photons.
    Note that these selections are applied on each photon, it is not based on the diphoton pair.
    """

    if electron_veto:
        e_veto_cut = (photons.electronVeto == 1)
    elif revert_electron_veto:
        e_veto_cut = (photons.electronVeto == 0)
    else:
        e_veto_cut = ak.ones_like(photons.electronVeto, dtype=bool)

    dr_cut = delta_r_mask(photons, electrons, 0.2)
    dr_cut_muon = delta_r_mask(photons, muons, 0.2)

    return(
        e_veto_cut
        & enabled_mask(~photons.pixelSeed, photons, self.apply_pixel_seed_veto)
        & enabled_mask(
            photons.pt > self.min_pt_photon,
            photons,
            self.apply_photon_pt,
        )
        & (photons.isScEtaEB | photons.isScEtaEE)
        & enabled_mask(
            dr_cut,
            photons,
            self.apply_electron_photon_dr,
        )
        & enabled_mask(
            dr_cut_muon,
            photons,
            self.apply_muon_photon_dr,
        )
        & enabled_mask(
            ak.where(
                photons.isScEtaEB,
                photons.mvaID > self.min_mvaid_EB,
                photons.mvaID > self.min_mvaid_EE,
            ),
            photons,
            self.apply_photon_mvaid,
        )
    )



def photon_preselection_bbgg(
    self,
    photons: ak.Array,
    electrons: ak.Array,
    muons: ak.Array,
    electron_veto=True,
    revert_electron_veto=False,
    year="2023",
    IsFlag=False,
) -> ak.Array:

    """
    Hierarchical photon selection.

    Hard requirements:
        1. Pixel seed veto
        2. Detector acceptance
        3. Electron veto (depending on configuration)

    Sequential quality requirements:
        C1: pT > minimum photon pT
        C2: C1 + Photon MVA ID
        C3: C2 + Electron isolation
        C4: C3 + Muon isolation

    Selection:
        - First try C4.
        - If fewer than two photons, relax to C3.
        - If fewer than two photons, relax to C2.
        - If fewer than two photons, relax to C1.
        - If fewer than two photons, relax pT as well (C0).

    At each stage, select the two highest-pT photons.

    Returns at most two photons per event.
    """

    # --------------------------------------------------------
    # Hard requirements
    # --------------------------------------------------------

    if electron_veto:
        e_veto_cut = photons.electronVeto == 1

    elif revert_electron_veto:
        e_veto_cut = photons.electronVeto == 0

    else:
        e_veto_cut = ak.ones_like(
            photons.electronVeto,
            dtype=bool
        )

    acceptance = photons.isScEtaEB | photons.isScEtaEE

    pixel_pass = ~photons.pixelSeed

    hard_mask = (
        e_veto_cut
        & acceptance
        & pixel_pass
    )

    # --------------------------------------------------------
    # Individual quality criteria
    # --------------------------------------------------------

    pt_pass = photons.pt > self.min_pt_photon

    mva_pass = ak.where(
        photons.isScEtaEB,
        photons.mvaID > self.min_mvaid_EB,
        photons.mvaID > self.min_mvaid_EE,
    )

    ele_iso_pass = delta_r_mask(
        photons, electrons, 0.2
    )

    mu_iso_pass = delta_r_mask(
        photons, muons, 0.2
    )

    # --------------------------------------------------------
    # Cumulative selection stages
    # --------------------------------------------------------

    cut_1 = pt_pass

    cut_2 = (
        cut_1
        & mva_pass
    )

    cut_3 = (
        cut_2
        & ele_iso_pass
    )

    cut_4 = (
        cut_3
        & mu_iso_pass
    )

    # Apply hard requirements
    cut_1 = cut_1 & hard_mask
    cut_2 = cut_2 & hard_mask
    cut_3 = cut_3 & hard_mask
    cut_4 = cut_4 & hard_mask

    # --------------------------------------------------------
    # Count photons satisfying each cumulative stage
    # --------------------------------------------------------

    n4 = ak.sum(cut_4, axis=1)
    n3 = ak.sum(cut_3, axis=1)
    n2 = ak.sum(cut_2, axis=1)
    n1 = ak.sum(cut_1, axis=1)

    # --------------------------------------------------------
    # Determine selection stage independently per event
    # --------------------------------------------------------

    use_cut_4 = n4 >= 2

    use_cut_3 = (
        (n4 < 2)
        & (n3 >= 2)
    )

    use_cut_2 = (
        (n4 < 2)
        & (n3 < 2)
        & (n2 >= 2)
    )

    use_cut_1 = (
        (n4 < 2)
        & (n3 < 2)
        & (n2 < 2)
        & (n1 >= 2)
    )

    # Final fallback: relax pT requirement too,
    # while retaining all hard requirements.
    use_cut_0 = n1 < 2

    # --------------------------------------------------------
    # Broadcast event-level masks to photon-level arrays
    # --------------------------------------------------------

    use_cut_4, _ = ak.broadcast_arrays(
        use_cut_4, cut_1
    )

    use_cut_3, _ = ak.broadcast_arrays(
        use_cut_3, cut_1
    )

    use_cut_2, _ = ak.broadcast_arrays(
        use_cut_2, cut_1
    )

    use_cut_1, _ = ak.broadcast_arrays(
        use_cut_1, cut_1
    )

    use_cut_0, _ = ak.broadcast_arrays(
        use_cut_0, cut_1
    )

    # --------------------------------------------------------
    # Select photons from the highest available stage
    # --------------------------------------------------------

    final_mask = (
        (use_cut_4 & cut_4)
        | (use_cut_3 & cut_3)
        | (use_cut_2 & cut_2)
        | (use_cut_1 & cut_1)
        | (use_cut_0 & hard_mask)
    )

    candidates = photons[final_mask]

    # --------------------------------------------------------
    # Sort by descending pT and retain leading two
    # --------------------------------------------------------

    order = ak.argsort(
        candidates.pt,
        axis=1,
        ascending=False,
        stable=True
    )

    candidates = candidates[order]

    candidates = candidates[:, :2]

    return candidates




def select_muons_bbgg_mask(
    self,
    muons: ak.highlevel.Array
) -> ak.highlevel.Array:

    pt_cut = muons.pt > self.muon_pt_threshold
    eta_cut = abs(muons.eta) < self.muon_max_eta
    # Hard cuts
    hard_mask = (
        (pt_cut)
        & (eta_cut)
        & (
            abs(muons.dxy) < self.muon_max_dxy
            if self.muon_max_dxy is not None
            else muons.pt > 0
        )
        & (
            abs(muons.dz) < self.muon_max_dz
            if self.muon_max_dz is not None
            else muons.pt > 0
        )
    )

    if self.mu_id_wp == "tight":
        id_cut = muons.tightId
    elif self.mu_id_wp == "medium":
        id_cut = muons.mediumId
    elif self.mu_id_wp == "loose":
        id_cut = muons.looseId
    else:
        id_cut = muons.pt > 0

    # print(muons.fields)  # DEBUG

    # if I understand https://twiki.cern.ch/twiki/bin/view/CMS/MuonRun32022#Medium_pT_15_GeV_to_200_GeV correctly, only loose and tight are PF isos are supported with a SF (so far?)
    # also very loose, very tight and very very tight WPs are available: 1=PFIsoVeryLoose, 2=PFIsoLoose, 3=PFIsoMedium, 4=PFIsoTight, 5=PFIsoVeryTight, 6=PFIsoVeryVeryTight
    if self.mu_iso_wp == "tight":
        iso_cut = muons.pfIsoId >= 4
    elif self.mu_iso_wp == "medium":
        iso_cut = muons.pfIsoId >= 3
    elif self.mu_iso_wp == "loose":
        iso_cut = muons.pfIsoId >= 2
    else:
        iso_cut = muons.pt > 0

    if self.global_muon:
        global_cut = muons.isGlobal
    else:
        global_cut = muons.pt > 0

    # return pt_cut & eta_cut & id_cut & iso_cut & global_cut & dxy_cut & dz_cut

    selected_mask = hard_mask & id_cut & iso_cut & global_cut

    return selected_mask, (
        hard_mask
        & enabled_mask(id_cut, muons, self.apply_muon_id)
        & enabled_mask(iso_cut, muons, self.apply_muon_iso)
        & enabled_mask(global_cut, muons, self.apply_global_muon)
    )




def select_muons_bbgg(
    self,
    muons: ak.highlevel.Array,
) -> tuple[ak.Array, ak.Array]:

    pt_cut = muons.pt > self.muon_pt_threshold
    eta_cut = abs(muons.eta) < self.muon_max_eta

    # --------------------------------------------------------
    # Hard cuts
    # --------------------------------------------------------

    hard_mask = (
        pt_cut
        & eta_cut
        & (
            abs(muons.dxy) < self.muon_max_dxy
            if self.muon_max_dxy is not None
            else muons.pt > 0
        )
        & (
            abs(muons.dz) < self.muon_max_dz
            if self.muon_max_dz is not None
            else muons.pt > 0
        )
    )

    muons = muons[hard_mask]

    # --------------------------------------------------------
    # Muon ID
    # --------------------------------------------------------

    if self.mu_id_wp == "tight":
        id_cut = muons.tightId
    elif self.mu_id_wp == "medium":
        id_cut = muons.mediumId
    elif self.mu_id_wp == "loose":
        id_cut = muons.looseId
    else:
        id_cut = muons.pt > 0

    # --------------------------------------------------------
    # Muon isolation
    # --------------------------------------------------------

    if self.mu_iso_wp == "tight":
        iso_cut = muons.pfIsoId >= 4
    elif self.mu_iso_wp == "medium":
        iso_cut = muons.pfIsoId >= 3
    elif self.mu_iso_wp == "loose":
        iso_cut = muons.pfIsoId >= 2
    else:
        iso_cut = muons.pt > 0

    # --------------------------------------------------------
    # Global muon requirement
    # --------------------------------------------------------

    global_cut = (
        muons.isGlobal
        if self.global_muon
        else muons.pt > 0
    )

    # --------------------------------------------------------
    # Cumulative selection
    # --------------------------------------------------------

    cut_1 = id_cut

    cut_2 = (
        cut_1
        & iso_cut
    )

    cut_3 = (
        cut_2
        & global_cut
    )

    # --------------------------------------------------------
    # Sort by pT first (lowest priority)
    # --------------------------------------------------------

    pt_order = ak.argsort(
        muons.pt,
        axis=1,
        ascending=False,
        stable=True,
    )

    muons = muons[pt_order]
    cut_1 = cut_1[pt_order]
    cut_2 = cut_2[pt_order]
    cut_3 = cut_3[pt_order]

    # --------------------------------------------------------
    # Lexicographical ranking:
    # 1. ID + Isolation + Global
    # 2. ID + Isolation
    # 3. ID
    # 4. pT breaks ties
    # --------------------------------------------------------

    muons = muons[ak.argsort(
        cut_1,
        axis=1,
        ascending=False,
        stable=True,
    )]

    cut_2 = cut_2[ak.argsort(
        cut_1,
        axis=1,
        ascending=False,
        stable=True,
    )]

    cut_3 = cut_3[ak.argsort(
        cut_1,
        axis=1,
        ascending=False,
        stable=True,
    )]

    # Rank by isolation within the ID-passing group
    order_iso = ak.argsort(
        cut_2,
        axis=1,
        ascending=False,
        stable=True,
    )

    muons = muons[order_iso]
    cut_3 = cut_3[order_iso]

    # Rank by global muon within the ID + isolation group
    order_global = ak.argsort(
        cut_3,
        axis=1,
        ascending=False,
        stable=True,
    )

    best_muons = muons[order_global]

    # --------------------------------------------------------
    # Fully selected muons
    # --------------------------------------------------------

    selected_mask = id_cut & iso_cut & global_cut

    selected_muons = muons[
        selected_mask[pt_order]
    ]

    return best_muons, selected_muons




def select_electrons_bbgg_mask(
    self,
    electrons: ak.highlevel.Array
) -> ak.highlevel.Array:
    pt_cut = electrons.pt > self.electron_pt_threshold

    eta_cut = abs(electrons.eta) < self.electron_max_eta

    if self.el_id_wp == "WP90":
        id_cut = electrons.mvaIso_WP90
    elif self.el_id_wp == "WP80":
        id_cut = electrons.mvaIso_WP80
    elif self.el_id_wp == "loose":
        id_cut = electrons.cutBased >= 2
    # WPL is not supported anymore with the Run 3 electron ID, CMSSW 130X v12 nanoAODs only have WP80 and WP90 options
    # elif self.el_id_wp == "WPL":
    #    id_cut = electrons.mvaIso_WPL
    else:
        id_cut = electrons.pt > 0.

    dxy_cut = abs(electrons.dxy) < self.electron_max_dxy if self.electron_max_dxy is not None else electrons.pt > 0
    dz_cut = abs(electrons.dz) < self.electron_max_dz if self.electron_max_dz is not None else electrons.pt > 0

    hard_mask = pt_cut & eta_cut & dxy_cut & dz_cut

    selected_mask = hard_mask & id_cut

    return selected_mask, (
        hard_mask
        & enabled_mask(id_cut, electrons, self.apply_electron_id)
    )


def select_electrons_bbgg(
    self,
    electrons: ak.highlevel.Array,
) -> tuple[ak.Array, ak.Array]:

    # Hard cuts
    pt_cut = electrons.pt > self.electron_pt_threshold
    eta_cut = abs(electrons.eta) < self.electron_max_eta

    dxy_cut = (
        abs(electrons.dxy) < self.electron_max_dxy
        if self.electron_max_dxy is not None
        else electrons.pt > 0
    )

    dz_cut = (
        abs(electrons.dz) < self.electron_max_dz
        if self.electron_max_dz is not None
        else electrons.pt > 0
    )

    hard_mask = pt_cut & eta_cut & dxy_cut & dz_cut

    # Electron ID
    if self.el_id_wp == "WP90":
        id_cut = electrons.mvaIso_WP90
    elif self.el_id_wp == "WP80":
        id_cut = electrons.mvaIso_WP80
    elif self.el_id_wp == "loose":
        id_cut = electrons.cutBased >= 2
    else:
        id_cut = electrons.pt > 0.

    # Apply hard cuts
    hard_electrons = electrons[hard_mask]
    hard_id_pass = id_cut[hard_mask]

    # Rank by pT first (secondary criterion)
    pt_order = ak.argsort(
        hard_electrons.pt,
        axis=1,
        ascending=False,
        stable=True,
    )

    hard_electrons = hard_electrons[pt_order]
    hard_id_pass = hard_id_pass[pt_order]

    # Rank ID-passing electrons first (primary criterion)
    id_order = ak.argsort(
        ak.values_astype(hard_id_pass, "int32"),
        axis=1,
        ascending=False,
        stable=True,
    )

    best_electrons = hard_electrons[id_order]

    # Electrons passing all cuts, ordered by descending pT
    selected_electrons = hard_electrons[hard_id_pass]

    return best_electrons, selected_electrons


def select_jets_bbgg_mask(
    self,
    jets: ak.highlevel.Array,
    diphotons: ak.highlevel.Array,
    muons: ak.highlevel.Array,
    electrons: ak.highlevel.Array,
    taus: ak.highlevel.Array = None,
) -> ak.highlevel.Array:
    
    print("Printing jet fields for debugging: ", jets.fields)
    # jet id selection:
    # https://twiki.cern.ch/twiki/bin/view/CMS/JetID13p6TeV#nanoAOD_Flags
    if (self.nano_version == 12) or (self.nano_version == 13):
        passJetIdTight, passJetIdTightLepVeto = jetIdFlags_v1213(
            jets, self.nano_version
        )
        if self.jet_jetId == "tight":  # Select jetId 2 or 6
            logger.info(
                "Applying jetID recipe of NanoAOD version %s", self.nano_version
            )
            jetId_cut = passJetIdTight
        elif self.jet_jetId == "tightLepVeto":  # Select jetId 6
            logger.info(
                "Applying jetID recipe of NanoAOD version %s", self.nano_version
            )
            jetId_cut = passJetIdTight & passJetIdTightLepVeto
        else:
            jetId_cut = ak.ones_like(jets.pt) > 0
            logger.warning("[ select_jets ] - No JetId applied")
    else:
        if self.jet_jetId == "tight":
            jetId_cut = jets.jetId >= 2
        elif self.jet_jetId == "tightLepVeto":
            jetId_cut = jets.jetId == 6
        else:
            jetId_cut = ak.ones_like(jets.pt) > 0
            logger.warning("[ select_jets ] - No JetId applied")
    logger.debug(
        f"[ select_jets ] - Total: {ak.sum(ak.flatten((ak.ones_like(jets.pt) > 0)))} - Pass tight jetId: {ak.sum(ak.flatten(jetId_cut))}"
    )
    pt_cut = jets.pt > self.jet_pt_threshold
    eta_cut = abs(jets.eta) < self.jet_max_eta
    dr_dipho_cut = ak.ones_like(pt_cut) > 0
    if (self.clean_jet_dipho) & (ak.num(diphotons.pt, axis=0) > 0):
        dr_dipho_cut = delta_r_mask(jets, diphotons, self.jet_dipho_min_dr)

    if (self.clean_jet_pho) & (ak.num(diphotons.pt, axis=0) > 0):
        lead = ak.zip(
            {
                "pt": diphotons.pho_lead.pt,
                "eta": diphotons.pho_lead.eta,
                "phi": diphotons.pho_lead.phi,
                "mass": diphotons.pho_lead.mass,
                "charge": diphotons.pho_lead.charge,
            }
        )
        lead = ak.with_name(lead, "PtEtaPhiMCandidate")
        sublead = ak.zip(
            {
                "pt": diphotons.pho_sublead.pt,
                "eta": diphotons.pho_sublead.eta,
                "phi": diphotons.pho_sublead.phi,
                "mass": diphotons.pho_sublead.mass,
                "charge": diphotons.pho_sublead.charge,
            }
        )
        jets = ak.with_name(jets, "PtEtaPhiMCandidate")
        sublead = ak.with_name(sublead, "PtEtaPhiMCandidate")
        dr_pho_lead_cut = delta_r_mask(jets, lead, self.jet_pho_min_dr)
        dr_pho_sublead_cut = delta_r_mask(jets, sublead, self.jet_pho_min_dr)
    else:
        dr_pho_lead_cut = jets.pt > -1
        dr_pho_sublead_cut = jets.pt > -1

    if (self.clean_jet_ele) & (ak.num(electrons.pt, axis=0) > 0):
        jets = ak.with_name(jets, "PtEtaPhiMCandidate")
        dr_electrons_cut = delta_r_mask(jets, electrons, self.jet_ele_min_dr)
    else:
        dr_electrons_cut = jets.pt > -1

    if (self.clean_jet_muo) & (ak.num(muons.pt, axis=0) > 0):
        jets = ak.with_name(jets, "PtEtaPhiMCandidate")
        dr_muons_cut = delta_r_mask(jets, muons, self.jet_muo_min_dr)
    else:
        dr_muons_cut = jets.pt > -1

    if taus is not None:
        if (self.clean_jet_tau) & (ak.num(taus.pt, axis=0) > 0):
            jets = ak.with_name(jets, "PtEtaPhiMCandidate")
            dr_taus_cut = delta_r_mask(jets, taus, self.jet_tau_min_dr)
        else:
            dr_taus_cut = jets.pt > -1
    else:
        dr_taus_cut = jets.pt > -1

    return (
        enabled_mask(jetId_cut, jets, self.apply_jet_id)
        & enabled_mask(pt_cut & eta_cut, jets, self.apply_jet_pt_eta)
        & enabled_mask(
            dr_pho_lead_cut & dr_pho_sublead_cut,
            jets,
            self.apply_photon_jet_dr,
        )
        & enabled_mask(
            dr_electrons_cut, jets, self.apply_electron_jet_dr
        )
        & enabled_mask(
            dr_muons_cut, jets, self.apply_muon_jet_dr
        )
        & dr_dipho_cut
        & dr_taus_cut
    )


def select_jets_bbgg(
    self,
    jets: ak.highlevel.Array,
    diphotons: ak.highlevel.Array,
    muons: ak.highlevel.Array,
    electrons: ak.highlevel.Array,
    taus: ak.highlevel.Array = None,
) -> ak.highlevel.Array:
    
    print("Printing jet fields for debugging: ", jets.fields)
    # jet id selection:
    # https://twiki.cern.ch/twiki/bin/view/CMS/JetID13p6TeV#nanoAOD_Flags
    if (self.nano_version == 12) or (self.nano_version == 13):
        passJetIdTight, passJetIdTightLepVeto = jetIdFlags_v1213(
            jets, self.nano_version
        )
        if self.jet_jetId == "tight":  # Select jetId 2 or 6
            logger.info(
                "Applying jetID recipe of NanoAOD version %s", self.nano_version
            )
            jetId_cut = passJetIdTight
        elif self.jet_jetId == "tightLepVeto":  # Select jetId 6
            logger.info(
                "Applying jetID recipe of NanoAOD version %s", self.nano_version
            )
            jetId_cut = passJetIdTight & passJetIdTightLepVeto
        else:
            jetId_cut = ak.ones_like(jets.pt) > 0
            logger.warning("[ select_jets ] - No JetId applied")
    else:
        if self.jet_jetId == "tight":
            jetId_cut = jets.jetId >= 2
        elif self.jet_jetId == "tightLepVeto":
            jetId_cut = jets.jetId == 6
        else:
            jetId_cut = ak.ones_like(jets.pt) > 0
            logger.warning("[ select_jets ] - No JetId applied")
    logger.debug(
        f"[ select_jets ] - Total: {ak.sum(ak.flatten((ak.ones_like(jets.pt) > 0)))} - Pass tight jetId: {ak.sum(ak.flatten(jetId_cut))}"
    )
    pt_cut = jets.pt > self.jet_pt_threshold
    eta_cut = abs(jets.eta) < self.jet_max_eta
    dr_dipho_cut = ak.ones_like(pt_cut) > 0
    if (self.clean_jet_dipho) & (ak.num(diphotons.pt, axis=0) > 0):
        dr_dipho_cut = delta_r_mask(jets, diphotons, self.jet_dipho_min_dr)

    if (self.clean_jet_pho) & (ak.num(diphotons.pt, axis=0) > 0):
        lead = ak.zip(
            {
                "pt": diphotons.pho_lead.pt,
                "eta": diphotons.pho_lead.eta,
                "phi": diphotons.pho_lead.phi,
                "mass": diphotons.pho_lead.mass,
                "charge": diphotons.pho_lead.charge,
            }
        )
        lead = ak.with_name(lead, "PtEtaPhiMCandidate")
        sublead = ak.zip(
            {
                "pt": diphotons.pho_sublead.pt,
                "eta": diphotons.pho_sublead.eta,
                "phi": diphotons.pho_sublead.phi,
                "mass": diphotons.pho_sublead.mass,
                "charge": diphotons.pho_sublead.charge,
            }
        )
        jets = ak.with_name(jets, "PtEtaPhiMCandidate")
        sublead = ak.with_name(sublead, "PtEtaPhiMCandidate")
        dr_pho_lead_cut = delta_r_mask(jets, lead, self.jet_pho_min_dr)
        dr_pho_sublead_cut = delta_r_mask(jets, sublead, self.jet_pho_min_dr)
    else:
        dr_pho_lead_cut = jets.pt > -1
        dr_pho_sublead_cut = jets.pt > -1

    if (self.clean_jet_ele) & (ak.num(electrons.pt, axis=0) > 0):
        jets = ak.with_name(jets, "PtEtaPhiMCandidate")
        dr_electrons_cut = delta_r_mask(jets, electrons, self.jet_ele_min_dr)
    else:
        dr_electrons_cut = jets.pt > -1

    if (self.clean_jet_muo) & (ak.num(muons.pt, axis=0) > 0):
        jets = ak.with_name(jets, "PtEtaPhiMCandidate")
        dr_muons_cut = delta_r_mask(jets, muons, self.jet_muo_min_dr)
    else:
        dr_muons_cut = jets.pt > -1

    if taus is not None:
        if (self.clean_jet_tau) & (ak.num(taus.pt, axis=0) > 0):
            jets = ak.with_name(jets, "PtEtaPhiMCandidate")
            dr_taus_cut = delta_r_mask(jets, taus, self.jet_tau_min_dr)
        else:
            dr_taus_cut = jets.pt > -1
    else:
        dr_taus_cut = jets.pt > -1

    # Medium b-tag working point
    btag_cut = jets.btagUParTAK4B > 0.1272

    # Quality score: number of criteria passed
    score = (
        ak.values_astype(jetId_cut, "int32")
        + ak.values_astype(pt_cut, "int32")
        + ak.values_astype(eta_cut, "int32")
        + ak.values_astype(dr_pho_lead_cut, "int32")
        + ak.values_astype(dr_pho_sublead_cut, "int32")
        + ak.values_astype(dr_electrons_cut, "int32")
        + ak.values_astype(dr_muons_cut, "int32")
        + ak.values_astype(dr_dipho_cut, "int32")
        + ak.values_astype(dr_taus_cut, "int32")
    )

    # First sort by pT descending (tie-breaker)
    pt_order = ak.argsort(
        jets.pt,
        axis=1,
        ascending=False,
        stable=True,
    )

    jets = jets[pt_order]
    score = score[pt_order]
    btag_cut = btag_cut[pt_order]

    # Then sort by quality score descending
    score_order = ak.argsort(
        score,
        axis=1,
        ascending=False,
        stable=True,
    )

    jets = jets[score_order]
    btag_cut = btag_cut[score_order]

    # Finally, medium b-tag passing jets first
    btag_order = ak.argsort(
        ak.values_astype(btag_cut, "int32"),
        axis=1,
        ascending=False,
        stable=True,
    )

    selected_jets = jets[jetId_cut & pt_cut & eta_cut & dr_pho_lead_cut & dr_pho_sublead_cut & dr_electrons_cut & dr_muons_cut & dr_dipho_cut & dr_taus_cut]

    return selected_jets, jets[btag_order]

