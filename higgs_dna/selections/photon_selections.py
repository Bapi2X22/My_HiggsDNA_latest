import awkward as ak
import numpy
from higgs_dna.selections.object_selections import delta_r_mask


# photon preselection for Run3 -> take as input nAOD Photon collection and return the Photons that pass
# cuts (pt, eta, sieie, mvaID, iso... etc)
#
def photon_preselection(
    self,
    photons: ak.Array,
    events: ak.Array,
    electron_veto=True,
    revert_electron_veto=False,
    year="2023",
    IsFlag=False
) -> ak.Array:
    """
    Apply preselection cuts to photons.
    Note that these selections are applied on each photon, it is not based on the diphoton pair.
    """
    # hlt-mimicking cuts
    rho = events.Rho.fixedGridRhoAll * ak.ones_like(photons.pt)
    photon_abs_eta = numpy.abs(photons.eta)
    if year in ["2016", "2016PreVFP", "2016PostVFP", "2017", "2018"]:
        # Run 2, use standard photon preselection
        pass_phoIso_rho_corr_EB = (
            (photon_abs_eta < self.eta_rho_corr)
            & (
                photons.pfPhoIso03 - rho * self.low_eta_rho_corr
                < self.max_pho_iso_EB_low_r9
            )
        ) | (
            # should almost never happen because of the requirement of (photons.isScEtaEB) earlier, thus might be slightly redundant
            (photon_abs_eta > self.eta_rho_corr)
            & (
                photons.pfPhoIso03 - rho * self.high_eta_rho_corr
                < self.max_pho_iso_EB_low_r9
            )
        )

        pass_phoIso_rho_corr_EE = (
            (photon_abs_eta < self.eta_rho_corr)
            & (
                photons.pfPhoIso03 - rho * self.low_eta_rho_corr
                < self.max_pho_iso_EE_low_r9
            )
        ) | (
            (photon_abs_eta > self.eta_rho_corr)
            & (
                photons.pfPhoIso03 - rho * self.high_eta_rho_corr
                < self.max_pho_iso_EE_low_r9
            )
        )
    else:
        # quadratic EA corrections in Run3 : https://indico.cern.ch/event/1204277/contributions/5064356/attachments/2538496/4369369/CutBasedPhotonID_20221031.pdf
        pass_phoIso_rho_corr_EB = (
            ((photon_abs_eta > 0.0) & (photon_abs_eta < 1.0))
            & (
                photons.pfPhoIso03 - (rho * self.EA1_EB1) - (rho * rho * self.EA2_EB1)
                < self.max_pho_iso_EB_low_r9
            )
        ) | (
            ((photon_abs_eta > 1.0) & (photon_abs_eta < 1.4442))
            & (
                photons.pfPhoIso03 - (rho * self.EA1_EB2) - (rho * rho * self.EA2_EB2)
                < self.max_pho_iso_EB_low_r9
            )
        )

        pass_phoIso_rho_corr_EE = (
            (
                ((photon_abs_eta > 1.566) & (photon_abs_eta < 2.0))
                & (
                    photons.pfPhoIso03
                    - (rho * self.EA1_EE1)
                    - (rho * rho * self.EA2_EE1)
                    < self.max_pho_iso_EE_low_r9
                )
            )
            | (
                ((photon_abs_eta > 2.0) & (photon_abs_eta < 2.2))
                & (
                    photons.pfPhoIso03
                    - (rho * self.EA1_EE2)
                    - (rho * rho * self.EA2_EE2)
                    < self.max_pho_iso_EE_low_r9
                )
            )
            | (
                ((photon_abs_eta > 2.2) & (photon_abs_eta < 2.3))
                & (
                    photons.pfPhoIso03
                    - (rho * self.EA1_EE3)
                    - (rho * rho * self.EA2_EE3)
                    < self.max_pho_iso_EE_low_r9
                )
            )
            | (
                ((photon_abs_eta > 2.3) & (photon_abs_eta < 2.4))
                & (
                    photons.pfPhoIso03
                    - (rho * self.EA1_EE4)
                    - (rho * rho * self.EA2_EE4)
                    < self.max_pho_iso_EE_low_r9
                )
            )
            | (
                ((photon_abs_eta > 2.4) & (photon_abs_eta < 2.5))
                & (
                    photons.pfPhoIso03
                    - (rho * self.EA1_EE5)
                    - (rho * rho * self.EA2_EE5)
                    < self.max_pho_iso_EE_low_r9
                )
            )
        )

    isEB_high_r9 = (photons.isScEtaEB) & (photons.r9 > self.min_full5x5_r9_EB_high_r9)
    isEE_high_r9 = (photons.isScEtaEE) & (photons.r9 > self.min_full5x5_r9_EE_high_r9)
    iso = photons.trkSumPtHollowConeDR03 if hasattr(photons, "trkSumPtHollowConeDR03") else photons.pfChargedIsoPFPV  # photons.pfChargedIsoPFPV for v11, photons.trkSumPtHollowConeDR03 v12 and above
    rel_iso = photons.pfRelIso03_chg if hasattr(photons, "pfRelIso03_chg") else photons.pfRelIso03_chg_quadratic  # photons.pfRelIso03_chg for v1?, photons.pfRelIso03_chg_quadratic v12 and above
    isEB_low_r9 = (
        (photons.isScEtaEB)
        & (photons.r9 > self.min_full5x5_r9_EB_low_r9)
        & (photons.r9 < self.min_full5x5_r9_EB_high_r9)
        & (
            iso
            < self.max_trkSumPtHollowConeDR03_EB_low_r9
        )
        & (photons.sieie < self.max_sieie_EB_low_r9)
        & (pass_phoIso_rho_corr_EB)
    )
    isEE_low_r9 = (
        (photons.isScEtaEE)
        & (photons.r9 > self.min_full5x5_r9_EE_low_r9)
        & (photons.r9 < self.min_full5x5_r9_EE_high_r9)
        & (
            iso
            < self.max_trkSumPtHollowConeDR03_EE_low_r9
        )
        & (photons.sieie < self.max_sieie_EE_low_r9)
        & (pass_phoIso_rho_corr_EE)
    )

    if electron_veto:
        e_veto_cut = (photons.electronVeto == 1)
    elif revert_electron_veto:
        e_veto_cut = (photons.electronVeto == 0)
    else:
        e_veto_cut = ak.ones_like(photons.electronVeto, dtype=bool)

    if IsFlag:
        photons["PassPresel"] = (
            e_veto_cut
            & (photons.pt > self.min_pt_photon)
            & (photons.isScEtaEB | photons.isScEtaEE)
            & (photons.mvaID > self.min_mvaid)
            & (photons.hoe < self.max_hovere)
            & (
                (photons.r9 > self.min_full5x5_r9)
                | (
                    rel_iso * photons.pt < self.max_chad_iso
                )
                | (rel_iso < self.max_chad_rel_iso)
            )
            & (isEB_high_r9 | isEB_low_r9 | isEE_high_r9 | isEE_low_r9)
        )
        return photons
    else:
        return photons[
            e_veto_cut
            & (photons.pt > self.min_pt_photon)
            & (photons.isScEtaEB | photons.isScEtaEE)
            & (photons.mvaID > self.min_mvaid)
            & (photons.hoe < self.max_hovere)
            & (
                (photons.r9 > self.min_full5x5_r9)
                | (
                    rel_iso * photons.pt < self.max_chad_iso
                )
                | (rel_iso < self.max_chad_rel_iso)
            )
            & (isEB_high_r9 | isEB_low_r9 | isEE_high_r9 | isEE_low_r9)
        ]

def photon_preselection_bbgg(
    self,
    photons: ak.Array,
    events: ak.Array,
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

    return photons[
        e_veto_cut
        & (~photons.pixelSeed)
        & (photons.pt > self.min_pt_photon)
        & (photons.isScEtaEB | photons.isScEtaEE)
        & dr_cut
        & dr_cut_muon
        &(ak.where(photons.isScEtaEB, photons.mvaID > self.min_mvaid_EB, photons.mvaID > self.min_mvaid_EE))
    ]
