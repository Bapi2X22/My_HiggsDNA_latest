from coffea import processor
from abc import abstractmethod
import awkward as ak
import numpy

from higgs_dna.tools.xgb_loader import load_bdt
from higgs_dna.tools.chained_quantile import ChainedQuantileRegression
from higgs_dna.metaconditions import photon_id_mva_weights
from higgs_dna.tools.photonid_mva import calculate_photonid_mva, load_photonid_mva
from higgs_dna.tools.photonid_mva import calculate_photonid_mva_run3, load_photonid_mva_run3
from higgs_dna.metaconditions import diphoton as diphoton_mva_dir
from higgs_dna.tools.diphoton_mva import calculate_retrained_diphoton_mva as calculate_diphoton_mva
from higgs_dna.utils.misc_utils import infer_nano_version

from typing import Any, Dict, List, Optional

import functools
import operator
import os
import warnings

import logging

logger = logging.getLogger(__name__)


class HggSkeletonProcessor(processor.ProcessorABC):  # type: ignore
    # Generic class-level defaults
    # muon selection cuts
    muon_pt_threshold = 24
    muon_max_eta = 2.4
    mu_id_wp = "medium"
    mu_iso_wp = "medium"
    muon_photon_min_dr = 0.2
    global_muon = True
    muon_max_dxy = None
    muon_max_dz = None

    # electron selection cuts
    electron_pt_threshold = 30
    electron_max_eta = 2.5
    electron_photon_min_dr = 0.2
    el_id_wp = "WP80"  # this includes isolation
    electron_max_dxy = None
    electron_max_dz = None

    # jet selection cuts
    jet_jetId = "tight"  # can be "tightLepVeto" or "tight": https://twiki.cern.ch/twiki/bin/view/CMS/JetID13p6TeV#nanoAOD_Flags
    jet_dipho_min_dr = 0.4
    jet_pho_min_dr = 0.4
    jet_ele_min_dr = 0.4
    jet_muo_min_dr = 0.4
    jet_pt_threshold = 20
    jet_max_eta = 2.4
    bjet_mva = "particleNet"  # Possible choices: particleNet, deepJet, robustParticleTransformer
    bjet_wp = "T"  # Possible choices: L, M, T, XT, XXT

    clean_jet_dipho = False
    clean_jet_pho = True
    clean_jet_ele = True
    clean_jet_muo = True

    # photon preselection cuts
    min_pt_photon = 15.0
    min_pt_lead_photon = 15.0
    min_mvaid = -0.9
    min_mvaid_EB = 0.0439603
    min_mvaid_EE = -0.249526
    max_sc_eta = 2.5
    gap_barrel_eta = 1.4442
    gap_endcap_eta = 1.566
    max_hovere = 0.08
    min_full5x5_r9 = 0.8
    max_chad_iso = 20.0
    max_chad_rel_iso = 0.3

    min_full5x5_r9_EB_high_r9 = 0.85
    min_full5x5_r9_EE_high_r9 = 0.9
    min_full5x5_r9_EB_low_r9 = 0.5
    min_full5x5_r9_EE_low_r9 = 0.8
    max_trkSumPtHollowConeDR03_EB_low_r9 = (6.0)  # for v11, we cut on Photon_pfChargedIsoPFPV
    max_trkSumPtHollowConeDR03_EE_low_r9 = 6.0  # Leaving the names of the preselection cut variables the same to change as little as possible
    max_sieie_EB_low_r9 = 0.015
    max_sieie_EE_low_r9 = 0.035
    max_pho_iso_EB_low_r9 = 4.0
    max_pho_iso_EE_low_r9 = 4.0

    eta_rho_corr = 1.5
    low_eta_rho_corr = 0.16544
    high_eta_rho_corr = 0.13212
    # EA values for Run3 from Egamma
    EA1_EB1 = 0.102056
    EA2_EB1 = -0.000398112
    EA1_EB2 = 0.0820317
    EA2_EB2 = -0.000286224
    EA1_EE1 = 0.0564915
    EA2_EE1 = -0.000248591
    EA1_EE2 = 0.0428606
    EA2_EE2 = -0.000171541
    EA1_EE3 = 0.0395282
    EA2_EE3 = -0.000121398
    EA1_EE4 = 0.0369761
    EA2_EE4 = -8.10369e-05
    EA1_EE5 = 0.0369417
    EA2_EE5 = -2.76885e-05

    def __init__(
        self,
        metaconditions: Dict[str, Any],
        systematics: Optional[Dict[str, List[str]]],
        corrections: Optional[Dict[str, List[str]]],
        apply_trigger: bool,
        output_location: Optional[str],
        taggers: Optional[List[Any]],
        nano_version: int,
        bTagEffFileName: Optional[str],
        trigger_group: str,
        analysis: str,
        applyCQR: bool,
        skipJetVetoMap: bool,
        year: Optional[Dict[str, List[str]]],
        fiducialCuts: str,
        doDeco: bool,
        Smear_sigma_m: bool,
        doFlow_corrections: bool,
        validate_with_electrons: bool,
        output_format: str,
    ) -> None:

        self.meta = metaconditions
        self.systematics = systematics if systematics is not None else {}
        self.corrections = corrections if corrections is not None else {}
        self.apply_trigger = apply_trigger
        self.output_location = output_location
        self.nano_version = nano_version
        self.bTagEffFileName = bTagEffFileName
        self.trigger_group = trigger_group
        self.analysis = analysis
        self.applyCQR = applyCQR
        self.skipJetVetoMap = skipJetVetoMap
        self.year = year if year is not None else {}
        self.fiducialCuts = fiducialCuts
        self.doDeco = doDeco
        self.Smear_sigma_m = Smear_sigma_m
        self.doFlow_corrections = doFlow_corrections
        self.validate_with_electrons = validate_with_electrons
        self.output_format = output_format
        self.name_convention = "Legacy"
        # no diphoton and photon id mva unless specified
        self.photonid_mva_EB = None
        self.photonid_mva_EE = None
        self.diphoton_mva = None

        logger.debug(f"Setting up processor with metaconditions: {self.meta}")

        if (self.bjet_mva != "deepJet") and (self.nano_version is not None) and (self.nano_version < 12):
            logger.warning(f"\n {self.bjet_mva} is supported for nanoAOD v12 and above. Please check to have a valid implementation of the HF tagger score...\n")

        self.taggers = []
        if taggers is not None:
            self.taggers = taggers
            self.taggers.sort(key=lambda x: x.priority)

        # self.prefixes = {"pho_lead": "lead", "pho_sublead": "sublead"}
        self.prefixes = {"pho_lead": "pholead", "pho_sublead": "phosublead", "bjet_lead": "bJetlead", "bjet_sublead": "bJetsublead", "bquark_lead": "genblead", "bquark_sublead": "genbsublead", "bjet_best": "bJetbest", "bjet": "bJet", "lepton":"lep"}

        if not self.doDeco:
            logger.info("Skipping Mass resolution decorrelation as required")
        else:
            logger.info("Performing Mass resolution decorrelation as required")

        # build the chained quantile regressions
        if self.applyCQR:
            try:
                self.chained_quantile: Optional[
                    ChainedQuantileRegression
                ] = ChainedQuantileRegression(**self.meta["PhoIdInputCorrections"])
            except Exception as e:
                warnings.warn(f"Could not instantiate ChainedQuantileRegression: {e}")
                self.chained_quantile = None
        else:
            logger.info("Skipping CQR as required")
            self.chained_quantile = None

        if self.validate_with_electrons:
            logger.info("Running the analysis with electrons reconstructed as photons. Using dielectron triggers.")
            self.trigger_group = ".*DoubleEG.*"
            self.analysis = "Dielectron"
            self.min_pt_photon = 12.0
            self.min_pt_lead_photon = 23.0
        else:
            # check that "ElectronRecoSF_Zee_val" is not part of corrections
            uses_Zee_val_SF = [ds for ds, corr in self.corrections.items() if "ElectronRecoSF_Zee_val" in corr]
            if uses_Zee_val_SF:
                msg = (
                    "Correction 'ElectronRecoSF_Zee_val' requires --validate-with-electrons. "
                    f"Remove it from: {', '.join(uses_Zee_val_SF)} or enable validation.")
                logger.error(msg)
                raise ValueError(msg)

    def apply_filters_and_triggers(self, events: ak.Array) -> ak.Array:
        # met filters
        met_filters = self.meta["flashggMetFilters"][self.data_kind]
        filtered = functools.reduce(
            operator.and_,
            (events.Flag[metfilter.split("_")[-1]] for metfilter in met_filters),
        )

        triggered = ak.ones_like(filtered)

        # Check: Do we apply trigger SF to MC?
        # If yes: We should not apply the trigger bits to MC
        # Also take into account case when no corrections are passed by using get instead of simple [] access
        if "TriggerSF" in self.corrections.get(events.metadata["dataset"], {}) and self.data_kind == "mc":
            self.apply_trigger = False
        elif "TriggerSF" not in self.corrections.get(events.metadata["dataset"], {}) and self.data_kind == "mc":
            logger.warning(
                "You are running over MC and not applying trigger SF. "
                "Because of this, the trigger bits will be applied to the MC. "
                "Please make sure this is what you want. Such a configuration "
                "should not be used for a final measurement with a Hgg signal MC sample."
            )

        if self.apply_trigger:
            trigger_names = []
            triggers = self.meta["TriggerPaths"][self.trigger_group][self.analysis]
            hlt = events.HLT
            for trigger in triggers:
                actual_trigger = trigger.replace("HLT_", "").replace("*", "")
                for field in hlt.fields:
                    if field.startswith(actual_trigger):
                        trigger_names.append(field)
            triggered = functools.reduce(
                operator.or_, (hlt[trigger_name] for trigger_name in trigger_names)
            )

        return events[filtered & triggered]

    def initialize_photonid_mva(self) -> None:
        # initialize photonid_mva
        photon_id_mva_dir = os.path.dirname(photon_id_mva_weights.__file__)
        try:
            logger.debug(
                f"Looking for {self.meta['flashggPhotons']['photonIdMVAweightfile_EB']} in {photon_id_mva_dir}"
            )
            self.photonid_mva_EB = load_photonid_mva(
                os.path.join(
                    photon_id_mva_dir,
                    self.meta["flashggPhotons"]["photonIdMVAweightfile_EB"],
                )
            )
            self.photonid_mva_EE = load_photonid_mva(
                os.path.join(
                    photon_id_mva_dir,
                    self.meta["flashggPhotons"]["photonIdMVAweightfile_EE"],
                )
            )
        except Exception as e:
            warnings.warn(f"Could not instantiate PhotonID MVA on the fly: {e}")
            self.photonid_mva_EB = None
            self.photonid_mva_EE = None

    def add_photonid_mva(
        self, photons: ak.Array, events: ak.Array
    ) -> ak.Array:
        photons["fixedGridRhoAll"] = events.Rho.fixedGridRhoAll * ak.ones_like(
            photons.pt
        )
        counts = ak.num(photons, axis=-1)
        photons = ak.flatten(photons)
        isEB = ak.to_numpy(numpy.abs(photons.eta) < 1.5)
        mva_EB = calculate_photonid_mva(
            (self.photonid_mva_EB, self.meta["flashggPhotons"]["inputs_EB"]), photons
        )
        mva_EE = calculate_photonid_mva(
            (self.photonid_mva_EE, self.meta["flashggPhotons"]["inputs_EE"]), photons
        )
        mva = ak.where(isEB, mva_EB, mva_EE)
        photons["mvaID"] = mva

        return ak.unflatten(photons, counts)

    def add_photonid_mva_run3(
        self, photons: ak.Array, events: ak.Array
    ) -> ak.Array:
        preliminary_path = os.path.join(os.path.dirname(__file__), '../tools/flows/run3_mvaID_models/')
        photonid_mva_EB, photonid_mva_EE = load_photonid_mva_run3(preliminary_path)

        rho = events.Rho.fixedGridRhoAll * ak.ones_like(photons.pt)
        rho = ak.flatten(rho)

        photons = ak.flatten(photons)

        isEB = ak.to_numpy(numpy.abs(photons.eta) < 1.5)
        mva_EB = calculate_photonid_mva_run3(
            [photonid_mva_EB, self.meta["flashggPhotons"]["inputs_EB"]], photons , rho
        )
        mva_EE = calculate_photonid_mva_run3(
            [photonid_mva_EE, self.meta["flashggPhotons"]["inputs_EE"]], photons, rho
        )
        mva = ak.where(isEB, mva_EB, mva_EE)
        photons["mvaID_run3"] = mva

        return mva

    def add_corr_photonid_mva_run3(
        self, photons: ak.Array, events: ak.Array
    ) -> ak.Array:
        preliminary_path = os.path.join(os.path.dirname(__file__), '../tools/flows/run3_mvaID_models/')
        photonid_mva_EB, photonid_mva_EE = load_photonid_mva_run3(preliminary_path)

        rho = events.Rho.fixedGridRhoAll * ak.ones_like(photons.pt)
        rho = ak.flatten(rho)

        photons = ak.flatten(photons)

        # Now calculating the corrected mvaID
        isEB = ak.to_numpy(numpy.abs(photons.eta) < 1.5)
        corr_mva_EB = calculate_photonid_mva_run3(
            [photonid_mva_EB, self.meta["flashggPhotons"]["inputs_EB_corr"]], photons, rho
        )
        corr_mva_EE = calculate_photonid_mva_run3(
            [photonid_mva_EE, self.meta["flashggPhotons"]["inputs_EE_corr"]], photons, rho
        )
        corr_mva = ak.where(isEB, corr_mva_EB, corr_mva_EE)

        return corr_mva

    def initialize_diphoton_mva(self) -> None:
        # initialize diphoton mva
        diphoton_weights_dir = os.path.dirname(diphoton_mva_dir.__file__)
        logger.debug(
            f"Base path to look for IDMVA weight files: {diphoton_weights_dir}"
        )

        try:
            self.diphoton_mva = load_bdt(
                os.path.join(
                    diphoton_weights_dir, self.meta["flashggDiPhotonMVA"]["weightFile"]
                )
            )
        except Exception as e:
            warnings.warn(f"Could not instantiate diphoton MVA: {e}")
            self.diphoton_mva = None

    def add_diphoton_mva(
        self, diphotons: ak.Array, events: ak.Array
    ) -> ak.Array:
        return calculate_diphoton_mva(
            self,
            (self.diphoton_mva, self.meta["HiggsDNA_DiPhotonMVA"]["inputs"]),
            diphotons,
            events,
        )

    def add_zero_photon_mass_and_charge(self, photons: ak.Array) -> ak.Array:
        """Add zero mass to the Photon object in the events."""
        # add zero photon mass and charge
        # TODO: remove this temporary fix when https://github.com/scikit-hep/vector/issues/498 is resolved
        photons["mass"] = ak.zeros_like(photons.pt)
        photons["charge"] = ak.zeros_like(photons.pt)
        return photons

    def resolve_nano_version(self, events: ak.Array) -> None:
        """Resolve self.nano_version from events when --nano-version=auto was passed."""
        if self.nano_version is not None:
            return
        self.nano_version = infer_nano_version(events)
        if self.nano_version is None:
            logger.error("Unable to infer NanoAOD version from events. Please provide it explicitly using the --nano-version argument.")
            raise ValueError("Unable to infer NanoAOD version from events. Please provide it explicitly using the --nano-version argument.")
        logger.info(f"Detected NanoAOD version: {self.nano_version}")

    @abstractmethod
    def process(self, events: ak.Array) -> Dict[Any, Any]:
        """Has to be implemented in the subclass."""
        pass

    def process_extra(self, events: ak.Array) -> ak.Array:
        """Optional to override."""
        return events, {}

    @abstractmethod
    # Required to be abstract method by coffea (design choice)
    def postprocess(self, accumulant: Dict[Any, Any]) -> Any:
        """Has to be implemented in the subclass."""
        pass
