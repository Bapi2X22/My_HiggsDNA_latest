from higgs_dna.workflows.base import HggBaseProcessor
from higgs_dna.tools.chained_quantile import ChainedQuantileRegression
from higgs_dna.tools.SC_eta import add_photon_SC_eta
from higgs_dna.tools.jetID import add_jetId
from higgs_dna.tools.EELeak_region import veto_EEleak_flag
from higgs_dna.tools.EcalBadCalibCrystal_events import remove_EcalBadCalibCrystal_events
from higgs_dna.tools.gen_helpers import get_fiducial_flag, get_genJets, get_higgs_gen_attributes
from higgs_dna.selections.diphoton_selections import build_diphoton_candidates, apply_fiducial_cut_det_level
from higgs_dna.selections.photon_selections import photon_preselection_bbgg
# from higgs_dna.selections.photon_preselections import photon_preselections
from higgs_dna.selections.lepton_selections import select_electrons_bbgg, select_muons_bbgg
from higgs_dna.selections.jet_selections import select_jets_bbgg, jetvetomap
from higgs_dna.selections.lumi_selections import select_lumis
from higgs_dna.utils.dumping_utils import (
    diphoton_ak_array,
    dump_ak_array,
    diphoton_list_to_pandas,
    dump_pandas,
    get_obj_syst_dict,
)
from higgs_dna.utils.misc_utils import choose_jet

from higgs_dna.systematics import object_systematics as available_object_systematics
from higgs_dna.systematics import object_corrections as available_object_corrections
from higgs_dna.systematics import weight_systematics as available_weight_systematics
from higgs_dna.systematics import weight_corrections as available_weight_corrections

import functools
import operator
import os
import warnings
from typing import Any, Dict, List, Optional
import awkward
import numpy
import xgboost as xgb
import pickle
import sys
import vector
from coffea import processor
from coffea.analysis_tools import Weights
from copy import deepcopy

import logging

logger = logging.getLogger(__name__)

vector.register_awkward()


class My_first_processor_WH_Cat1_HDNA_presel(HggBaseProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        model_file = os.path.join(
            os.path.dirname(__file__),
            "models",
            "trained_model.pkl",
        )

        with open(model_file, "rb") as f:
            modelDict = pickle.load(f)

        self.bdt_model = modelDict["xgbModel"]

        self.inputVars = [
            "leppt","lepeta",
            "first_jet_eta","second_jet_eta",
            "pT1_by_mbb","pT2_by_mbb",
            "n_bJets",
            "pholead_eta","phosublead_eta",
            "first_jet_B","second_jet_B",
            "pholead_mvaID","phosublead_mvaID",
            "pT1_by_mgg","pT2_by_mgg",
            "delphi_gg","delphi_bb","delphi_bbgg",
            "mass_point"
        ]
    # def __init__(
    #     self,
    #     metaconditions: Dict[str, Any],
    #     systematics: Dict[str, List[Any]] = None,
    #     corrections: Dict[str, List[Any]] = None,
    #     apply_trigger: bool = False,
    #     output_location: Optional[str] = None,
    #     taggers: Optional[List[Any]] = None,
    #     trigger_group=".*DoubleEG.*",
    #     analysis="mainAnalysis",
    #     skipCQR: bool = False,
    #     skipJetVetoMap: bool = False,
    #     year: Dict[str, List[str]] = None,
    #     fiducialCuts: str = "classical",
    #     doDeco: bool = False,
    #     Smear_sigma_m: bool = False,
    #     doFlow_corrections: bool = False,
    #     output_format: str = "parquet",
    #     nano_version: int = None,
    # ) -> None:
    #     super().__init__(
    #         metaconditions,
    #         systematics=systematics,
    #         corrections=corrections,
    #         apply_trigger=apply_trigger,
    #         output_location=output_location,
    #         taggers=taggers,
    #         trigger_group=trigger_group,
    #         analysis=analysis,
    #         skipCQR=skipCQR,
    #         skipJetVetoMap=skipJetVetoMap,
    #         year=year,
    #         fiducialCuts=fiducialCuts,
    #         doDeco=doDeco,
    #         Smear_sigma_m=Smear_sigma_m,
    #         doFlow_corrections=doFlow_corrections,
    #         output_format=output_format,
    #         nano_version = nano_version,
    #     )


    def choose_nth_object_variable(self, variable, n, fill_value):
        """
        this helper function is used to create flat collection from a jagged collection,
        parameters:
        * variable: (awkward array) selected variable from the object
        * n: (int) nth object to be selected
        * fill_value: (float) value with which to fill the padded none.
        """
        variable = variable[
            awkward.local_index(variable) == n
        ]
        variable = awkward.pad_none(
            variable, 1
        )
        variable = awkward.flatten(
            awkward.fill_none(variable, fill_value)
        )
        return variable

    def process(self, events: awkward.Array) -> Dict[Any, Any]:
        dataset_name = events.metadata["dataset"]

        # data or monte carlo?
        self.data_kind = "mc" if hasattr(events, "GenPart") else "data"

        n_events = len(events)
        print(f"Dataset {dataset_name}: {n_events} events")

        # here we start recording possible coffea accumulators
        # most likely histograms, could be counters, arrays, ...
        histos_etc = {}
        histos_etc[dataset_name] = {}
        if self.data_kind == "mc":
            histos_etc[dataset_name]["nTot"] = int(
                awkward.num(events.genWeight, axis=0)
            )
            histos_etc[dataset_name]["nPos"] = int(awkward.sum(events.genWeight > 0))
            histos_etc[dataset_name]["nNeg"] = int(awkward.sum(events.genWeight < 0))
            histos_etc[dataset_name]["nEff"] = int(
                histos_etc[dataset_name]["nPos"] - histos_etc[dataset_name]["nNeg"]
            )
            histos_etc[dataset_name]["genWeightSum"] = float(
                awkward.sum(events.genWeight)
            )
        else:
            histos_etc[dataset_name]["nTot"] = int(len(events))
            histos_etc[dataset_name]["nPos"] = int(histos_etc[dataset_name]["nTot"])
            histos_etc[dataset_name]["nNeg"] = int(0)
            histos_etc[dataset_name]["nEff"] = int(histos_etc[dataset_name]["nTot"])
            histos_etc[dataset_name]["genWeightSum"] = float(len(events))

        # lumi mask
        if self.data_kind == "data":
            try:
                lumimask = select_lumis(self.year[dataset_name][0], events, logger)
                events = events[lumimask]
            except:
                logger.info(
                    f"[ lumimask ] Skip now! Unable to find year info of {dataset_name}"
                )
        # apply jetvetomap: only retain events that without any jets in the EE leakage region
        # if not self.skipJetVetoMap:
        #     events = jetvetomap(
        #         self, events, logger, dataset_name, year=self.year[dataset_name][0]
        #     )
        # metadata array to append to higgsdna output
        metadata = {}

        if self.data_kind == "mc":
            # Add sum of gen weights before selection for normalisation in postprocessing
            metadata["sum_genw_presel"] = str(awkward.sum(events.genWeight))
        else:
            metadata["sum_genw_presel"] = "Data"

        # apply filters and triggers
        events = self.apply_filters_and_triggers(events)
        n_events_2 = len(events)
        print(f"after trigger: Dataset {dataset_name}: {n_events_2} events")

        # remove events affected by EcalBadCalibCrystal
        if self.data_kind == "data":
            events = remove_EcalBadCalibCrystal_events(events)

        # we need ScEta for corrections and systematics, it is present in NanoAODv13+ and can be calculated using PV for older versions
        # events.Photon = add_photon_SC_eta(events.Photon, events.PV)
        events["Photon"] = add_photon_SC_eta(events.Photon, events.PV)
        Jets = events.Jet
        Jet_id = add_jetId(Jets, self.nano_version, self.year[dataset_name][0], flattenUnflatten=True)
        events["Jet"] = awkward.with_field(Jets, Jet_id, "jetId")


        # add veto EE leak branch for photons, could also be used for electrons
        # if (
        #     self.year[dataset_name][0] == "2022EE"
        #     or self.year[dataset_name][0] == "2022postEE"
        # ):
        #     events.Photon = veto_EEleak_flag(self, events.Photon)

        # read which systematics and corrections to process
        try:
            correction_names = self.corrections[dataset_name]
        except KeyError:
            correction_names = []
        try:
            systematic_names = self.systematics[dataset_name]
        except KeyError:
            systematic_names = []

#----------------------------------------------Adding the pt_raw----------------------------------------------------------------------------

        # save raw pt if we use scale/smearing corrections
        # These needs to be before the smearing of the mass resolution in order to have the raw pt for the function
        s_or_s_applied = False
        s_or_s_ele_applied = False
        jec_applied = False

        for correction in correction_names:
            if "scale" or "smearing" in correction.lower():
                if "Electron" in correction:
                    s_or_s_ele_applied = True
                if "jec_jet_syst" in correction.lower():
                    jec_applied = True
                else:
                    s_or_s_applied = True
        print("s_or_s_applied: ", s_or_s_applied)
        print("s_or_s_ele_applied: ", s_or_s_ele_applied)
        if s_or_s_applied:
            print("Adding pt_raw to photons")
            events.Photon["pt_raw"] = events.Photon.pt
        if s_or_s_ele_applied:
            events["Electron"] = awkward.with_field(events.Electron, events.Electron.pt, "pt_raw")
        if jec_applied:
            events["Jet"] = awkward.with_field(events.Jet, events.Jet.pt, "jet_pt_raw")
            events["Jet"] = awkward.with_field(events.Jet, events.Jet.mass, "jet_mass_raw")

        print("Photon_fields", events.Photon.fields)
        print("Electron_fields", events.Electron.fields)


        for correction_name in correction_names:
            if correction_name in available_object_corrections.keys():
                logger.info(
                    f"Applying correction {correction_name} to dataset {dataset_name}"
                )
                varying_function = available_object_corrections[correction_name]
                events = varying_function(
                    events=events, year=self.year[dataset_name][0]
                )
            elif correction_name in available_weight_corrections:
                # event weight corrections will be applied after photon preselection / application of further taggers
                continue
            else:
                # may want to throw an error instead, needs to be discussed
                warnings.warn(f"Could not process correction {correction_name}.")
                continue

        #-----------------------------------------------------Store the gen information-------------------------------------------------------------
        if self.data_kind == "mc":

            gen = events.GenPart
            Gen_photons = gen[(gen.pdgId == 22) & (gen.status == 1)]
            mother_idx = Gen_photons.genPartIdxMother
            from_a_mask = gen[mother_idx].pdgId == 35
            photons_from_a = Gen_photons[from_a_mask]

            sorted_gen_photons = photons_from_a[awkward.argsort(photons_from_a.pt, axis=1, ascending=False)]

            leading_gen_photon_pt = self.choose_nth_object_variable(sorted_gen_photons.pt, 0, -999.0)
            subleading_gen_photon_pt = self.choose_nth_object_variable(sorted_gen_photons.pt, 1, -999.0)
            leading_gen_photon_eta = self.choose_nth_object_variable(sorted_gen_photons.eta, 0, -999.0)
            subleading_gen_photon_eta = self.choose_nth_object_variable(sorted_gen_photons.eta, 1, -999.0)
            leading_gen_photon_phi = self.choose_nth_object_variable(sorted_gen_photons.phi, 0, -999.0)
            subleading_gen_photon_phi = self.choose_nth_object_variable(sorted_gen_photons.phi, 1, -999.0)

        # event_idx = awkward.local_index(events)
        event_idx = awkward.local_index(events, axis=0)
        original_photons = events.Photon
        nPhotons = len(events.Photon)
        print("Number of photons: ", nPhotons)
        # NOTE: jet jerc systematics are added in the correction functions and handled later
        original_jets = events.Jet

        # systematic object variations
        for systematic_name in systematic_names:
            if systematic_name in available_object_systematics.keys():
                systematic_dct = available_object_systematics[systematic_name]
                if systematic_dct["object"] == "Photon":
                    logger.info(
                        f"Adding systematic {systematic_name} to photons collection of dataset {dataset_name}"
                    )
                    original_photons.add_systematic(
                        # passing the arguments here explicitly since I want to pass the events to the varying function. If there is a more elegant / flexible way, just change it!
                        name=systematic_name,
                        kind=systematic_dct["args"]["kind"],
                        what=systematic_dct["args"]["what"],
                        varying_function=functools.partial(
                            systematic_dct["args"]["varying_function"],
                            events=events,
                            year=self.year[dataset_name][0],
                        )
                        # name=systematic_name, **systematic_dct["args"]
                    )
                # to be implemented for other objects here
            elif systematic_name in available_weight_systematics:
                # event weight systematics will be applied after photon preselection / application of further taggers
                continue
            else:
                # may want to throw an error instead, needs to be discussed
                warnings.warn(
                    f"Could not process systematic variation {systematic_name}."
                )
                continue

        # Applying systematic variations
        photons_dct = {}
        photons_dct["nominal"] = original_photons
        logger.debug(original_photons.systematics.fields)
        for systematic in original_photons.systematics.fields:
            for variation in original_photons.systematics[systematic].fields:
                # deepcopy to allow for independent calculations on photon variables with CQR
                photons_dct[f"{systematic}_{variation}"] = deepcopy(
                    original_photons.systematics[systematic][variation]
                )

        # NOTE: jet jerc systematics are added in the corrections, now extract those variations and create the dictionary
        jerc_syst_list, jets_dct = get_obj_syst_dict(original_jets, ["pt", "mass"])
        # object systematics dictionary
        logger.debug(f"[ jerc systematics ] {jerc_syst_list}")

        # Build the flattened array of all possible variations
        variations_combined = []
        variations_combined.append(original_photons.systematics.fields)
        # NOTE: jet jerc systematics are not added with add_systematics
        variations_combined.append(jerc_syst_list)
        # Flatten
        variations_flattened = sum(variations_combined, [])  # Begin with empty list and keep concatenating
        # Attach _down and _up
        variations = [item + suffix for item in variations_flattened for suffix in ['_down', '_up']]
        # Add nominal to the list
        variations.append('nominal')
        logger.debug(f"[systematics variations] {variations}")

        original_events = events
        for variation in variations:
            photons, jets = photons_dct["nominal"], original_events.Jet
            # photons, jets = photons_dct["nominal"], events.Jet
            if variation == "nominal":
                pass  # Do nothing since we already get the unvaried, but nominally corrected objets above
            elif variation in [*photons_dct]:  # [*dict] gets the keys of the dict since Python >= 3.5
                photons = photons_dct[variation]
            elif variation in [*jets_dct]:
                jets = jets_dct[variation]
            do_variation = variation  # We can also simplify this a bit but for now it works

            # if self.chained_quantile is not None:
            #     photons = self.chained_quantile.apply(photons, events)
            # # recompute photonid_mva on the fly
            # if self.photonid_mva_EB and self.photonid_mva_EE:
            #     photons = self.add_photonid_mva(photons, events)

            electrons = original_events.Electron
            muons = original_events.Muon

            # photon preselection
            # photons, b_jets, leps = photon_preselections(self, photons, jets, electrons, muons, events, year=self.year[dataset_name][0])
            # leps = awkward.concatenate([events.Electron, events.Muon], axis=1)
            # photons, b_jets, leps = photons, jets, leps

            # sort photons in each event descending in pt
            # make descending-pt combinations of photons
            # photons = photons[awkward.argsort(photons.pt, ascending=False)]
            photons["charge"] = awkward.zeros_like(
                photons.pt
            )  # added this because charge is not a property of photons in nanoAOD v11. We just assume every photon has charge zero...
            # diphotons = awkward.combinations(
            #     photons, 2, fields=["pho_lead", "pho_sublead"]
            # )
            # diphotons = build_diphoton_candidates(photons, self.min_pt_lead_photon)

            # print("Length of diphoton array: ", len(diphotons))

            # electrons = electrons[select_electrons(self, electrons, diphotons=diphotons)]
            electrons = electrons[select_electrons_bbgg(self, electrons)]
            # muons = muons[select_muons(self, muons, diphotons=diphotons)]
            muons = muons[select_muons_bbgg(self, muons)]
            photons = photon_preselection_bbgg(self, photons, original_events, electrons, muons, year=self.year[dataset_name][0], electron_veto=False)
            diphotons = build_diphoton_candidates(photons, self.min_pt_lead_photon)
            # jets = jets[select_jets(self, jets, diphotons, muons, electrons, taus=None)]
            jets = jets[select_jets_bbgg(self, jets, diphotons, muons, electrons, taus=None)]


            #photons = photon_preselection(self, photons, original_events, year=self.year[dataset_name][0], electron_veto=False)
            # photons = photon_preselection(self, photons, original_events, year=self.year[dataset_name][0])
            # photons = photon_preselection(self, photons, events, year=self.year[dataset_name][0])


#--------------------------------Now do your event selection according to bbgg analysis----------------------------------------------------
            one_ele = awkward.num(electrons) == 1
            one_mu = awkward.num(muons) == 1

            if self.data_kind == "mc":
                # MC: keep both channels
                lepton_channel_mask = one_ele ^ one_mu

            else:
                # Data
                if "EGamma" in dataset_name:          # or however you identify the EGamma dataset
                    lepton_channel_mask = one_ele & (awkward.num(muons) == 0)

                elif "Muon" in dataset_name:       # or "SingleMuon", "MuonEG", etc.
                    lepton_channel_mask = one_mu & (awkward.num(electrons) == 0)

                else:
                    raise ValueError(f"Unknown data stream: {dataset_name}")

            b_jets = jets[jets.btagUParTAK4B > 0.1272]
            at_least_one_bjets = awkward.num(b_jets) >= 1
            at_least_two_photons = awkward.num(photons) >= 2

            b_jets = jets[jets.btagUParTAK4B > 0.1272]
            at_least_one_bjets = awkward.num(b_jets) >= 1
            at_least_two_photons = awkward.num(photons) >= 2

            event_mask = lepton_channel_mask & at_least_one_bjets & at_least_two_photons

            # breakpoint()

            photons = photons[event_mask]
            electrons = electrons[event_mask]
            muons = muons[event_mask]
            b_jets = b_jets[event_mask]
            n_bjets = awkward.num(b_jets)
            events = original_events[event_mask]
            # events = events[event_mask]

            photons = photons[awkward.argsort(photons.pt, ascending=False)]

            diphotons = build_diphoton_candidates(photons, self.min_pt_lead_photon)

            selected_event_idx = event_idx[event_mask]

            

            # Store b-Jets information             
            # b_jets = b_jets[awkward.argsort(b_jets.pt, ascending=False)]
            # b_jets = b_jets[awkward.argsort(b_jets.btagUParTAK4B, ascending=False)][:, :2]
            # b_jets = b_jets[awkward.argsort(b_jets.btagUParTAK4B, ascending=False)]
            b_jets = b_jets[awkward.argsort(b_jets.pt, ascending=False)]

            b_jets["charge"] = awkward.zeros_like(
                b_jets.pt
            )  # added this because charge is not a property of photons in nanoAOD v11. We just assume every photon has charge zero...
            diJets = awkward.combinations(
                b_jets, 2, fields=["bjet_lead", "bjet_sublead"]
            )

            NDiphotons = len(diphotons)
            print(f"Dataset {dataset_name}: {NDiphotons} Diphotons")

            leps = awkward.concatenate([electrons, muons], axis=1)


            # the remaining cut is to select the leading photons
            # the previous sort assures the order
            # diphotons = diphotons[
            #     diphotons["pho_lead"].pt > self.min_pt_lead_photon
            # ]

            # now turn the diphotons into candidates with four momenta and such
            diphoton_4mom = diphotons["pho_lead"] + diphotons["pho_sublead"]
            dipho_pt_evt = awkward.firsts(diphoton_4mom.pt)
            dipho_pt_evt = awkward.fill_none(dipho_pt_evt, 0.0)

            diphotons["pt"] = diphoton_4mom.pt
#-------------------------------------------------For false categorization only----------------------------------------------------
            diphotons["dipho_pt"] = diphoton_4mom.pt
            diJets = awkward.with_field(
                diJets,
                dipho_pt_evt,
                "dipho_pt",
            )

            leps = awkward.with_field(
                leps,
                dipho_pt_evt,
                "dipho_pt",
            )

#-----------------------------------------------------------------------------------------------------------------------------------
            diphotons["eta"] = diphoton_4mom.eta
            diphotons["phi"] = diphoton_4mom.phi
            diphotons["mass"] = diphoton_4mom.mass
            diphotons["charge"] = diphoton_4mom.charge

            diphoton_pz = diphoton_4mom.z
            diphoton_e = diphoton_4mom.energy

            diphotons["rapidity"] = 0.5 * numpy.log((diphoton_e + diphoton_pz) / (diphoton_e - diphoton_pz))

            diphotons = awkward.with_name(diphotons, "PtEtaPhiMCandidate")

            # sort diphotons by pT
            diphotons = diphotons[
                awkward.argsort(diphotons.pt, ascending=False)
            ]

            NDiphotons_2 = len(diphotons)
            print(f"Pos2: Dataset {dataset_name}: {NDiphotons_2} Diphotons")


            # diphotons = diphotons[fid_det_passed]
            NDiphotons_3 = len(diphotons)
            print(f"Pos3: Dataset {dataset_name}: {NDiphotons_3} Diphotons")

            # baseline modifications to diphotons
            # if self.diphoton_mva is not None:
            #     diphotons = self.add_diphoton_mva(diphotons, events)
            NDiphotons_4 = len(diphotons)
            print(f"Pos4: Dataset {dataset_name}: {NDiphotons_4} Diphotons")
            # workflow specific processing
            events, process_extra = self.process_extra(events)
            histos_etc.update(process_extra)

            if self.data_kind == "mc" and "WH-2024M" in dataset_name:
                A = events.GenPart[events.GenPart.pdgId == 35]
                mass_point = awkward.to_numpy(
                    awkward.firsts(A.mass)
                )
            else:
                mass_grid = numpy.array(
                    [12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
                )

                mass_point = mass_grid[
                    awkward.to_numpy(events.event) % len(mass_grid)
                ]


            first_jet_pt = choose_jet(b_jets.pt, 0, -999.0)
            first_jet_pt_raw = choose_jet(b_jets.jet_pt_raw, 0, -999.0) if jec_applied else None
            first_jet_eta = choose_jet(b_jets.eta, 0, -999.0)
            first_jet_phi = choose_jet(b_jets.phi, 0, -999.0)
            first_jet_mass = choose_jet(b_jets.mass, 0, -999.0)
            first_jet_mass_raw = choose_jet(b_jets.jet_mass_raw, 0, -999.0) if jec_applied else None
            # first_jet_charge = choose_jet(jets.charge, 0, -999.0)
            first_jet_B = choose_jet(b_jets.btagUParTAK4B, 0, -999.0)
            first_jet_probb = choose_jet(b_jets.btagUParTAK4probb, 0, -999.0)
            first_jet_probbb = choose_jet(b_jets.btagUParTAK4probbb, 0, -999.0)
            diff_first_jet_probb_probbb = first_jet_probb - first_jet_probbb
            first_jet_charge = -999.0

            second_jet_pt = choose_jet(b_jets.pt, 1, -999.0)
            second_jet_pt_raw = choose_jet(b_jets.jet_pt_raw, 1, -999.0) if jec_applied else None
            second_jet_eta = choose_jet(b_jets.eta, 1, -999.0)
            second_jet_phi = choose_jet(b_jets.phi, 1, -999.0)
            second_jet_mass = choose_jet(b_jets.mass, 1, -999.0)
            second_jet_mass_raw = choose_jet(b_jets.jet_mass_raw, 1, -999.0) if jec_applied else None
            # second_jet_charge = choose_jet(jets.charge, 1, -999.0)
            second_jet_B = choose_jet(b_jets.btagUParTAK4B, 1, -999.0)
            second_jet_probb = choose_jet(b_jets.btagUParTAK4probb, 1, -999.0)
            second_jet_probbb = choose_jet(b_jets.btagUParTAK4probbb, 1, -999.0)

            second_jet_charge = -999.0

            electron_pt = choose_jet(electrons.pt, 0, -999.0)
            electron_eta = choose_jet(electrons.eta, 0, -999.0)
            electron_phi = choose_jet(electrons.phi, 0, -999.0)
            electron_mass = choose_jet(electrons.mass, 0, -999.0)
            electron_charge = choose_jet(electrons.charge, 0, -999.0)
            muon_pt = choose_jet(muons.pt, 0, -999.0)
            muon_eta = choose_jet(muons.eta, 0, -999.0)
            muon_phi = choose_jet(muons.phi, 0, -999.0)
            muon_mass = choose_jet(muons.mass, 0, -999.0)
            muon_charge = choose_jet(muons.charge, 0, -999.0)

            print("Length of diphoton array: ", len(diphotons))
            print("diphoton array: ", diphotons)
            print("Length of n_bjets array: ", len(n_bjets))
            print("n_bjets array: ", n_bjets)


            # ---------------- Lepton ----------------
            electron_pt  = choose_jet(electrons.pt, 0, -999.0)
            electron_eta = choose_jet(electrons.eta, 0, -999.0)

            muon_pt  = choose_jet(muons.pt, 0, -999.0)
            muon_eta = choose_jet(muons.eta, 0, -999.0)

            leppt = awkward.where(
                awkward.num(electrons) == 1,
                electron_pt,
                muon_pt
            )

            lepeta = awkward.where(
                awkward.num(electrons) == 1,
                electron_eta,
                muon_eta
            )

            # ---------------- Photon quantities ----------------
            mgg = diphotons.mass[:, 0]

            pT1_by_mgg = diphotons.pho_lead.pt[:, 0] / mgg
            pT2_by_mgg = diphotons.pho_sublead.pt[:, 0] / mgg

            delphi_gg = numpy.abs(
                diphotons.pho_lead[:, 0].delta_phi(
                    diphotons.pho_sublead[:, 0]
                )
            )

            # ---------------- Jets ----------------
            Njets = awkward.num(jets[event_mask])

            has2b = awkward.num(b_jets) >= 2

            jet1 = awkward.zip(
                {
                    "pt": first_jet_pt,
                    "eta": first_jet_eta,
                    "phi": first_jet_phi,
                    "mass": first_jet_mass,
                },
                with_name="Momentum4D",
            )

            jet2 = awkward.zip(
                {
                    "pt": second_jet_pt,
                    "eta": second_jet_eta,
                    "phi": second_jet_phi,
                    "mass": second_jet_mass,
                },
                with_name="Momentum4D",
            )

            bb = jet1 + jet2


            mbb = awkward.where(
                has2b,
                bb.mass,
                -999.0,
            )

            pT1_by_mbb = awkward.where(
                has2b,
                first_jet_pt / mbb,
                -999.0,
            )

            pT2_by_mbb = awkward.where(
                has2b,
                second_jet_pt / mbb,
                -999.0,
            )

            delphi_bb = awkward.where(
                has2b,
                numpy.abs(
                    jet1.deltaphi(jet2)
                ),
                -999.0,
            )

            # ---------------- bb vs gg ----------------

            # gg = diphotons.pho_lead + diphotons.pho_sublead
            gg = awkward.firsts(
                diphotons.pho_lead + diphotons.pho_sublead
            )

            delphi_bbgg = awkward.where(
                has2b,
                numpy.abs(
                    bb.deltaphi(gg)
                ),
                -999.0,
            )


            diphotons["n_bJets"] = n_bjets

            diphotons["first_jet_pt"] = first_jet_pt
            diphotons["first_jet_pt_raw"] = first_jet_pt_raw if jec_applied else None
            diphotons["first_jet_eta"] = first_jet_eta
            diphotons["first_jet_phi"] = first_jet_phi
            diphotons["first_jet_mass"] = first_jet_mass
            diphotons["first_jet_mass_raw"] = first_jet_mass_raw if jec_applied else None
            diphotons["first_jet_charge"] = first_jet_charge
            diphotons["first_jet_B"] = first_jet_B
            diphotons["first_jet_probb"] = first_jet_probb
            diphotons["first_jet_probbb"] = first_jet_probbb
            diphotons["diff_first_jet_probb_probbb"] = diff_first_jet_probb_probbb

            diphotons["second_jet_pt"] = second_jet_pt
            diphotons["second_jet_pt_raw"] = second_jet_pt_raw if jec_applied else None
            diphotons["second_jet_eta"] = second_jet_eta
            diphotons["second_jet_phi"] = second_jet_phi
            diphotons["second_jet_mass"] = second_jet_mass
            diphotons["second_jet_mass_raw"] = second_jet_mass_raw if jec_applied else None
            diphotons["second_jet_charge"] = second_jet_charge
            diphotons["second_jet_B"] = second_jet_B
            diphotons["second_jet_probb"] = second_jet_probb
            diphotons["second_jet_probbb"] = second_jet_probbb

            diphotons["electron_pt"] = electron_pt
            diphotons["electron_eta"] = electron_eta
            diphotons["electron_phi"] = electron_phi
            diphotons["electron_mass"] = electron_mass
            diphotons["electron_charge"] = electron_charge
            diphotons["muon_pt"] = muon_pt
            diphotons["muon_eta"] = muon_eta
            diphotons["muon_phi"] = muon_phi
            diphotons["muon_mass"] = muon_mass
            diphotons["muon_charge"] = muon_charge


            diphotons["leppt"]       = leppt
            diphotons["lepeta"]      = lepeta
            diphotons["pT1_by_mbb"]  = pT1_by_mbb
            diphotons["pT2_by_mbb"]  = pT2_by_mbb
            diphotons["Njets"]       = Njets
            diphotons["pT1_by_mgg"]  = pT1_by_mgg
            diphotons["pT2_by_mgg"]  = pT2_by_mgg
            diphotons["delphi_gg"]   = delphi_gg
            diphotons["delphi_bb"]   = delphi_bb
            diphotons["delphi_bbgg"] = delphi_bbgg

            # --------------------------------------------------
            # Build feature matrix in EXACT training order
            # --------------------------------------------------

            pholead_eta      = awkward.to_numpy(diphotons.pho_lead.eta[:,0])
            phosublead_eta   = awkward.to_numpy(diphotons.pho_sublead.eta[:,0])

            pholead_mvaID    = awkward.to_numpy(diphotons.pho_lead.mvaID[:,0])
            phosublead_mvaID = awkward.to_numpy(diphotons.pho_sublead.mvaID[:,0])

            for name, var in {
                "leppt": leppt,
                "lepeta": lepeta,
                "pT1_by_mbb": pT1_by_mbb,
                "pT2_by_mbb": pT2_by_mbb,
                "delphi_gg": delphi_gg,
                "delphi_bb": delphi_bb,
                "delphi_bbgg": delphi_bbgg,
            }.items():

                print(name)
                print(awkward.type(var))

            # breakpoint()

            X = numpy.column_stack([
                awkward.to_numpy(leppt),
                awkward.to_numpy(lepeta),

                awkward.to_numpy(first_jet_eta),
                awkward.to_numpy(second_jet_eta),

                awkward.to_numpy(pT1_by_mbb),
                awkward.to_numpy(pT2_by_mbb),

                awkward.to_numpy(n_bjets),

                pholead_eta,
                phosublead_eta,

                awkward.to_numpy(first_jet_B),
                awkward.to_numpy(second_jet_B),

                pholead_mvaID,
                phosublead_mvaID,

                awkward.to_numpy(pT1_by_mgg),
                awkward.to_numpy(pT2_by_mgg),

                awkward.to_numpy(delphi_gg),
                awkward.to_numpy(delphi_bb),
                awkward.to_numpy(delphi_bbgg),

                mass_point
            ])

            inputVars = [
                "leppt","lepeta",
                "first_jet_eta","second_jet_eta",
                "pT1_by_mbb","pT2_by_mbb",
                "n_bJets",
                "pholead_eta","phosublead_eta",
                "first_jet_B","second_jet_B",
                "pholead_mvaID","phosublead_mvaID",
                "pT1_by_mgg","pT2_by_mgg",
                "delphi_gg","delphi_bb","delphi_bbgg",
                "mass_point"
            ]

#------------------------------------------------------------------------------------------------------------
            # has2b = awkward.to_numpy(n_bjets >= 2)

            # bdt_score = numpy.full(len(n_bjets), -999.0)

            # X_2b = X[has2b]

            # if len(X_2b) > 0:
            #     dmatrix = xgb.DMatrix(
            #         X_2b,
            #         feature_names=self.inputVars
            #     )

            #     bdt_score[has2b] = self.bdt_model.predict(dmatrix)

            # # dmatrix = xgb.DMatrix(X, feature_names=self.inputVars)

            # # bdt_score = self.bdt_model.predict(dmatrix)

            # diphotons["BDT_score"] = bdt_score
            # diphotons["mass_point"] = mass_point

#-----------------------------------------------------------------------------------------------------------------------

            # has2b = awkward.to_numpy(n_bjets >= 2)

            # # Initialize the randomized/assigned mass-point score
            # bdt_score = numpy.full(len(n_bjets), -999.0)

            # for m in mass_grid:

            #     X_tmp = X.copy()
            #     X_tmp[:, -1] = m

            #     score_full = numpy.full(len(n_bjets), -999.0)

            #     X_2b = X_tmp[has2b]

            #     if len(X_2b) > 0:
            #         dmatrix = xgb.DMatrix(
            #             X_2b,
            #             feature_names=self.inputVars
            #         )

            #         score_full[has2b] = self.bdt_model.predict(dmatrix)

            #     # Save the score for this mass hypothesis
            #     diphotons[f"BDT_{m}"] = score_full

            #     # Fill BDT_score for events whose assigned mass hypothesis is m
            #     bdt_score[mass_point == m] = score_full[mass_point == m]

            # diphotons["BDT_score"] = bdt_score
            # diphotons["mass_point"] = mass_point

            # Signal: evaluate only the true mass point
            if self.data_kind == "mc" and "WH-2024M" in dataset_name:

                X_tmp = X.copy()
                X_tmp[:, -1] = mass_point

                bdt_score = numpy.full(len(n_bjets), -999.0)

                X_2b = X_tmp[has2b]

                if len(X_2b) > 0:
                    dmatrix = xgb.DMatrix(
                        X_2b,
                        feature_names=self.inputVars,
                    )
                    bdt_score[has2b] = self.bdt_model.predict(dmatrix)

            # Background (and data): evaluate all mass hypotheses
            else:

                bdt_score = numpy.full(len(n_bjets), -999.0)

                for m in mass_grid:

                    X_tmp = X.copy()
                    X_tmp[:, -1] = m

                    score_full = numpy.full(len(n_bjets), -999.0)

                    X_2b = X_tmp[has2b]

                    if len(X_2b) > 0:
                        dmatrix = xgb.DMatrix(
                            X_2b,
                            feature_names=self.inputVars,
                        )

                        score_full[has2b] = self.bdt_model.predict(dmatrix)

                    # Store all BDTs only for background
                    diphotons[f"BDT_{m}"] = score_full

                    # Fill the assigned score
                    bdt_score[mass_point == m] = score_full[mass_point == m]

            # Common output
            diphotons["BDT_score"] = bdt_score
            diphotons["mass_point"] = mass_point

            ### end of your code ###

            # run taggers on the events list with added diphotons
            # the shape here is ensured to be broadcastable
            for tagger in self.taggers:
                (
                    diphotons["_".join([tagger.name, str(tagger.priority)])],
                    tagger_extra,
                ) = tagger(
                    events, diphotons
                )  # creates new column in diphotons - tagger priority, or 0, also return list of histrograms here?
                histos_etc.update(tagger_extra)

            # if there are taggers to run, arbitrate by them first
            # Deal with order of tagger priorities
            # Turn from diphoton jagged array to whether or not an event was selected
            if len(self.taggers):
                counts = awkward.num(diphotons.pt, axis=1)
                flat_tags = numpy.stack(
                    (
                        awkward.flatten(
                            diphotons[
                                "_".join([tagger.name, str(tagger.priority)])
                            ]
                        )
                        for tagger in self.taggers
                    ),
                    axis=1,
                )
                tags = awkward.from_regular(
                    awkward.unflatten(flat_tags, counts), axis=2
                )
                winner = awkward.min(tags[tags != 0], axis=2)
                diphotons["best_tag"] = winner

                # lowest priority is most important (ascending sort)
                # leave in order of diphoton pT in case of ties (stable sort)
                sorted = awkward.argsort(diphotons.best_tag, stable=True)
                diphotons = diphotons[sorted]

            diphotons = awkward.firsts(diphotons)
            # set diphotons as part of the event record
            events[f"diphotons_{do_variation}"] = diphotons
            # annotate diphotons with event information
            diphotons["event"] = events.event
            diphotons["lumi"] = events.luminosityBlock
            diphotons["run"] = events.run
            # nPV just for validation of pileup reweighting
            diphotons["nPV"] = events.PV.npvs
            diphotons["fixedGridRhoAll"] = events.Rho.fixedGridRhoAll
            # annotate diphotons with dZ information (difference between z position of GenVtx and PV) as required by flashggfinalfits
            if self.data_kind == "mc":
                diphotons["genWeight"] = events.genWeight
                diphotons["dZ"] = events.GenVtx.z - events.PV.z
                # Necessary for differential xsec measurements in final fits ("truth" variables)
                diphotons["HTXS_Higgs_pt"] = events.HTXS.Higgs_pt
                diphotons["HTXS_Higgs_y"] = events.HTXS.Higgs_y
                diphotons["HTXS_njets30"] = events.HTXS.njets30  # Need to clarify if this variable is suitable, does it fulfill abs(eta_j) < 2.5? Probably not
                # Preparation for HTXS measurements later, start with stage 0 to disentangle VH into WH and ZH for final fits
                diphotons["HTXS_stage_0"] = events.HTXS.stage_0
            # Fill zeros for data because there is no GenVtx for data, obviously
            else:
                diphotons["dZ"] = awkward.zeros_like(events.PV.z)

            # drop events without a preselected diphoton candidate
            # drop events without a tag, if there are tags
            if len(self.taggers):
                selection_mask = ~(
                    awkward.is_none(diphotons)
                    | awkward.is_none(diphotons.best_tag)
                )
                diphotons = diphotons[selection_mask]
            else:
                selection_mask = ~awkward.is_none(diphotons)
                diphotons = diphotons[selection_mask]

            # return if there is no surviving events
            if len(diphotons) == 0:
                # logger.debug("No surviving events in this run, return now!")
                logger.debug("No surviving events in this run. Writing empty parquet.")
                # return histos_etc
            if self.data_kind == "mc":
                # initiate Weight container here, after selection, since event selection cannot easily be applied to weight container afterwards
                event_weights = Weights(size=len(events[selection_mask]))

                # corrections to event weights:
                for correction_name in correction_names:
                    if correction_name in available_weight_corrections:
                        logger.info(
                            f"Adding correction {correction_name} to weight collection of dataset {dataset_name}"
                        )
                        varying_function = available_weight_corrections[
                            correction_name
                        ]
                        event_weights = varying_function(
                            events=events[selection_mask],
                            photons=events[f"diphotons_{do_variation}"][
                                selection_mask
                            ],
                            weights=event_weights,
                            dataset_name=dataset_name,
                            year=self.year[dataset_name][0],
                        )

                # systematic variations of event weights go to nominal output dataframe:
                if do_variation == "nominal":
                    for systematic_name in systematic_names:
                        if systematic_name in available_weight_systematics:
                            logger.info(
                                f"Adding systematic {systematic_name} to weight collection of dataset {dataset_name}"
                            )
                            if systematic_name == "LHEScale":
                                if hasattr(events, "LHEScaleWeight"):
                                    diphotons["nweight_LHEScale"] = awkward.num(
                                        events.LHEScaleWeight[selection_mask],
                                        axis=1,
                                    )
                                    diphotons[
                                        "weight_LHEScale"
                                    ] = events.LHEScaleWeight[selection_mask]
                                else:
                                    logger.info(
                                        f"No {systematic_name} Weights in dataset {dataset_name}"
                                    )
                            elif systematic_name == "LHEPdf":
                                if hasattr(events, "LHEPdfWeight"):
                                    # two AlphaS weights are removed
                                    diphotons["nweight_LHEPdf"] = (
                                        awkward.num(
                                            events.LHEPdfWeight[selection_mask],
                                            axis=1,
                                        )
                                        - 2
                                    )
                                    diphotons[
                                        "weight_LHEPdf"
                                    ] = events.LHEPdfWeight[selection_mask][
                                        :, :-2
                                    ]
                                else:
                                    logger.info(
                                        f"No {systematic_name} Weights in dataset {dataset_name}"
                                    )
                            else:
                                varying_function = available_weight_systematics[
                                    systematic_name
                                ]
                                event_weights = varying_function(
                                    events=events[selection_mask],
                                    photons=events[f"diphotons_{do_variation}"][
                                        selection_mask
                                    ],
                                    weights=event_weights,
                                    dataset_name=dataset_name,
                                    year=self.year[dataset_name][0],
                                )

                diphotons["weight_central"] = event_weights.weight()
                # Store variations with respect to central weight
                if do_variation == "nominal":
                    if len(event_weights.variations):
                        logger.info(
                            "Adding systematic weight variations to nominal output file."
                        )
                    for modifier in event_weights.variations:
                        diphotons["weight_" + modifier] = event_weights.weight(
                            modifier=modifier
                        )

                # Multiply weight by genWeight for normalisation in post-processing chain
                event_weights._weight = (
                    events["genWeight"][selection_mask]
                    * diphotons["weight_central"]
                )
                diphotons["weight"] = event_weights.weight()

            # Add weight variables (=1) for data for consistent datasets
            else:
                diphotons["weight_central"] = awkward.ones_like(
                    diphotons["event"]
                )
                diphotons["weight"] = awkward.ones_like(diphotons["event"])


            if self.output_location is not None:
                if self.output_format == "root":
                    df = diphoton_list_to_pandas(self, diphotons)
                else:
                    akarr = diphoton_ak_array(self, diphotons)

                    # Remove fixedGridRhoAll from photons to avoid having event-level info per photon
                    akarr = akarr[
                        [
                            field
                            for field in akarr.fields
                            if "lead_fixedGridRhoAll" not in field
                        ]
                    ]

                fname = (
                    events.attrs["@events_factory"]
                    ._partition_key.replace("/", "_")
                    + f".{self.output_format}"
                )

                fname = fname.replace("%2F", "").replace("%3B1", "")

                subdirs = []
                if "dataset" in events.metadata:
                    subdirs.append(events.metadata["dataset"])

                subdirs.append(do_variation)

                if self.output_format == "root":
                    dump_pandas(self, df, fname, self.output_location, subdirs)
                else:
                    dump_ak_array(
                        self,
                        akarr,
                        fname,
                        self.output_location,
                        metadata,
                        subdirs,
                    )


        return histos_etc

    def process_extra(self, events: awkward.Array) -> awkward.Array:
        return events, {}

    def postprocess(self, accumulant: Dict[Any, Any]) -> Any:
        pass
