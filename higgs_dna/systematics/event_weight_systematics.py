import numpy as np
import json
import os
from scipy.interpolate import interp1d
import correctionlib
import awkward as ak
from higgs_dna.utils.misc_utils import choose_jet
from higgs_dna.utils.misc_utils import evaluate_ctag_wp, evaluate_btag_sf_eff_multiwp, compute_btag_multiwp
from higgs_dna.tools.gen_helpers import get_genJets
import logging
import ast

logger = logging.getLogger(__name__)


def SF_photon_ID(
    photons, weights, year="2017", WP="Loose", is_correction=True, **kwargs
):
    """
    Applies the photon ID scale-factor and corresponding uncertainties for the customised cut on the EGamma MVA ID (Run 3)
    JLS removed the support for the EGamma MVA ID SFs from https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration for Run 2 for now as this is not commonly used in the Hgg group
    Take action yourself or contact us if you need those!
    """
    # era/year defined as parameter of the function
    avail_years = ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024"]
    if year not in avail_years:
        logger.warning(f"\n WARNING: only photon ID SFs for the year strings {avail_years} are already implemented! \n Exiting. \n")
        logger.warning("If you need the SFs for the central Egamma MVA ID for Run 2 UL, take action yourself or contact us!")
        exit()

    if year == "2022preEE":
        json_file = os.path.join(os.path.dirname(__file__), "JSONs/SF_photon_ID/2022/PhotonIDMVA_2022PreEE.json")
    elif year == "2022postEE":
        json_file = os.path.join(os.path.dirname(__file__), "JSONs/SF_photon_ID/2022/PhotonIDMVA_2022PostEE.json")
    elif year == "2023preBPix":
        json_file = os.path.join(os.path.dirname(__file__), "JSONs/SF_photon_ID/2023preBPix/IDMVA0p19_2023PreBPiX.json")
    elif year == "2023postBPix":
        json_file = os.path.join(os.path.dirname(__file__), "JSONs/SF_photon_ID/2023postBPix/IDMVA0p19_2023PostBPiX.json")
    elif year == "2024":
        json_file = os.path.join(os.path.dirname(__file__), "JSONs/SF_photon_ID/2024/2024_phoid0p24_SF.json")

    if "2023" in year or "2024" in year:
        evaluator = correctionlib.CorrectionSet.from_file(json_file)["IDMVA_SF"]
    else:
        evaluator = correctionlib.CorrectionSet.from_file(json_file)["PhotonIDMVA_SF"]

    # In principle, we should use the fully correct formula https://indico.cern.ch/event/1360948/contributions/5783762/attachments/2788516/4870824/24_02_02_HIG-23-014_PreAppPres.pdf#page=7
    # However, if the SF is pt-binned, the approximation of the multiplication of the two SFs is fully exact
    # N.B. These phoID SFs are computed for the workin point optimised for the fiducial XS analysis (0.25 for 22, and 0.19 for 23)
    if "2022" in year or "2023" in year or "2024" in year:
        if is_correction:
            # only calculate correction to nominal weight
            if "2024" in year:
                sf_lead = evaluator.evaluate(
                    abs(photons["pho_lead"].ScEta), photons["pho_lead"].r9, photons["pho_lead"].pt, "nominal"
                )
                sf_sublead = evaluator.evaluate(
                    abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].r9, photons["pho_sublead"].pt, "nominal"
                )
            else:
                sf_lead = evaluator.evaluate(
                    abs(photons["pho_lead"].ScEta), photons["pho_lead"].pt, "nominal"
                )
                sf_sublead = evaluator.evaluate(
                    abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].pt, "nominal"
                )
            sf = sf_lead * sf_sublead

            sfup, sfdown = None, None

        else:
            # only calculate systs

            sf = np.ones(len(weights._weight))
            if "2024" in year:
                sf_lead = evaluator.evaluate(
                    abs(photons["pho_lead"].ScEta), photons["pho_lead"].r9, photons["pho_lead"].pt, "nominal"
                )
                sf_sublead = evaluator.evaluate(
                    abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].r9, photons["pho_sublead"].pt, "nominal"
                )
                _sf = sf_lead * sf_sublead

                sf_unc_lead = evaluator.evaluate(
                    abs(photons["pho_lead"].ScEta), photons["pho_lead"].r9, photons["pho_lead"].pt, "uncertainty"
                )
                sf_unc_sublead = evaluator.evaluate(
                    abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].r9, photons["pho_sublead"].pt, "uncertainty"
                )
            else:
                sf_lead = evaluator.evaluate(
                    abs(photons["pho_lead"].ScEta), photons["pho_lead"].pt, "nominal"
                )
                sf_sublead = evaluator.evaluate(
                    abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].pt, "nominal"
                )
                _sf = sf_lead * sf_sublead

                sf_unc_lead = evaluator.evaluate(
                    abs(photons["pho_lead"].ScEta), photons["pho_lead"].pt, "uncertainty"
                )
                sf_unc_sublead = evaluator.evaluate(
                    abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].pt, "uncertainty"
                )

            sfup = (sf_lead + sf_unc_lead) * (sf_sublead + sf_unc_sublead) / _sf

            sfdown = (sf_lead - sf_unc_lead) * (sf_sublead - sf_unc_sublead) / _sf

    name = "SF_photon_ID_corr" if is_correction else "SF_photon_ID"
    weights.add(name=name, weight=sf, weightUp=sfup, weightDown=sfdown)

    return weights


def Pileup(events, weights, year, is_correction=True, **kwargs):
    """
    Function to apply either the pileup correction to MC to make it match the pileup profile of a certain year/period,
    or the respective uncertainties.
    The parameter `year` needs to be specified as one of ["2022preEE", "2022postEE", "23preBPix", "23postBPix"] for Run-3 or ["2016preVFP", "2016postVFP", "2017", "2018"] for Run-2.
    By now, the Run-2 and Run-3 up to 2023D files are available from LUM POG in the correctionlib format...
    The pileup histos for Run-3 were produced by Junquan, the JSONs for Run-2 and Run-3 first need to be pulled with `scripts/pull_files.py`!
    """
    path_to_json = os.path.join(os.path.dirname(__file__), "../systematics/JSONs/pileup/pileup_{}.json.gz".format(year))
    if "16" in year:
        name = "Collisions16_UltraLegacy_goldenJSON"
    elif "17" in year:
        name = "Collisions17_UltraLegacy_goldenJSON"
    elif "18" in year:
        name = "Collisions18_UltraLegacy_goldenJSON"
    elif "22preEE" in year:
        name = "Collisions2022_355100_357900_eraBCD_GoldenJson"
    elif "22postEE" in year:
        name = "Collisions2022_359022_362760_eraEFG_GoldenJson"
    elif "23preBPix" in year:
        name = "Collisions2023_366403_369802_eraBC_GoldenJson"
    elif "23postBPix" in year:
        name = "Collisions2023_369803_370790_eraD_GoldenJson"
    elif "24" in year:
        name = "Collisions24_BCDEFGHI_goldenJSON"
    elif "25" in year:
        name = "Collisions25_Prompt_goldenJSON"

    evaluator = correctionlib.CorrectionSet.from_file(path_to_json)[name]

    if is_correction:
        sf = evaluator.evaluate(events.Pileup.nTrueInt, "nominal")
        sfup, sfdown = None, None
    else:
        sf = np.ones(len(weights._weight))
        sf_nom = evaluator.evaluate(events.Pileup.nTrueInt, "nominal")

        sfup = evaluator.evaluate(events.Pileup.nTrueInt, "up") / sf_nom
        sfdown = evaluator.evaluate(events.Pileup.nTrueInt, "down") / sf_nom

    name = "Pileup_corr" if is_correction else "Pileup"
    weights.add(name=name, weight=sf, weightUp=sfup, weightDown=sfdown)

    return weights


def L1PreFiring(events, weights, year="2017", is_correction=True, **kwargs):
    """Function to apply either the L1Prefiring correction to simulation to make it match the prefiring rate of a certain year/period, or the respective uncertainties. The parameter `year` needs to be specified as one of ["2016preVFP", "2016postVFP", "2017", "2018"] for Run-2. Note that the weights are stored in both Run 2 NanoAODv15 and NanoAODv9."""
    avail_years = ["2016", "2016preVFP", "2016postVFP", "2017", "2018"]
    if year not in avail_years:
        raise ValueError(
            f"Invalid year '{year}'. Only the following years are supported for L1PreFiring corrections: {avail_years} \n")
    if "L1PreFiringWeight" not in events.fields:
        logger.info("No L1Prefiring weights found in the events, skipping systematic/correction.")
        return weights
    if is_correction:
        sf = events.L1PreFiringWeight.Nom
        sfup, sfdown = None, None
    else:
        sf = np.ones(len(events))
        sf_nom = events.L1PreFiringWeight.Nom
        sfup = events.L1PreFiringWeight.Up / sf_nom
        sfdown = events.L1PreFiringWeight.Dn / sf_nom

    name = "L1Prefiring_corr" if is_correction else "L1Prefiring"
    weights.add(name=name, weight=sf, weightUp=sfup, weightDown=sfdown)
    return weights


def LoosePhoIdSF(photons, weights, year="2017", is_correction=True, **kwargs):
    """
    LoosePhoIdSF: correction to the event weight on a per photon level, impacting one of the high importance input variable of the DiphotonBDT, binned in eta and r9.
    for original implementation look at: https://github.com/cms-analysis/flashgg/blob/2677dfea2f0f40980993cade55144636656a8a4f/Systematics/python/flashggDiPhotonSystematics2017_Legacy_cfi.py
    And for presentation on the study: https://indico.cern.ch/event/963617/contributions/4103623/attachments/2141570/3608645/Zee_Validation_UL2017_Update_09112020_Prasant.pdf

    Run2: Taken from flashgg, the SFs correspond to the custom Hgg PhoID, the loose working point was set to -0.9
    Run3: Computed with the EGM PhoID, the loose working point is set to -0.7
    """

    # era/year defined as parameter of the function, only 2017 is implemented up to now
    avail_years = ["2016", "2016preVFP", "2016postVFP", "2017", "2018", "2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024"]
    if year not in avail_years:
        print(f"\n WARNING: only LoosePhoIDSF corrections for the year strings {avail_years} are already implemented! \n Exiting. \n")
        exit()
    elif "2016" in year:
        year = "2016"

    json_file = os.path.join(os.path.dirname(__file__), f"JSONs/LoosePhoIDSF/{year}/LoosePhoIDSF_{year}.json")
    if year in ["2016", "2017", "2018"]:
        evaluator = correctionlib.CorrectionSet.from_file(json_file)["LooseMvaSF"]
        if is_correction:
            # only calculate correction to nominal weight
            sf_lead = evaluator.evaluate(
                "nominal", photons["pho_lead"].ScEta, photons["pho_lead"].r9
            )
            sf_sublead = evaluator.evaluate(
                "nominal", photons["pho_sublead"].ScEta, photons["pho_sublead"].r9
            )
            sf = sf_lead * sf_sublead

            sfup, sfdown = None, None

        else:
            # only calculate systs
            sf = np.ones(len(weights._weight))
            sf_lead = evaluator.evaluate(
                "nominal", photons["pho_lead"].ScEta, photons["pho_lead"].r9
            )
            sf_sublead = evaluator.evaluate(
                "nominal", photons["pho_sublead"].ScEta, photons["pho_sublead"].r9
            )
            _sf = sf_lead * sf_sublead

            sfup_lead = evaluator.evaluate(
                "up", photons["pho_lead"].ScEta, photons["pho_lead"].r9
            )
            sfup_sublead = evaluator.evaluate(
                "up", photons["pho_sublead"].ScEta, photons["pho_sublead"].r9
            )
            sfup = sfup_lead * sfup_sublead / _sf

            sfdown_lead = evaluator.evaluate(
                "down", photons["pho_lead"].ScEta, photons["pho_lead"].r9
            )
            sfdown_sublead = evaluator.evaluate(
                "down", photons["pho_sublead"].ScEta, photons["pho_sublead"].r9
            )
            sfdown = sfdown_lead * sfdown_sublead / _sf

    elif year in ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024"]:
        evaluator = correctionlib.CorrectionSet.from_file(json_file)["IDMVA_SF"]
        if is_correction:
            # only calculate correction to nominal weight
            sf_lead = evaluator.evaluate(
                abs(photons["pho_lead"].ScEta), photons["pho_lead"].r9, photons["pho_lead"].pt, "nominal"
            )
            sf_sublead = evaluator.evaluate(
                abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].r9, photons["pho_sublead"].pt, "nominal"
            )
            sf = sf_lead * sf_sublead

            sfup, sfdown = None, None

        else:
            # only calculate systs
            sf = np.ones(len(weights._weight))

            sf_lead = evaluator.evaluate(
                abs(photons["pho_lead"].ScEta), photons["pho_lead"].r9, photons["pho_lead"].pt, "nominal"
            )
            sf_sublead = evaluator.evaluate(
                abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].r9, photons["pho_sublead"].pt, "nominal"
            )
            _sf = sf_lead * sf_sublead

            sf_unc_lead = evaluator.evaluate(
                abs(photons["pho_lead"].ScEta), photons["pho_lead"].r9, photons["pho_lead"].pt, "uncertainty"
            )
            sf_unc_sublead = evaluator.evaluate(
                abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].r9, photons["pho_sublead"].pt, "uncertainty"
            )

            sfup = (sf_lead + sf_unc_lead) * (sf_sublead + sf_unc_sublead) / _sf

            sfdown = (sf_lead - sf_unc_lead) * (sf_sublead - sf_unc_sublead) / _sf

    name = "LoosePhoIDSF_corr" if is_correction else "LoosePhoIDSF"
    weights.add(name=name, weight=sf, weightUp=sfup, weightDown=sfdown)

    return weights


def ElectronVetoSF(photons, weights, year="2017", is_correction=True, **kwargs):
    """
    ElectronVetoSF: correction to the event weight on a per photon level, Conversion safe veto efficiency with event counting method: To check if the FSR photons are passing the e-veto or not.
    binned in abs(SCeta) and r9.
    for original implementation look at: https://github.com/cms-analysis/flashgg/blob/2677dfea2f0f40980993cade55144636656a8a4f/Systematics/python/flashggDiPhotonSystematics2017_Legacy_cfi.py
    And for presentation on the study: https://indico.cern.ch/event/961164/contributions/4089584/attachments/2135019/3596299/Zmmg_UL2017%20With%20CorrMC_Hgg%20(02.11.2020).pdf
    """

    # era/year defined as parameter of the function
    avail_years = ["2016", "2016preVFP", "2016postVFP", "2017", "2018", "2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024"]
    if year not in avail_years:
        logger.warning(f"\n WARNING: only eVetoSF corrections for the year strings {avail_years} are already implemented! \n Exiting. \n")
        exit()
    elif "2016" in year:
        year = "2016"

    if year in ["2016", "2017", "2018"]:
        # 2017 file should be renamed with the year in its name...
        json_file = os.path.join(os.path.dirname(__file__), f"JSONs/ElectronVetoSF/{year}/eVetoSF_{year}.json")
        evaluator = correctionlib.CorrectionSet.from_file(json_file)["ElectronVetoSF"]
        if is_correction:
            # only calculate correction to nominal weight
            sf_lead = evaluator.evaluate(
                "nominal", abs(photons["pho_lead"].ScEta), photons["pho_lead"].r9
            )
            sf_sublead = evaluator.evaluate(
                "nominal", abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].r9
            )
            sf = sf_lead * sf_sublead

            sfup, sfdown = None, None

        else:
            # only calculate systs
            sf = np.ones(len(weights._weight))
            sf_lead = evaluator.evaluate(
                "nominal", abs(photons["pho_lead"].ScEta), photons["pho_lead"].r9
            )
            sf_sublead = evaluator.evaluate(
                "nominal", abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].r9
            )
            _sf = sf_lead * sf_sublead

            sfup_lead = evaluator.evaluate(
                "up", abs(photons["pho_lead"].ScEta), photons["pho_lead"].r9
            )
            sfup_sublead = evaluator.evaluate(
                "up", abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].r9
            )
            sfup = sfup_lead * sfup_sublead / _sf

            sfdown_lead = evaluator.evaluate(
                "down", abs(photons["pho_lead"].ScEta), photons["pho_lead"].r9
            )
            sfdown_sublead = evaluator.evaluate(
                "down", abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].r9
            )
            sfdown = sfdown_lead * sfdown_sublead / _sf

    elif year in ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024"]:
        # presentation of the updated 2022 SF with dR>0.1: https://indico.cern.ch/event/1536748/contributions/6471184/attachments/3056856/5405041/202504_Zmmg_eveto_DRG0p1_ForEG_Updated.pdf
        if year == "2022preEE":
            json_file = os.path.join(os.path.dirname(__file__), "JSONs/ElectronVetoSF/2022/preEE_CSEV_SFcorrections.json")
        if year == "2022postEE":
            json_file = os.path.join(os.path.dirname(__file__), "JSONs/ElectronVetoSF/2022/postEE_CSEV_SFcorrections.json")
        # presentation of 2023 SF with dR>0.1: https://indico.cern.ch/event/1536748/contributions/6471184/attachments/3056856/5405041/202504_Zmmg_eveto_DRG0p1_ForEG_Updated.pdf
        if year == "2023preBPix":
            json_file = os.path.join(os.path.dirname(__file__), "JSONs/ElectronVetoSF/2023/preBPix_CSEV_SFcorrections.json")
        if year == "2023postBPix":
            json_file = os.path.join(os.path.dirname(__file__), "JSONs/ElectronVetoSF/2023/postBPix_CSEV_SFcorrections.json")
        # Preliminary 2024 results, has to be changed once the official SFs are available
        if year == "2024":
            json_file = os.path.join(os.path.dirname(__file__), "JSONs/ElectronVetoSF/2024/CSEV_SFcorrections.json")
        evaluator = correctionlib.CorrectionSet.from_file(json_file)["CSEV_SFs"]

        if is_correction:
            # only calculate correction to nominal weight
            sf_lead = evaluator.evaluate(
                abs(photons["pho_lead"].ScEta), photons["pho_lead"].r9, "nominal"
            )
            sf_sublead = evaluator.evaluate(
                abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].r9, "nominal"
            )
            sf = sf_lead * sf_sublead

            sfup, sfdown = None, None

        else:
            # only calculate systs
            sf = np.ones(len(weights._weight))
            sf_lead = evaluator.evaluate(
                abs(photons["pho_lead"].ScEta), photons["pho_lead"].r9, "nominal"
            )
            sf_sublead = evaluator.evaluate(
                abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].r9, "nominal"
            )
            _sf = sf_lead * sf_sublead

            unc_lead = evaluator.evaluate(
                abs(photons["pho_lead"].ScEta), photons["pho_lead"].r9, "uncertainty"
            )
            unc_sublead = evaluator.evaluate(
                abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].r9, "uncertainty"
            )

            sfup = (sf_lead + unc_lead) * (sf_sublead + unc_sublead) / _sf
            sfdown = (sf_lead - unc_lead) * (sf_sublead - unc_sublead) / _sf

    name = "ElectronVetoSF_corr" if is_correction else "ElectronVetoSF"
    weights.add(name=name, weight=sf, weightUp=sfup, weightDown=sfdown)

    return weights


def PreselSF(photons, weights, year="2017", is_correction=True, **kwargs):
    """
    Preselection SF: correction to the event weight on a per photon level for UL2017. Dt:17/11/2020
    Binned in abs(SCeta) and r9.
    For original implementation look at: https://github.com/cms-analysis/flashgg/blob/2677dfea2f0f40980993cade55144636656a8a4f/Systematics/python/flashggDiPhotonSystematics2017_Legacy_cfi.py
    Link to the Presentation: https://indico.cern.ch/event/963617/contributions/4103623/attachments/2141570/3608645/Zee_Validation_UL2017_Update_09112020_Prasant.pdf
    """

    # era/year defined as parameter of the function
    avail_years = ["2016", "2016preVFP", "2016postVFP", "2017", "2018", "2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024"]
    if year not in avail_years:
        logger.warning(f"\n WARNING: only PreselSF corrections for the year strings {avail_years} are already implemented! \n Exiting. \n")
        exit()
    elif "2016" in year:
        year = "2016"

    if year in ["2016", "2017", "2018"]:
        json_file = os.path.join(os.path.dirname(__file__), f"JSONs/Preselection/{year}/PreselSF_{year}.json")
    elif year == "2022preEE":
        json_file = os.path.join(os.path.dirname(__file__), "JSONs/Preselection/2022/Preselection_2022PreEE.json")
    elif year == "2022postEE":
        json_file = os.path.join(os.path.dirname(__file__), "JSONs/Preselection/2022/Preselection_2022PostEE.json")
    elif year == "2023preBPix":
        json_file = os.path.join(os.path.dirname(__file__), "JSONs/Preselection/2023preBPix/Preselection_2023PreBPix.json")
    elif year == "2023postBPix":
        json_file = os.path.join(os.path.dirname(__file__), "JSONs/Preselection/2023postBPix/Preselection_2023PostBPiX.json")
    elif year == "2024":
        json_file = os.path.join(os.path.dirname(__file__), "JSONs/Preselection/2024/Preselection_2024.json")

    if year in ["2016", "2017", "2018"]:
        evaluator = correctionlib.CorrectionSet.from_file(json_file)["PreselSF"]
    elif "2022" in year or "2023" in year or "2024" in year:
        evaluator = correctionlib.CorrectionSet.from_file(json_file)["Preselection_SF"]

    if year in ["2016", "2017", "2018"]:
        if is_correction:
            # only calculate correction to nominal weight
            sf_lead = evaluator.evaluate(
                "nominal", abs(photons["pho_lead"].ScEta), photons["pho_lead"].r9
            )
            sf_sublead = evaluator.evaluate(
                "nominal", abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].r9
            )
            sf = sf_lead * sf_sublead

            sfup, sfdown = None, None

        else:
            # only calculate systs
            sf = np.ones(len(weights._weight))
            sf_lead = evaluator.evaluate(
                "nominal", abs(photons["pho_lead"].ScEta), photons["pho_lead"].r9
            )
            sf_sublead = evaluator.evaluate(
                "nominal", abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].r9
            )
            _sf = sf_lead * sf_sublead

            sfup_lead = evaluator.evaluate(
                "up", abs(photons["pho_lead"].ScEta), photons["pho_lead"].r9
            )
            sfup_sublead = evaluator.evaluate(
                "up", abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].r9
            )
            sfup = sfup_lead * sfup_sublead / _sf

            sfdown_lead = evaluator.evaluate(
                "down", abs(photons["pho_lead"].ScEta), photons["pho_lead"].r9
            )
            sfdown_sublead = evaluator.evaluate(
                "down", abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].r9
            )
            sfdown = sfdown_lead * sfdown_sublead / _sf

    # In principle, we should use the fully correct formula https://indico.cern.ch/event/1360948/contributions/5783762/attachments/2788516/4870824/24_02_02_HIG-23-014_PreAppPres.pdf#page=7
    # However, if the SF is pt-binned, the approximation of the multiplication of the two SFs is fully exact
    # N.B. The preselection SFs for Run3 are without the loose photon ID cut
    elif "2022" in year or "2023" in year or "2024" in year:
        if is_correction:
            # only calculate correction to nominal weight
            sf_lead = evaluator.evaluate(
                abs(photons["pho_lead"].ScEta), photons["pho_lead"].r9, photons["pho_lead"].pt, "nominal"
            )
            sf_sublead = evaluator.evaluate(
                abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].r9, photons["pho_sublead"].pt, "nominal"
            )
            sf = sf_lead * sf_sublead

            sfup, sfdown = None, None

        else:
            # only calculate systs

            # Slightly different calculation compared to 2017
            # In the 2022 JSONs, the delta is saved as the uncertainty, not the up/down variations of (SF+-delta) themselves
            # Note that the uncertainty is assumed to be symmetric

            sf = np.ones(len(weights._weight))
            sf_lead = evaluator.evaluate(
                abs(photons["pho_lead"].ScEta), photons["pho_lead"].r9, photons["pho_lead"].pt, "nominal"
            )
            sf_sublead = evaluator.evaluate(
                abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].r9, photons["pho_sublead"].pt, "nominal"
            )
            _sf = sf_lead * sf_sublead

            sf_unc_lead = evaluator.evaluate(
                abs(photons["pho_lead"].ScEta), photons["pho_lead"].r9, photons["pho_lead"].pt, "uncertainty"
            )
            sf_unc_sublead = evaluator.evaluate(
                abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].r9, photons["pho_sublead"].pt, "uncertainty"
            )

            sfup = (sf_lead + sf_unc_lead) * (sf_sublead + sf_unc_sublead) / _sf

            sfdown = (sf_lead - sf_unc_lead) * (sf_sublead - sf_unc_sublead) / _sf

    name = "PreselSF_corr" if is_correction else "PreselSF"
    weights.add(name=name, weight=sf, weightUp=sfup, weightDown=sfdown)

    return weights


def TriggerSF(photons, weights, year="2017", is_correction=True, **kwargs):
    """
    Trigger SF: for full 2017 legacy  B-F dataset. Trigger scale factors for use without HLT applied in MC
    Binned in abs(SCeta), r9 and pt.
    For original implementation look at: https://github.com/cms-analysis/flashgg/blob/2677dfea2f0f40980993cade55144636656a8a4f/Systematics/python/flashggDiPhotonSystematics2017_Legacy_cfi.py
    """

    # era/year defined as parameter of the function
    avail_years = ["2016", "2016preVFP", "2016postVFP", "2017", "2018", "2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024", "2025"]
    if year not in avail_years:
        logger.warning(f"\n WARNING: only TriggerSF corrections for the year strings {avail_years} are already implemented! \n Exiting. \n")
        exit()
    elif "2016" in year:
        year = "2016"

    if year in ["2016", "2017", "2018", "2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024", "2025"]:
        json_file_lead = os.path.join(os.path.dirname(__file__), f"JSONs/TriggerSF/{year}/TriggerSF_lead_{year}.json")
        json_file_sublead = os.path.join(os.path.dirname(__file__), f"JSONs/TriggerSF/{year}/TriggerSF_sublead_{year}.json")

    evaluator_lead = correctionlib.CorrectionSet.from_file(json_file_lead)["TriggerSF"]
    evaluator_sublead = correctionlib.CorrectionSet.from_file(json_file_sublead)["TriggerSF"]

    if year in ["2016", "2017", "2018"]:
        if is_correction:
            # only calculate correction to nominal weight
            sf_lead = evaluator_lead.evaluate(
                "nominal", abs(photons["pho_lead"].ScEta), photons["pho_lead"].r9, photons["pho_lead"].pt
            )
            sf_sublead = evaluator_sublead.evaluate(
                "nominal", abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].r9, photons["pho_sublead"].pt
            )
            sf = sf_lead * sf_sublead

            sfup, sfdown = None, None

        else:
            # only calculate systs
            sf = np.ones(len(weights._weight))
            sf_lead = evaluator_lead.evaluate(
                "nominal", abs(photons["pho_lead"].ScEta), photons["pho_lead"].r9, photons["pho_lead"].pt
            )
            sf_sublead = evaluator_sublead.evaluate(
                "nominal", abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].r9, photons["pho_sublead"].pt
            )
            _sf = sf_lead * sf_sublead

            sfup_lead = evaluator_lead.evaluate(
                "up", abs(photons["pho_lead"].ScEta), photons["pho_lead"].r9, photons["pho_lead"].pt
            )
            sfup_sublead = evaluator_sublead.evaluate(
                "up", abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].r9, photons["pho_sublead"].pt
            )
            sfup = sfup_lead * sfup_sublead / _sf

            sfdown_lead = evaluator_lead.evaluate(
                "down", abs(photons["pho_lead"].ScEta), photons["pho_lead"].r9, photons["pho_lead"].pt
            )
            sfdown_sublead = evaluator_sublead.evaluate(
                "down", abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].r9, photons["pho_sublead"].pt
            )
            sfdown = sfdown_lead * sfdown_sublead / _sf

    elif year in ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024", "2025"]:

        # If flow corrections are applied, we use the raw (uncorrected) r9 for the trigger SF evaluation
        if hasattr(photons["pho_lead"], 'raw_r9'):
            sf_lead_p_lead = evaluator_lead.evaluate(
                "nominal", abs(photons["pho_lead"].ScEta), photons["pho_lead"].raw_r9, photons["pho_lead"].pt
            )
            sf_lead_p_sublead = evaluator_lead.evaluate(
                "nominal", abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].raw_r9, photons["pho_sublead"].pt
            )
            sf_sublead_p_lead = evaluator_sublead.evaluate(
                "nominal", abs(photons["pho_lead"].ScEta), photons["pho_lead"].raw_r9, photons["pho_lead"].pt
            )
            sf_sublead_p_sublead = evaluator_sublead.evaluate(
                "nominal", abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].raw_r9, photons["pho_sublead"].pt
            )
        else:
            sf_lead_p_lead = evaluator_lead.evaluate(
                "nominal", abs(photons["pho_lead"].ScEta), photons["pho_lead"].r9, photons["pho_lead"].pt
            )
            sf_lead_p_sublead = evaluator_lead.evaluate(
                "nominal", abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].r9, photons["pho_sublead"].pt
            )
            sf_sublead_p_lead = evaluator_sublead.evaluate(
                "nominal", abs(photons["pho_lead"].ScEta), photons["pho_lead"].r9, photons["pho_lead"].pt
            )
            sf_sublead_p_sublead = evaluator_sublead.evaluate(
                "nominal", abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].r9, photons["pho_sublead"].pt
            )

        if is_correction:
            # only calculate correction to nominal weight
            sf = sf_lead_p_lead * sf_sublead_p_sublead + sf_lead_p_sublead * sf_sublead_p_lead - sf_lead_p_lead * sf_lead_p_sublead

            sfup, sfdown = None, None

        else:
            # only calculate systs
            sf = np.ones(len(weights._weight))
            # get nominal SF to divide it out
            _sf = sf_lead_p_lead * sf_sublead_p_sublead + sf_lead_p_sublead * sf_sublead_p_lead - sf_lead_p_lead * sf_lead_p_sublead

            # up SF
            sfup_lead_p_lead = evaluator_lead.evaluate(
                "up", abs(photons["pho_lead"].ScEta), photons["pho_lead"].r9, photons["pho_lead"].pt
            )
            sfup_lead_p_sublead = evaluator_lead.evaluate(
                "up", abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].r9, photons["pho_sublead"].pt
            )
            sfup_sublead_p_lead = evaluator_sublead.evaluate(
                "up", abs(photons["pho_lead"].ScEta), photons["pho_lead"].r9, photons["pho_lead"].pt
            )
            sfup_sublead_p_sublead = evaluator_sublead.evaluate(
                "up", abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].r9, photons["pho_sublead"].pt
            )
            sfup = (sfup_lead_p_lead * sfup_sublead_p_sublead + sfup_lead_p_sublead * sfup_sublead_p_lead - sfup_lead_p_lead * sfup_lead_p_sublead) / _sf

            # down SF
            sfdown_lead_p_lead = evaluator_lead.evaluate(
                "down", abs(photons["pho_lead"].ScEta), photons["pho_lead"].r9, photons["pho_lead"].pt
            )
            sfdown_lead_p_sublead = evaluator_lead.evaluate(
                "down", abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].r9, photons["pho_sublead"].pt
            )
            sfdown_sublead_p_lead = evaluator_sublead.evaluate(
                "down", abs(photons["pho_lead"].ScEta), photons["pho_lead"].r9, photons["pho_lead"].pt
            )
            sfdown_sublead_p_sublead = evaluator_sublead.evaluate(
                "down", abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].r9, photons["pho_sublead"].pt
            )
            sfdown = (sfdown_lead_p_lead * sfdown_sublead_p_sublead + sfdown_lead_p_sublead * sfdown_sublead_p_lead - sfdown_lead_p_lead * sfdown_lead_p_sublead) / _sf

    name = "TriggerSF_corr" if is_correction else "TriggerSF"
    weights.add(name=name, weight=sf, weightUp=sfup, weightDown=sfdown)

    return weights


def calculate_NNLOPS_sf(events, dataset_name, generator):
    json_file = os.path.join(os.path.dirname(__file__), "JSONs/NNLOPS_reweight.json")
    if (
        all(s not in dataset_name.lower() for s in ('glugluhh', 'gghh'))
        and any(s in dataset_name.lower() for s in ("ggh", "glugluh"))
    ):
        # Extract NNLOPS weights from json file
        json_file = os.path.join(os.path.dirname(__file__), "JSONs/NNLOPS_reweight.json")
        with open(json_file, "r") as jf:
            nnlops_reweight = json.load(jf)

        # Load reweight factors for specific generator
        nnlops_reweight = nnlops_reweight[generator]

        # Build linear splines for different njet bins
        spline_0jet = interp1d(
            nnlops_reweight["0jet"]["pt"], nnlops_reweight["0jet"]["weight"]
        )
        spline_1jet = interp1d(
            nnlops_reweight["1jet"]["pt"], nnlops_reweight["1jet"]["weight"]
        )
        spline_2jet = interp1d(
            nnlops_reweight["2jet"]["pt"], nnlops_reweight["2jet"]["weight"]
        )
        spline_ge3jet = interp1d(
            nnlops_reweight["3jet"]["pt"], nnlops_reweight["3jet"]["weight"]
        )

        # Load truth Higgs pt and njets (pt>30) from events
        higgs_pt = events.HTXS.Higgs_pt.to_numpy()
        njets30 = events.HTXS.njets30.to_numpy()

        # Extract scale factors from splines and mask for different jet bins
        # Define maximum pt values as interpolated splines only go up so far
        sf = (
            (njets30 == 0) * spline_0jet(np.minimum(higgs_pt, 125.0))
            + (njets30 == 1) * spline_1jet(np.minimum(higgs_pt, 625.0))
            + (njets30 == 2) * spline_2jet(np.minimum(higgs_pt, 800.0))
            + (njets30 >= 3) * spline_ge3jet(np.minimum(higgs_pt, 925.0))
        )
    else:
        logger.info(f"\n WARNING: You asked for NNLOPS reweighting SF for dataset with {dataset_name} but this does not appear like a ggF to single Higgs sample.")
        sf = np.ones(len(events))

    return sf


def NNLOPS(
    events, dataset_name, weights, is_correction=True, generator="mcatnlo", **kwargs
):
    """
    --- NNLOPS reweighting for ggH events to be applied to NLO Madgraph (and Powheg).
    Swap generator argument to 'powheg' if to be applied to powheg events
    Reweight event based on truth Higgs pt and number of jets, extracted from HTXS object
    Constructs njet-dependent linear splines based on input data, functions of Higgs pt
    Reweighting is applied always if correction is specified in runner JSON.
    Warning is thrown if ggh or glugluh is not in the name.
    """
    if is_correction:
        sf = calculate_NNLOPS_sf(events, dataset_name, generator)
        weights.add("NNLOPS", sf, None, None)
    else:
        raise RuntimeError(
            "NNLOPS reweighting is only a flat correction, not a systematic"
        )

    return weights


def AlphaS(photons, events, weights, dataset_name, **kwargs):
    """
    AlphaS weights variations are the last two of the PDF replicas, e.g.,
    https://github.com/cms-sw/cmssw/blob/d37d2797dffc978a78da2fafec3ba480071a0e67/PhysicsTools/NanoAOD/python/genWeightsTable_cfi.py#L10
    https://lhapdfsets.web.cern.ch/current/NNPDF31_nnlo_as_0118_mc_hessian_pdfas/NNPDF31_nnlo_as_0118_mc_hessian_pdfas.info
    """
    systematic = "AlphaS Weight"
    try:
        weights.add(
            name="AlphaS",
            weight=np.ones(len(events)),
            weightUp=events.LHEPdfWeight[:, -1],
            weightDown=events.LHEPdfWeight[:, -2],
        )
    except (AttributeError, IndexError):
        logger.warning(
            f"No LHEPdf Weights in dataset {dataset_name}, skip systematic: {systematic}"
        )
        return weights

    return weights


def PartonShower(photons, events, weights, dataset_name, **kwargs):
    """
    Parton Shower weights:
    https://github.com/cms-sw/cmssw/blob/caeae4110ddbada1cfdac195404b3c618584e8fb/PhysicsTools/NanoAOD/plugins/GenWeightsTableProducer.cc#L533-L534
    """
    systematic = "PartonShower weight"
    try:
        weights.add(
            name="PS_ISR",
            weight=np.ones(len(events)),
            weightUp=events.PSWeight[:, 0],
            weightDown=events.PSWeight[:, 2],
        )

        weights.add(
            name="PS_FSR",
            weight=np.ones(len(events)),
            weightUp=events.PSWeight[:, 1],
            weightDown=events.PSWeight[:, 3],
        )
    except (AttributeError, IndexError):
        logger.warning(
            f"No PS Weights in dataset {dataset_name}, skip systematic: {systematic}"
        )
        return weights

    return weights


def bTagShapeSF(events, weights, ShapeSF_name, is_correction=True, year="2017", **kwargs):
    avail_years = ["2016preVFP", "2016postVFP", "2017", "2018", "2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024"]

    if year == "2024":
        logger.warning("Current 2024 bTagShapeSF are not implemented, 2023PostBPix is used! These ntuples should not be used for a final physics result!")

    if year not in avail_years:
        print(f"\n WARNING: only scale corrections for the year strings {avail_years} are already implemented! \n Exiting. \n")
        exit()

    if (ShapeSF_name in ["particleNet_shape", "robustParticleTransformer_shape"]) and (year in ["2016preVFP", "2016postVFP", "2017", "2018"]):
        print(f"\n WARNING: The ShapeSF {ShapeSF_name} is not available for the year {year}. \n Exiting. \n")
        exit()

    ShapeSF_name_to_discriminant = {
        "deepJet_shape": "btagDeepFlav_B",
        "particleNet_shape": "btagPNetB",
        "robustParticleTransformer_shape": "btagRobustParTAK4B",
    }

    btag_systematics = [
        "lf",
        "hf",
        "cferr1",
        "cferr2",
        "lfstats1",
        "lfstats2",
        "hfstats1",
        "hfstats2",
        "jes",
    ]
    inputFilePath = "JSONs/bTagSF/"
    btag_correction_configs = {
        "2016preVFP": {
            "file": os.path.join(
                inputFilePath , "2016preVFP_UL/btagging.json.gz"
            ),
            "method": ShapeSF_name,
            "systs": btag_systematics,
        },
        "2016postVFP": {
            "file": os.path.join(
                inputFilePath , "2016postVFP_UL/btagging.json.gz"
            ),
            "method": ShapeSF_name,
            "systs": btag_systematics,
        },
        "2017": {
            "file": os.path.join(
                inputFilePath , "2017_UL/btagging.json.gz"
            ),
            "method": ShapeSF_name,
            "systs": btag_systematics,
        },
        "2018": {
            "file": os.path.join(
                inputFilePath , "2018_UL/btagging.json.gz"
            ),
            "method": ShapeSF_name,
            "systs": btag_systematics,
        },
        "2022preEE":{
            "file": os.path.join(
                inputFilePath , "2022_Summer22/btagging.json.gz"
            ),
            "method": ShapeSF_name,
            "systs": btag_systematics,
        },
        "2022postEE":{
            "file": os.path.join(
                inputFilePath , "2022_Summer22EE/btagging.json.gz"
            ),
            "method": ShapeSF_name,
            "systs": btag_systematics,
        },
        "2023preBPix":{
            "file": os.path.join(
                inputFilePath , "2023_Summer23/btagging.json.gz"
            ),
            "method": ShapeSF_name,
            "systs": btag_systematics,
        },
        "2023postBPix":{
            "file": os.path.join(
                inputFilePath , "2023_Summer23BPix/btagging.json.gz"
            ),
            "method": ShapeSF_name,
            "systs": btag_systematics,
        },
        # 2024 is still preliminary! Will be changed once official SFs are available
        "2024":{
            "file": os.path.join(
                inputFilePath , "2023_Summer23BPix/btagging.json.gz"
            ),
            "method": ShapeSF_name,
            "systs": btag_systematics,
        },
    }
    jsonpog_file = os.path.join(
        os.path.dirname(__file__), btag_correction_configs[year]["file"]
    )
    evaluator = correctionlib.CorrectionSet.from_file(jsonpog_file)[
        btag_correction_configs[year]["method"]
    ]

    dummy_sf = ak.ones_like(events["event"])

    # Removing jets with eta beyond 2.5 and has negative discriminant score. (No bining exist in input JSON file for such jets)
    relevant_jets = events["sel_jets"][
        (np.abs(events["sel_jets"].eta) < 2.5)
        & (events["sel_jets"][ShapeSF_name_to_discriminant[ShapeSF_name]] >= 0)
    ]

    # only calculate correction to nominal weight
    # we will evaluate the scale factors relative to all jets to be multiplied
    jet_pt = relevant_jets.pt
    jet_eta = np.abs(relevant_jets.eta)
    jet_hFlav = relevant_jets.hFlav
    jet_discriminant = relevant_jets[
        ShapeSF_name_to_discriminant[ShapeSF_name]
    ]

    # Convert the jets in one dimension array and store the orignal structure of the ak array in counts
    flat_jet_pt = ak.flatten(jet_pt)
    flat_jet_eta = ak.flatten(jet_eta)
    flat_jet_discriminant = ak.flatten(jet_discriminant)
    flat_jet_hFlav = ak.flatten(jet_hFlav)

    counts = ak.num(jet_hFlav)

    logger.info("Warning: you have to normalise b-tag weights afterwards so that they do not change the yield!")
    Weight_Name = ""
    if is_correction:
        Weight_Name = "bTagSF"
        _sf = []
        # Evluate the scale factore per jet and unflatten the scale fatores in original structure
        _sf = ak.unflatten(
            evaluator.evaluate(
                "central",
                flat_jet_hFlav,
                flat_jet_eta,
                flat_jet_pt,
                flat_jet_discriminant,
            ),
            counts
        )
        # Multiply the scale factore of all jets in a even
        sf = ak.prod(_sf, axis=1)

        sfs_up = [None for _ in btag_systematics]
        sfs_down = [None for _ in btag_systematics]

    else:
        Weight_Name = "bTagSF_sys"
        # only calculate correction to nominal weight
        # replace by accessing partial weight!
        _sf = []
        # Evluate the scale factore per jet and unflatten the scale fatores in original structure
        _sf_central = evaluator.evaluate(
            "central",
            flat_jet_hFlav,
            flat_jet_eta,
            flat_jet_pt,
            flat_jet_discriminant,
        )
        # Multiply the scale factore of all jets in a even

        sf = ak.values_astype(dummy_sf, np.float32)
        sf_central = ak.prod(
            ak.unflatten(_sf_central, counts),
            axis=1
        )

        variations = {}

        # Define a condiation based the jet flavour because the json file are defined for the 4(c),5(b),0(lf) flavour jets
        flavour_condition = np.logical_or(jet_hFlav < 4, jet_hFlav > 5)
        # Replace the flavour to 0 (lf) if the jet flavour is neither 4 nor 5
        jet_hFlav_JSONrestricted = ak.where(flavour_condition, 0, jet_hFlav)
        flat_jet_hFlav_JSONrestricted = ak.flatten(jet_hFlav_JSONrestricted)
        # We need a dmmy sf array set to one to multiply for flavour dependent systentic variation
        flat_dummy_sf = ak.ones_like(flat_jet_hFlav_JSONrestricted)

        for syst_name in btag_correction_configs[year]["systs"]:

            # we will append the scale factors relative to all jets to be multiplied
            _sfup = []
            _sfdown = []
            variations[syst_name] = {}

            if "cferr" in syst_name:
                # we to remember which jet is correspond to c(hadron flv 4) jets
                cjet_masks = flat_jet_hFlav_JSONrestricted == 4

                flat_jet_hFlavC_JSONrestricted = ak.where(flat_jet_hFlav_JSONrestricted != 4, 4 ,flat_jet_hFlav_JSONrestricted)
                _Csfup = evaluator.evaluate(
                    "up_" + syst_name,
                    flat_jet_hFlavC_JSONrestricted,
                    flat_jet_eta,
                    flat_jet_pt,
                    flat_jet_discriminant,
                )

                _Csfdown = evaluator.evaluate(
                    "down_" + syst_name,
                    flat_jet_hFlavC_JSONrestricted,
                    flat_jet_eta,
                    flat_jet_pt,
                    flat_jet_discriminant,
                )
                _Csfup = ak.where(
                    cjet_masks,
                    _Csfup,
                    flat_dummy_sf,
                )
                _Csfdown = ak.where(
                    cjet_masks,
                    _Csfdown,
                    flat_dummy_sf,
                )
                # Replace all the calculated sf with 1 when there is light jet or with flavour b otherwise keep the cerntral weight
                _sfcentral_Masked_notC = ak.where(
                    ~cjet_masks,
                    _sf_central,
                    flat_dummy_sf,
                )
                _sfup = ak.unflatten(np.multiply(_sfcentral_Masked_notC, _Csfup), counts)
                _sfdown = ak.unflatten(np.multiply(_sfcentral_Masked_notC, _Csfdown), counts)
            else:
                # We to remember which jet is correspond to c(hadron flv 4) jets
                cjet_masks = flat_jet_hFlav_JSONrestricted == 4

                flat_jet_hFlavNonC_JSONrestricted = ak.where(cjet_masks, 0, flat_jet_hFlav_JSONrestricted)

                _NonCsfup = evaluator.evaluate(
                    "up_" + syst_name,
                    flat_jet_hFlavNonC_JSONrestricted,
                    flat_jet_eta,
                    flat_jet_pt,
                    flat_jet_discriminant,
                )

                _NonCsfdown = evaluator.evaluate(
                    "down_" + syst_name,
                    flat_jet_hFlavNonC_JSONrestricted,
                    flat_jet_eta,
                    flat_jet_pt,
                    flat_jet_discriminant,
                )

                _NonCsfup = ak.where(
                    ~cjet_masks,
                    _NonCsfup,
                    flat_dummy_sf,
                )
                _NonCsfdown = ak.where(
                    ~cjet_masks,
                    _NonCsfdown,
                    flat_dummy_sf,
                )
                # Replace all the calculated sf with 1 when there is c jet otherwise keep the cerntral weight
                _sfcentral_Masked_C = ak.where(
                    cjet_masks,
                    _sf_central,
                    flat_dummy_sf,
                )
                _sfup = ak.unflatten(np.multiply(_sfcentral_Masked_C, _NonCsfup), counts)
                _sfdown = ak.unflatten(np.multiply(_sfcentral_Masked_C, _NonCsfdown), counts)

            sf_up = ak.prod(_sfup, axis=1)
            sf_down = ak.prod(_sfdown, axis=1)
            variations[syst_name]["up"] = sf_up
            variations[syst_name]["down"] = sf_down
        # coffea weights.add_multivariation() wants a list of arrays for the multiple up and down variations
        # we devide sf_central because cofea processor save the up and down vartion by multiplying the central weights
        sfs_up = [variations[syst_name]["up"] / sf_central for syst_name in btag_systematics]
        sfs_down = [variations[syst_name]["down"] / sf_central for syst_name in btag_systematics]

    weights.add_multivariation(
        name=Weight_Name,
        weight=sf,
        modifierNames=btag_systematics,
        weightsUp=sfs_up,
        weightsDown=sfs_down,
        shift=False,
    )

    return weights


def bTagFixedWP(events, jets, weights, dataset_name, mva_name, wp, bTagEffFileName, btagEffDatasetName=None, is_correction=True, year="2017", is_Run2_v15=False, **kwargs):

    avail_years = ["2016preVFP", "2016postVFP", "2017", "2018", "2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024"]
    if year not in avail_years:
        logger.error(f"\n Only fixed WP Scale Factors for the year strings {avail_years} are already implemented! \n Exiting. \n")
        exit()

    avail_modes = ["L", "M", "T", "XT", "XXT"]
    if wp not in avail_modes:
        logger.error(f"\n Only fixed WP Scale Factors for the mode strings {avail_modes} are already implemented! \n Exiting. \n")
        exit()

    eff_dataset_name = btagEffDatasetName if btagEffDatasetName is not None else dataset_name

    inputFilePath = "JSONs/"
    if bTagEffFileName is None:
        bTagEffFileName = "midRun3"

    Run2_btag_json = ""
    if is_Run2_v15:
        Run2_btag_json = "_v15"

    btageff_correction_configs = {
        "2016preVFP":{
            "file": os.path.join(
                inputFilePath , "bTagEff/2016preVFP_UL/" + bTagEffFileName + Run2_btag_json + ".json.gz"
            )
        },
        "2016postVFP":{
            "file": os.path.join(
                inputFilePath , "bTagEff/2016postVFP_UL/" + bTagEffFileName + Run2_btag_json + ".json.gz"
            )
        },
        "2017":{
            "file": os.path.join(
                inputFilePath , "bTagEff/2017_UL/" + bTagEffFileName + Run2_btag_json + ".json.gz"
            )
        },
        "2018":{
            "file": os.path.join(
                inputFilePath , "bTagEff/2018_UL/" + bTagEffFileName + Run2_btag_json + ".json.gz"
            )
        },
        "2022preEE":{
            "file": os.path.join(
                inputFilePath , "bTagEff/2022_Summer22/" + bTagEffFileName + Run2_btag_json + ".json.gz"
            )
        },
        "2022postEE":{
            "file": os.path.join(
                inputFilePath , "bTagEff/2022_Summer22EE/" + bTagEffFileName + Run2_btag_json + ".json.gz"
            )
        },
        "2023preBPix":{
            "file": os.path.join(
                inputFilePath , "bTagEff/2023_Summer23/" + bTagEffFileName + Run2_btag_json + ".json.gz"
            )
        },
        "2023postBPix":{
            "file": os.path.join(
                inputFilePath , "bTagEff/2023_Summer23BPix/" + bTagEffFileName + Run2_btag_json + ".json.gz"
            )
        },
        "2024":{
            "file": os.path.join(
                inputFilePath , "bTagEff/2024_Summer24/" + bTagEffFileName + Run2_btag_json + ".json.gz"
            )
        }
    }

    btageff_jsonpog_file = os.path.join(
        os.path.dirname(__file__), btageff_correction_configs[year]["file"]
    )

    try:
        btageff_clib = correctionlib.CorrectionSet.from_file(btageff_jsonpog_file)
        btageff_dict = ast.literal_eval(btageff_clib._data)

        avail_procs = [current_proc["key"] for current_proc in btageff_dict["corrections"][0]["data"]["content"]]
    except OSError:
        logger.error("\n Error when reading the dataset name from the correction lib. \n")
        exit()

    if (eff_dataset_name in avail_procs):

        mva_name_to_btag_wp_name = {
            "particleNet": "particleNet_wp_values",
            "deepJet": "deepJet_wp_values",
            "robustParticleTransformer": "robustParticleTransformer_wp_values",
            "UParTAK4": "UParTAK4_wp_values"
        }

        mva_name_to_discriminator = {
            "particleNet": "btagPNetB",
            "deepJet": "btagDeepFlavB",
            "robustParticleTransformer": "btagRobustParTAK4B",
            "UParTAK4": "btagUParTAK4B"
        }

        mva_name_to_btag_sf_name = {
            "light": {
                "particleNet": "particleNet_light",
                "deepJet": "deepJet_light",
                "robustParticleTransformer": "robustParticleTransformer_light",
                "UParTAK4": "UParTAK4_light"
            },
            "comb": {
                "particleNet": "particleNet_comb",
                "deepJet": "deepJet_comb",
                "robustParticleTransformer": "robustParticleTransformer_comb",
                "UParTAK4": "UParTAK4_comb"
            }
        }

        btag_systematics = [
            "correlated",
            "uncorrelated"
        ]
        btag_systematics_modifiers = [
            "correlated",
            f"{year}",
        ]

        btag_correction_configs = {
            "2016preVFP": {
                "file": os.path.join(
                    inputFilePath , "bTagSF/2016preVFP_UL/btagging" + Run2_btag_json + ".json.gz"
                ),
                "wp": mva_name_to_btag_wp_name[mva_name],
                "sf_light": mva_name_to_btag_sf_name["light"][mva_name],
                "sf_comb": mva_name_to_btag_sf_name["comb"][mva_name]
            },
            "2016postVFP": {
                "file": os.path.join(
                    inputFilePath , "bTagSF/2016postVFP_UL/btagging" + Run2_btag_json + ".json.gz"
                ),
                "wp": mva_name_to_btag_wp_name[mva_name],
                "sf_light": mva_name_to_btag_sf_name["light"][mva_name],
                "sf_comb": mva_name_to_btag_sf_name["comb"][mva_name]
            },
            "2017": {
                "file": os.path.join(
                    inputFilePath , "bTagSF/2017_UL/btagging" + Run2_btag_json + ".json.gz"
                ),
                "wp": mva_name_to_btag_wp_name[mva_name],
                "sf_light": mva_name_to_btag_sf_name["light"][mva_name],
                "sf_comb": mva_name_to_btag_sf_name["comb"][mva_name]
            },
            "2018": {
                "file": os.path.join(
                    inputFilePath , "bTagSF/2018_UL/btagging" + Run2_btag_json + ".json.gz"
                ),
                "wp": mva_name_to_btag_wp_name[mva_name],
                "sf_light": mva_name_to_btag_sf_name["light"][mva_name],
                "sf_comb": mva_name_to_btag_sf_name["comb"][mva_name]
            },
            "2022preEE":{
                "file": os.path.join(
                    inputFilePath , "bTagSF/2022_Summer22/btagging" + Run2_btag_json + ".json.gz"
                ),
                "wp": mva_name_to_btag_wp_name[mva_name],
                "sf_light": mva_name_to_btag_sf_name["light"][mva_name],
                "sf_comb": mva_name_to_btag_sf_name["comb"][mva_name]
            },
            "2022postEE":{
                "file": os.path.join(
                    inputFilePath , "bTagSF/2022_Summer22EE/btagging" + Run2_btag_json + ".json.gz"
                ),
                "wp": mva_name_to_btag_wp_name[mva_name],
                "sf_light": mva_name_to_btag_sf_name["light"][mva_name],
                "sf_comb": mva_name_to_btag_sf_name["comb"][mva_name]
            },
            "2023preBPix":{
                "file": os.path.join(
                    inputFilePath , "bTagSF/2023_Summer23/btagging" + Run2_btag_json + ".json.gz"
                ),
                "wp": mva_name_to_btag_wp_name[mva_name],
                "sf_light": mva_name_to_btag_sf_name["light"][mva_name],
                "sf_comb": mva_name_to_btag_sf_name["comb"][mva_name]
            },
            "2023postBPix":{
                "file": os.path.join(
                    inputFilePath , "bTagSF/2023_Summer23BPix/btagging" + Run2_btag_json + ".json.gz"
                ),
                "wp": mva_name_to_btag_wp_name[mva_name],
                "sf_light": mva_name_to_btag_sf_name["light"][mva_name],
                "sf_comb": mva_name_to_btag_sf_name["comb"][mva_name]
            },
            "2024":{
                "file": os.path.join(
                    inputFilePath , "bTagSF/2024_Summer24/btagging" + Run2_btag_json + ".json.gz"
                ),
                "wp": mva_name_to_btag_wp_name[mva_name],
                "sf_light": mva_name_to_btag_sf_name["light"][mva_name],
                "sf_comb": mva_name_to_btag_sf_name["comb"][mva_name]
            },
        }

        btagsf_jsonpog_file = os.path.join(
            os.path.dirname(__file__), btag_correction_configs[year]["file"]
        )

        # Import evaluators
        light_evaluator = correctionlib.CorrectionSet.from_file(btagsf_jsonpog_file)[btag_correction_configs[year]["sf_light"]]
        heavy_btagsf_jsonpog_file = btagsf_jsonpog_file
        if is_Run2_v15 and ("2016preVFP" in year or "2016postVFP" in year or "2017" in year or "2018" in year):
            heavy_btagsf_jsonpog_file = os.path.join(
                os.path.dirname(__file__), inputFilePath, "bTagSF/2024_Summer24/btagging.json.gz"
            )
            logger.warning("Run2 v15 missing comb SF, fallback to 2024_Summer24 comb SF.")
        heavy_evaluator = correctionlib.CorrectionSet.from_file(heavy_btagsf_jsonpog_file)[btag_correction_configs[year]["sf_comb"]]
        btageff_evaluator = correctionlib.CorrectionSet.from_file(btageff_jsonpog_file)["btagging_efficiencies"]

        chosenWP = correctionlib.CorrectionSet.from_file(btagsf_jsonpog_file)[btag_correction_configs[year]["wp"]].evaluate(wp)

        # Removing jets with eta beyond 2.5 and has negative discriminant score. (No bining exist in input JSON file for such jets)
        relevant_jets = events["sel_jets"]

        if jets is not None:
            logger.info("using the dedicated jets input for b-tagging SF corrections and systematics")
            relevant_jets = jets

        # Era-dependent gen-jet pT threshold:
        # - Run2 (2016–2018): pT > 30 GeV (Run2 recommendation)
        # - Run3 (2022–2025): pT > 20 GeV (Run3 recommendation https://btv-wiki.docs.cern.ch/ScaleFactors/#important-notes)
        if year in ["2016preVFP", "2016postVFP", "2017", "2018"]:
            gen_jet_min_pt = 30
        else:
            gen_jet_min_pt = 20

        tagged_jets = relevant_jets[
            ((relevant_jets.pt) > gen_jet_min_pt)
            & (np.abs(relevant_jets.eta) < 2.5)
            & (relevant_jets[mva_name_to_discriminator[mva_name]] >= chosenWP)
        ]

        untagged_jets = relevant_jets[
            ((relevant_jets.pt) > gen_jet_min_pt)
            & (np.abs(relevant_jets.eta) < 2.5)
            & (relevant_jets[mva_name_to_discriminator[mva_name]] < chosenWP)
        ]

        # Split jetcollection in heavy (hFlav == 4, 5) and in light flavor (hFlav == 0)
        heavy_tagged_jets = tagged_jets[(tagged_jets.hFlav >= 4)]
        light_tagged_jets = tagged_jets[(tagged_jets.hFlav == 0)]

        heavy_untagged_jets = untagged_jets[(untagged_jets.hFlav >= 4)]
        light_untagged_jets = untagged_jets[(untagged_jets.hFlav == 0)]

        # only calculate correction to nominal weight
        # we will evaluate the scale factors relative to all jets to be multiplied

        ###################
        ### TAGGED JETS ###
        ###################

        # light tagged jets
        light_tagged_jet_pt = light_tagged_jets.pt
        light_tagged_jet_eta = np.abs(light_tagged_jets.eta)
        light_tagged_jet_hFlav = light_tagged_jets.hFlav

        # Convert the jets in one dimension array and store the orignal structure of the ak array in counts
        flat_light_tagged_jet_pt = ak.flatten(light_tagged_jet_pt)
        flat_light_tagged_jet_eta = ak.flatten(light_tagged_jet_eta)
        flat_light_tagged_jet_hFlav = ak.flatten(light_tagged_jet_hFlav)

        light_tagged_counts = ak.num(light_tagged_jet_hFlav)

        # heavy tagged jets
        heavy_tagged_jet_pt = heavy_tagged_jets.pt
        heavy_tagged_jet_eta = np.abs(heavy_tagged_jets.eta)
        heavy_tagged_jet_hFlav = heavy_tagged_jets.hFlav

        # Convert the jets in one dimension array and store the orignal structure of the ak array in counts
        flat_heavy_tagged_jet_pt = ak.flatten(heavy_tagged_jet_pt)
        flat_heavy_tagged_jet_eta = ak.flatten(heavy_tagged_jet_eta)
        flat_heavy_tagged_jet_hFlav = ak.flatten(heavy_tagged_jet_hFlav)

        heavy_tagged_counts = ak.num(heavy_tagged_jet_hFlav)

        #####################
        ### UNTAGGED JETS ###
        #####################

        # light untagged jets
        light_untagged_jet_pt = light_untagged_jets.pt
        light_untagged_jet_eta = np.abs(light_untagged_jets.eta)
        light_untagged_jet_hFlav = light_untagged_jets.hFlav

        # Convert the jets in one dimension array and store the orignal structure of the ak array in counts
        flat_light_untagged_jet_pt = ak.flatten(light_untagged_jet_pt)
        flat_light_untagged_jet_eta = ak.flatten(light_untagged_jet_eta)
        flat_light_untagged_jet_hFlav = ak.flatten(light_untagged_jet_hFlav)

        light_untagged_counts = ak.num(light_untagged_jet_hFlav)

        # heavy untagged jets
        heavy_untagged_jet_pt = heavy_untagged_jets.pt
        heavy_untagged_jet_eta = np.abs(heavy_untagged_jets.eta)
        heavy_untagged_jet_hFlav = heavy_untagged_jets.hFlav

        # Convert the jets in one dimension array and store the orignal structure of the ak array in counts
        flat_heavy_untagged_jet_pt = ak.flatten(heavy_untagged_jet_pt)
        flat_heavy_untagged_jet_eta = ak.flatten(heavy_untagged_jet_eta)
        flat_heavy_untagged_jet_hFlav = ak.flatten(heavy_untagged_jet_hFlav)

        heavy_untagged_counts = ak.num(heavy_untagged_jet_hFlav)

        # only calculate correction to nominal weight
        # replace by accessing partial weight!
        _light_tagged_central_sf = []
        _heavy_tagged_central_sf = []

        _light_untagged_central_sf = []
        _light_untagged_btagEff = []
        _heavy_untagged_central_sf = []
        _heavy_untagged_btagEff = []

        ###################
        ### TAGGED JETS ###
        ###################

        # Evluate the scale factore per jet and unflatten the scale fatores in original structure
        _light_tagged_central_sf = ak.unflatten(
            light_evaluator.evaluate(
                "central",
                wp,
                flat_light_tagged_jet_hFlav,
                flat_light_tagged_jet_eta,
                flat_light_tagged_jet_pt
            ),
            light_tagged_counts
        )

        flat_heavy_tagged_central_sf = heavy_evaluator.evaluate(
            "central",
            wp,
            flat_heavy_tagged_jet_hFlav,
            flat_heavy_tagged_jet_eta,
            flat_heavy_tagged_jet_pt
        )
        _heavy_tagged_central_sf = ak.unflatten(
            flat_heavy_tagged_central_sf,
            heavy_tagged_counts
        )

        #####################
        ### UNTAGGED JETS ###
        #####################

        # Evluate the scale factore per jet and unflatten the scale fatores in original structure
        _light_untagged_central_sf = ak.unflatten(
            light_evaluator.evaluate(
                "central",
                wp,
                flat_light_untagged_jet_hFlav,
                flat_light_untagged_jet_eta,
                flat_light_untagged_jet_pt
            ),
            light_untagged_counts
        )

        _light_untagged_btagEff = ak.unflatten(
            btageff_evaluator.evaluate(
                eff_dataset_name,
                wp,
                flat_light_untagged_jet_hFlav,
                flat_light_untagged_jet_pt
            ),
            light_untagged_counts
        )

        flat_heavy_untagged_central_sf = heavy_evaluator.evaluate(
            "central",
            wp,
            flat_heavy_untagged_jet_hFlav,
            flat_heavy_untagged_jet_eta,
            flat_heavy_untagged_jet_pt
        )
        _heavy_untagged_central_sf = ak.unflatten(
            flat_heavy_untagged_central_sf,
            heavy_untagged_counts
        )

        _heavy_untagged_btagEff = ak.unflatten(
            btageff_evaluator.evaluate(
                eff_dataset_name,
                wp,
                flat_heavy_untagged_jet_hFlav,
                flat_heavy_untagged_jet_pt
            ),
            heavy_untagged_counts
        )

        # Tagged jets
        light_tagged_central_prod = ak.prod(_light_tagged_central_sf, axis=1)  # Product over the tagged jets
        heavy_tagged_central_prod = ak.prod(_heavy_tagged_central_sf, axis=1)
        tagged_central = heavy_tagged_central_prod * light_tagged_central_prod

        # Untagged jets
        untagged_heavy_numerator_central_prod = _heavy_untagged_central_sf * _heavy_untagged_btagEff
        untagged_heavy_central = ak.prod((1 - untagged_heavy_numerator_central_prod) / (1 - _heavy_untagged_btagEff), axis=1)

        untagged_light_numerator_central_prod = _light_untagged_central_sf * _light_untagged_btagEff
        untagged_light_central = ak.prod((1 - untagged_light_numerator_central_prod) / (1 - _light_untagged_btagEff), axis=1)
        untagged_central = untagged_heavy_central * untagged_light_central

        w_btag_light_central = light_tagged_central_prod * untagged_light_central
        w_btag_heavy_central = heavy_tagged_central_prod * untagged_heavy_central
        w_btag_central = tagged_central * untagged_central

        if is_correction:
            w_btag = w_btag_central

            w_btag_up = [None for _ in btag_systematics]
            w_btag_down = [None for _ in btag_systematics]

            weights.add_multivariation(
                name="bTagFixedWP",
                weight=w_btag,
                modifierNames=btag_systematics,
                weightsUp=w_btag_up,
                weightsDown=w_btag_down,
                shift=False,
            )

            return weights

        else:
            light_variations = {}
            heavy_variations = {}

            for syst_name in btag_systematics:
                light_variations[syst_name] = {}
                heavy_variations[syst_name] = {}

                _light_tagged_up_sf = ak.unflatten(
                    light_evaluator.evaluate(
                        "up_" + syst_name,
                        wp,
                        flat_light_tagged_jet_hFlav,
                        flat_light_tagged_jet_eta,
                        flat_light_tagged_jet_pt,
                    ),
                    light_tagged_counts,
                )
                _light_tagged_down_sf = ak.unflatten(
                    light_evaluator.evaluate(
                        "down_" + syst_name,
                        wp,
                        flat_light_tagged_jet_hFlav,
                        flat_light_tagged_jet_eta,
                        flat_light_tagged_jet_pt,
                    ),
                    light_tagged_counts,
                )
                _heavy_tagged_up_sf = ak.unflatten(
                    heavy_evaluator.evaluate(
                        "up_" + syst_name,
                        wp,
                        flat_heavy_tagged_jet_hFlav,
                        flat_heavy_tagged_jet_eta,
                        flat_heavy_tagged_jet_pt,
                    ),
                    heavy_tagged_counts,
                )
                _heavy_tagged_down_sf = ak.unflatten(
                    heavy_evaluator.evaluate(
                        "down_" + syst_name,
                        wp,
                        flat_heavy_tagged_jet_hFlav,
                        flat_heavy_tagged_jet_eta,
                        flat_heavy_tagged_jet_pt,
                    ),
                    heavy_tagged_counts,
                )

                _light_untagged_up_sf = ak.unflatten(
                    light_evaluator.evaluate(
                        "up_" + syst_name,
                        wp,
                        flat_light_untagged_jet_hFlav,
                        flat_light_untagged_jet_eta,
                        flat_light_untagged_jet_pt,
                    ),
                    light_untagged_counts,
                )
                _light_untagged_down_sf = ak.unflatten(
                    light_evaluator.evaluate(
                        "down_" + syst_name,
                        wp,
                        flat_light_untagged_jet_hFlav,
                        flat_light_untagged_jet_eta,
                        flat_light_untagged_jet_pt,
                    ),
                    light_untagged_counts,
                )
                _heavy_untagged_up_sf = ak.unflatten(
                    heavy_evaluator.evaluate(
                        "up_" + syst_name,
                        wp,
                        flat_heavy_untagged_jet_hFlav,
                        flat_heavy_untagged_jet_eta,
                        flat_heavy_untagged_jet_pt,
                    ),
                    heavy_untagged_counts,
                )
                _heavy_untagged_down_sf = ak.unflatten(
                    heavy_evaluator.evaluate(
                        "down_" + syst_name,
                        wp,
                        flat_heavy_untagged_jet_hFlav,
                        flat_heavy_untagged_jet_eta,
                        flat_heavy_untagged_jet_pt,
                    ),
                    heavy_untagged_counts,
                )

                light_tagged_up_prod = ak.prod(_light_tagged_up_sf, axis=1)
                light_tagged_down_prod = ak.prod(_light_tagged_down_sf, axis=1)
                heavy_tagged_up_prod = ak.prod(_heavy_tagged_up_sf, axis=1)
                heavy_tagged_down_prod = ak.prod(_heavy_tagged_down_sf, axis=1)

                untagged_heavy_numerator_up_prod = _heavy_untagged_up_sf * _heavy_untagged_btagEff
                untagged_heavy_up = ak.prod((1 - untagged_heavy_numerator_up_prod) / (1 - _heavy_untagged_btagEff), axis=1)

                untagged_heavy_numerator_down_prod = _heavy_untagged_down_sf * _heavy_untagged_btagEff
                untagged_heavy_down = ak.prod((1 - untagged_heavy_numerator_down_prod) / (1 - _heavy_untagged_btagEff), axis=1)

                untagged_light_numerator_up_prod = _light_untagged_up_sf * _light_untagged_btagEff
                untagged_light_up = ak.prod((1 - untagged_light_numerator_up_prod) / (1 - _light_untagged_btagEff), axis=1)

                untagged_light_numerator_down_prod = _light_untagged_down_sf * _light_untagged_btagEff
                untagged_light_down = ak.prod((1 - untagged_light_numerator_down_prod) / (1 - _light_untagged_btagEff), axis=1)

                w_btag_light_up = light_tagged_up_prod * untagged_light_up
                w_btag_light_down = light_tagged_down_prod * untagged_light_down
                w_btag_heavy_up = heavy_tagged_up_prod * untagged_heavy_up
                w_btag_heavy_down = heavy_tagged_down_prod * untagged_heavy_down

                light_variations[syst_name]["up"] = w_btag_light_up
                light_variations[syst_name]["down"] = w_btag_light_down
                heavy_variations[syst_name]["up"] = w_btag_heavy_up
                heavy_variations[syst_name]["down"] = w_btag_heavy_down

            w_btag = ak.values_astype(ak.ones_like(events.event), np.float32)
            light_up = [light_variations[syst_name]["up"] / w_btag_light_central for syst_name in btag_systematics]
            light_down = [light_variations[syst_name]["down"] / w_btag_light_central for syst_name in btag_systematics]
            heavy_up = [heavy_variations[syst_name]["up"] / w_btag_heavy_central for syst_name in btag_systematics]
            heavy_down = [heavy_variations[syst_name]["down"] / w_btag_heavy_central for syst_name in btag_systematics]

            weights.add_multivariation(
                name="btagSFlight",
                weight=w_btag,
                modifierNames=btag_systematics_modifiers,
                weightsUp=light_up,
                weightsDown=light_down,
                shift=False,
            )
            weights.add_multivariation(
                name="btagSFbc",
                weight=w_btag,
                modifierNames=btag_systematics_modifiers,
                weightsUp=heavy_up,
                weightsDown=heavy_down,
                shift=False,
            )
            return weights

    else:
        logger.error(f"\n You specified the Btagging SF for dataset with {dataset_name}. First compute the Btagging efficiency correctionlib for your analysis before proceeding. \n")
        exit()


def bTagFixedWP_bbgg(events, jets, weights, dataset_name, mva_name, wp, bTagEffFileName, btagEffDatasetName=None, is_correction=True, year="2017", is_Run2_v15=False, **kwargs):

    avail_years = ["2016preVFP", "2016postVFP", "2017", "2018", "2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024"]
    if year not in avail_years:
        logger.error(f"\n Only fixed WP Scale Factors for the year strings {avail_years} are already implemented! \n Exiting. \n")
        exit()

    avail_modes = ["L", "M", "T", "XT", "XXT"]
    if wp not in avail_modes:
        logger.error(f"\n Only fixed WP Scale Factors for the mode strings {avail_modes} are already implemented! \n Exiting. \n")
        exit()

    eff_dataset_name = btagEffDatasetName if btagEffDatasetName is not None else dataset_name

    inputFilePath = "JSONs/"
    if bTagEffFileName is None:
        bTagEffFileName = "midRun3"

    Run2_btag_json = ""
    if is_Run2_v15:
        Run2_btag_json = "_v15"

    btageff_correction_configs = {
        "2016preVFP":{
            "file": os.path.join(
                inputFilePath , "bTagEff/2016preVFP_UL/" + bTagEffFileName + Run2_btag_json + ".json.gz"
            )
        },
        "2016postVFP":{
            "file": os.path.join(
                inputFilePath , "bTagEff/2016postVFP_UL/" + bTagEffFileName + Run2_btag_json + ".json.gz"
            )
        },
        "2017":{
            "file": os.path.join(
                inputFilePath , "bTagEff/2017_UL/" + bTagEffFileName + Run2_btag_json + ".json.gz"
            )
        },
        "2018":{
            "file": os.path.join(
                inputFilePath , "bTagEff/2018_UL/" + bTagEffFileName + Run2_btag_json + ".json.gz"
            )
        },
        "2022preEE":{
            "file": os.path.join(
                inputFilePath , "bTagEff/2022_Summer22/" + bTagEffFileName + Run2_btag_json + ".json.gz"
            )
        },
        "2022postEE":{
            "file": os.path.join(
                inputFilePath , "bTagEff/2022_Summer22EE/" + bTagEffFileName + Run2_btag_json + ".json.gz"
            )
        },
        "2023preBPix":{
            "file": os.path.join(
                inputFilePath , "bTagEff/2023_Summer23/" + bTagEffFileName + Run2_btag_json + ".json.gz"
            )
        },
        "2023postBPix":{
            "file": os.path.join(
                inputFilePath , "bTagEff/2023_Summer23BPix/" + bTagEffFileName + Run2_btag_json + ".json.gz"
            )
        },
        "2024":{
            "file": os.path.join(
                inputFilePath , "bTagEff/2024_Summer24/" + bTagEffFileName + Run2_btag_json + ".json"
            )
        }
    }

    btageff_jsonpog_file = os.path.join(
        os.path.dirname(__file__), btageff_correction_configs[year]["file"]
    )

    try:
        btageff_clib = correctionlib.CorrectionSet.from_file(btageff_jsonpog_file)
        btageff_dict = ast.literal_eval(btageff_clib._data)

        avail_procs = [current_proc["key"] for current_proc in btageff_dict["corrections"][0]["data"]["content"]]
    except OSError:
        logger.error("\n Error when reading the dataset name from the correction lib. \n")
        exit()

    if (eff_dataset_name in avail_procs):

        mva_name_to_btag_wp_name = {
            "particleNet": "particleNet_wp_values",
            "deepJet": "deepJet_wp_values",
            "robustParticleTransformer": "robustParticleTransformer_wp_values",
            "UParTAK4": "UParTAK4_wp_values"
        }

        mva_name_to_discriminator = {
            "particleNet": "btagPNetB",
            "deepJet": "btagDeepFlavB",
            "robustParticleTransformer": "btagRobustParTAK4B",
            "UParTAK4": "btagUParTAK4B"
        }

        mva_name_to_btag_sf_name = {
            "light": {
                "particleNet": "particleNet_light",
                "deepJet": "deepJet_light",
                "robustParticleTransformer": "robustParticleTransformer_light",
                "UParTAK4": "UParTAK4_light"
            },
            "comb": {
                "particleNet": "particleNet_comb",
                "deepJet": "deepJet_comb",
                "robustParticleTransformer": "robustParticleTransformer_comb",
                "UParTAK4": "UParTAK4_comb"
            }
        }

        btag_systematics = [
            "correlated",
            "uncorrelated"
        ]
        btag_systematics_modifiers = [
            "correlated",
            f"{year}",
        ]

        btag_correction_configs = {
            "2016preVFP": {
                "file": os.path.join(
                    inputFilePath , "bTagSF/2016preVFP_UL/btagging" + Run2_btag_json + ".json.gz"
                ),
                "wp": mva_name_to_btag_wp_name[mva_name],
                "sf_light": mva_name_to_btag_sf_name["light"][mva_name],
                "sf_comb": mva_name_to_btag_sf_name["comb"][mva_name]
            },
            "2016postVFP": {
                "file": os.path.join(
                    inputFilePath , "bTagSF/2016postVFP_UL/btagging" + Run2_btag_json + ".json.gz"
                ),
                "wp": mva_name_to_btag_wp_name[mva_name],
                "sf_light": mva_name_to_btag_sf_name["light"][mva_name],
                "sf_comb": mva_name_to_btag_sf_name["comb"][mva_name]
            },
            "2017": {
                "file": os.path.join(
                    inputFilePath , "bTagSF/2017_UL/btagging" + Run2_btag_json + ".json.gz"
                ),
                "wp": mva_name_to_btag_wp_name[mva_name],
                "sf_light": mva_name_to_btag_sf_name["light"][mva_name],
                "sf_comb": mva_name_to_btag_sf_name["comb"][mva_name]
            },
            "2018": {
                "file": os.path.join(
                    inputFilePath , "bTagSF/2018_UL/btagging" + Run2_btag_json + ".json.gz"
                ),
                "wp": mva_name_to_btag_wp_name[mva_name],
                "sf_light": mva_name_to_btag_sf_name["light"][mva_name],
                "sf_comb": mva_name_to_btag_sf_name["comb"][mva_name]
            },
            "2022preEE":{
                "file": os.path.join(
                    inputFilePath , "bTagSF/2022_Summer22/btagging" + Run2_btag_json + ".json.gz"
                ),
                "wp": mva_name_to_btag_wp_name[mva_name],
                "sf_light": mva_name_to_btag_sf_name["light"][mva_name],
                "sf_comb": mva_name_to_btag_sf_name["comb"][mva_name]
            },
            "2022postEE":{
                "file": os.path.join(
                    inputFilePath , "bTagSF/2022_Summer22EE/btagging" + Run2_btag_json + ".json.gz"
                ),
                "wp": mva_name_to_btag_wp_name[mva_name],
                "sf_light": mva_name_to_btag_sf_name["light"][mva_name],
                "sf_comb": mva_name_to_btag_sf_name["comb"][mva_name]
            },
            "2023preBPix":{
                "file": os.path.join(
                    inputFilePath , "bTagSF/2023_Summer23/btagging" + Run2_btag_json + ".json.gz"
                ),
                "wp": mva_name_to_btag_wp_name[mva_name],
                "sf_light": mva_name_to_btag_sf_name["light"][mva_name],
                "sf_comb": mva_name_to_btag_sf_name["comb"][mva_name]
            },
            "2023postBPix":{
                "file": os.path.join(
                    inputFilePath , "bTagSF/2023_Summer23BPix/btagging" + Run2_btag_json + ".json.gz"
                ),
                "wp": mva_name_to_btag_wp_name[mva_name],
                "sf_light": mva_name_to_btag_sf_name["light"][mva_name],
                "sf_comb": mva_name_to_btag_sf_name["comb"][mva_name]
            },
            "2024":{
                "file": os.path.join(
                    inputFilePath , "bTagSF/2024_Summer24/btagging" + Run2_btag_json + ".json.gz"
                ),
                "wp": mva_name_to_btag_wp_name[mva_name],
                "sf_light": mva_name_to_btag_sf_name["light"][mva_name],
                "sf_comb": mva_name_to_btag_sf_name["comb"][mva_name]
            },
        }

        btagsf_jsonpog_file = os.path.join(
            os.path.dirname(__file__), btag_correction_configs[year]["file"]
        )

        # Import evaluators
        light_evaluator = correctionlib.CorrectionSet.from_file(btagsf_jsonpog_file)[btag_correction_configs[year]["sf_light"]]
        heavy_btagsf_jsonpog_file = btagsf_jsonpog_file
        if is_Run2_v15 and ("2016preVFP" in year or "2016postVFP" in year or "2017" in year or "2018" in year):
            heavy_btagsf_jsonpog_file = os.path.join(
                os.path.dirname(__file__), inputFilePath, "bTagSF/2024_Summer24/btagging.json.gz"
            )
            logger.warning("Run2 v15 missing comb SF, fallback to 2024_Summer24 comb SF.")
        heavy_evaluator = correctionlib.CorrectionSet.from_file(heavy_btagsf_jsonpog_file)[btag_correction_configs[year]["sf_comb"]]
        btageff_evaluator = correctionlib.CorrectionSet.from_file(btageff_jsonpog_file)["btagging_efficiencies"]

        chosenWP = correctionlib.CorrectionSet.from_file(btagsf_jsonpog_file)[btag_correction_configs[year]["wp"]].evaluate(wp)

        # Removing jets with eta beyond 2.5 and has negative discriminant score. (No bining exist in input JSON file for such jets)
        # relevant_jets = events["sel_jets"]

        if jets is not None:
            logger.info("using the dedicated jets input for b-tagging SF corrections and systematics")
            relevant_jets = jets

        # Era-dependent gen-jet pT threshold:
        # - Run2 (2016–2018): pT > 30 GeV (Run2 recommendation)
        # - Run3 (2022–2025): pT > 20 GeV (Run3 recommendation https://btv-wiki.docs.cern.ch/ScaleFactors/#important-notes)
        if year in ["2016preVFP", "2016postVFP", "2017", "2018"]:
            gen_jet_min_pt = 30
        else:
            gen_jet_min_pt = 20

        tagged_jets = relevant_jets[
            ((relevant_jets.pt) > gen_jet_min_pt)
            & (np.abs(relevant_jets.eta) < 2.5)
            & (relevant_jets[mva_name_to_discriminator[mva_name]] >= chosenWP)
        ]

        untagged_jets = relevant_jets[
            ((relevant_jets.pt) > gen_jet_min_pt)
            & (np.abs(relevant_jets.eta) < 2.5)
            & (relevant_jets[mva_name_to_discriminator[mva_name]] < chosenWP)
        ]

        # Split jetcollection in heavy (hadronFlavour == 4, 5) and in light flavor (hadronFlavour == 0)
        heavy_tagged_jets = tagged_jets[(tagged_jets.hadronFlavour >= 4)]
        light_tagged_jets = tagged_jets[(tagged_jets.hadronFlavour == 0)]

        heavy_untagged_jets = untagged_jets[(untagged_jets.hadronFlavour >= 4)]
        light_untagged_jets = untagged_jets[(untagged_jets.hadronFlavour == 0)]

        # only calculate correction to nominal weight
        # we will evaluate the scale factors relative to all jets to be multiplied

        ###################
        ### TAGGED JETS ###
        ###################

        # light tagged jets
        light_tagged_jet_pt = light_tagged_jets.pt
        light_tagged_jet_eta = np.abs(light_tagged_jets.eta)
        light_tagged_jet_hFlav = light_tagged_jets.hadronFlavour

        # Convert the jets in one dimension array and store the orignal structure of the ak array in counts
        flat_light_tagged_jet_pt = ak.flatten(light_tagged_jet_pt)
        flat_light_tagged_jet_eta = ak.flatten(light_tagged_jet_eta)
        flat_light_tagged_jet_hFlav = ak.flatten(light_tagged_jet_hFlav)

        light_tagged_counts = ak.num(light_tagged_jet_hFlav)

        # heavy tagged jets
        heavy_tagged_jet_pt = heavy_tagged_jets.pt
        heavy_tagged_jet_eta = np.abs(heavy_tagged_jets.eta)
        heavy_tagged_jet_hFlav = heavy_tagged_jets.hadronFlavour

        # Convert the jets in one dimension array and store the orignal structure of the ak array in counts
        flat_heavy_tagged_jet_pt = ak.flatten(heavy_tagged_jet_pt)
        flat_heavy_tagged_jet_eta = ak.flatten(heavy_tagged_jet_eta)
        flat_heavy_tagged_jet_hFlav = ak.flatten(heavy_tagged_jet_hFlav)

        heavy_tagged_counts = ak.num(heavy_tagged_jet_hFlav)

        #####################
        ### UNTAGGED JETS ###
        #####################

        # light untagged jets
        light_untagged_jet_pt = light_untagged_jets.pt
        light_untagged_jet_eta = np.abs(light_untagged_jets.eta)
        light_untagged_jet_hFlav = light_untagged_jets.hadronFlavour

        # Convert the jets in one dimension array and store the orignal structure of the ak array in counts
        flat_light_untagged_jet_pt = ak.flatten(light_untagged_jet_pt)
        flat_light_untagged_jet_eta = ak.flatten(light_untagged_jet_eta)
        flat_light_untagged_jet_hFlav = ak.flatten(light_untagged_jet_hFlav)

        light_untagged_counts = ak.num(light_untagged_jet_hFlav)

        # heavy untagged jets
        heavy_untagged_jet_pt = heavy_untagged_jets.pt
        heavy_untagged_jet_eta = np.abs(heavy_untagged_jets.eta)
        heavy_untagged_jet_hFlav = heavy_untagged_jets.hadronFlavour

        # Convert the jets in one dimension array and store the orignal structure of the ak array in counts
        flat_heavy_untagged_jet_pt = ak.flatten(heavy_untagged_jet_pt)
        flat_heavy_untagged_jet_eta = ak.flatten(heavy_untagged_jet_eta)
        flat_heavy_untagged_jet_hFlav = ak.flatten(heavy_untagged_jet_hFlav)

        heavy_untagged_counts = ak.num(heavy_untagged_jet_hFlav)

        # only calculate correction to nominal weight
        # replace by accessing partial weight!
        _light_tagged_central_sf = []
        _heavy_tagged_central_sf = []

        _light_untagged_central_sf = []
        _light_untagged_btagEff = []
        _heavy_untagged_central_sf = []
        _heavy_untagged_btagEff = []

        ###################
        ### TAGGED JETS ###
        ###################

        # Evluate the scale factore per jet and unflatten the scale fatores in original structure
        _light_tagged_central_sf = ak.unflatten(
            light_evaluator.evaluate(
                "central",
                wp,
                flat_light_tagged_jet_hFlav,
                flat_light_tagged_jet_eta,
                flat_light_tagged_jet_pt
            ),
            light_tagged_counts
        )

        flat_heavy_tagged_central_sf = heavy_evaluator.evaluate(
            "central",
            wp,
            flat_heavy_tagged_jet_hFlav,
            flat_heavy_tagged_jet_eta,
            flat_heavy_tagged_jet_pt
        )
        _heavy_tagged_central_sf = ak.unflatten(
            flat_heavy_tagged_central_sf,
            heavy_tagged_counts
        )

        #####################
        ### UNTAGGED JETS ###
        #####################

        # Evluate the scale factore per jet and unflatten the scale fatores in original structure
        _light_untagged_central_sf = ak.unflatten(
            light_evaluator.evaluate(
                "central",
                wp,
                flat_light_untagged_jet_hFlav,
                flat_light_untagged_jet_eta,
                flat_light_untagged_jet_pt
            ),
            light_untagged_counts
        )

        _light_untagged_btagEff = ak.unflatten(
            btageff_evaluator.evaluate(
                eff_dataset_name,
                wp,
                flat_light_untagged_jet_hFlav,
                np.abs(flat_light_untagged_jet_eta),
                flat_light_untagged_jet_pt
            ),
            light_untagged_counts
        )

        flat_heavy_untagged_central_sf = heavy_evaluator.evaluate(
            "central",
            wp,
            flat_heavy_untagged_jet_hFlav,
            flat_heavy_untagged_jet_eta,
            flat_heavy_untagged_jet_pt
        )
        _heavy_untagged_central_sf = ak.unflatten(
            flat_heavy_untagged_central_sf,
            heavy_untagged_counts
        )

        _heavy_untagged_btagEff = ak.unflatten(
            btageff_evaluator.evaluate(
                eff_dataset_name,
                wp,
                flat_heavy_untagged_jet_hFlav,
                np.abs(flat_heavy_untagged_jet_eta),
                flat_heavy_untagged_jet_pt
            ),
            heavy_untagged_counts
        )

        # Tagged jets
        light_tagged_central_prod = ak.prod(_light_tagged_central_sf, axis=1)  # Product over the tagged jets
        heavy_tagged_central_prod = ak.prod(_heavy_tagged_central_sf, axis=1)
        tagged_central = heavy_tagged_central_prod * light_tagged_central_prod

        # Untagged jets
        untagged_heavy_numerator_central_prod = _heavy_untagged_central_sf * _heavy_untagged_btagEff
        untagged_heavy_central = ak.prod((1 - untagged_heavy_numerator_central_prod) / (1 - _heavy_untagged_btagEff), axis=1)

        untagged_light_numerator_central_prod = _light_untagged_central_sf * _light_untagged_btagEff
        untagged_light_central = ak.prod((1 - untagged_light_numerator_central_prod) / (1 - _light_untagged_btagEff), axis=1)
        untagged_central = untagged_heavy_central * untagged_light_central

        w_btag_light_central = light_tagged_central_prod * untagged_light_central
        w_btag_heavy_central = heavy_tagged_central_prod * untagged_heavy_central
        w_btag_central = tagged_central * untagged_central

        if is_correction:
            w_btag = w_btag_central

            print("Weight_btag: ", w_btag)

            w_btag_up = [None for _ in btag_systematics]
            w_btag_down = [None for _ in btag_systematics]

            weights.add_multivariation(
                name="bTagFixedWP",
                weight=w_btag,
                modifierNames=btag_systematics,
                weightsUp=w_btag_up,
                weightsDown=w_btag_down,
                shift=False,
            )

            return weights

        else:
            light_variations = {}
            heavy_variations = {}

            for syst_name in btag_systematics:
                light_variations[syst_name] = {}
                heavy_variations[syst_name] = {}

                _light_tagged_up_sf = ak.unflatten(
                    light_evaluator.evaluate(
                        "up_" + syst_name,
                        wp,
                        flat_light_tagged_jet_hFlav,
                        flat_light_tagged_jet_eta,
                        flat_light_tagged_jet_pt,
                    ),
                    light_tagged_counts,
                )
                _light_tagged_down_sf = ak.unflatten(
                    light_evaluator.evaluate(
                        "down_" + syst_name,
                        wp,
                        flat_light_tagged_jet_hFlav,
                        flat_light_tagged_jet_eta,
                        flat_light_tagged_jet_pt,
                    ),
                    light_tagged_counts,
                )
                _heavy_tagged_up_sf = ak.unflatten(
                    heavy_evaluator.evaluate(
                        "up_" + syst_name,
                        wp,
                        flat_heavy_tagged_jet_hFlav,
                        flat_heavy_tagged_jet_eta,
                        flat_heavy_tagged_jet_pt,
                    ),
                    heavy_tagged_counts,
                )
                _heavy_tagged_down_sf = ak.unflatten(
                    heavy_evaluator.evaluate(
                        "down_" + syst_name,
                        wp,
                        flat_heavy_tagged_jet_hFlav,
                        flat_heavy_tagged_jet_eta,
                        flat_heavy_tagged_jet_pt,
                    ),
                    heavy_tagged_counts,
                )

                _light_untagged_up_sf = ak.unflatten(
                    light_evaluator.evaluate(
                        "up_" + syst_name,
                        wp,
                        flat_light_untagged_jet_hFlav,
                        flat_light_untagged_jet_eta,
                        flat_light_untagged_jet_pt,
                    ),
                    light_untagged_counts,
                )
                _light_untagged_down_sf = ak.unflatten(
                    light_evaluator.evaluate(
                        "down_" + syst_name,
                        wp,
                        flat_light_untagged_jet_hFlav,
                        flat_light_untagged_jet_eta,
                        flat_light_untagged_jet_pt,
                    ),
                    light_untagged_counts,
                )
                _heavy_untagged_up_sf = ak.unflatten(
                    heavy_evaluator.evaluate(
                        "up_" + syst_name,
                        wp,
                        flat_heavy_untagged_jet_hFlav,
                        flat_heavy_untagged_jet_eta,
                        flat_heavy_untagged_jet_pt,
                    ),
                    heavy_untagged_counts,
                )
                _heavy_untagged_down_sf = ak.unflatten(
                    heavy_evaluator.evaluate(
                        "down_" + syst_name,
                        wp,
                        flat_heavy_untagged_jet_hFlav,
                        flat_heavy_untagged_jet_eta,
                        flat_heavy_untagged_jet_pt,
                    ),
                    heavy_untagged_counts,
                )

                light_tagged_up_prod = ak.prod(_light_tagged_up_sf, axis=1)
                light_tagged_down_prod = ak.prod(_light_tagged_down_sf, axis=1)
                heavy_tagged_up_prod = ak.prod(_heavy_tagged_up_sf, axis=1)
                heavy_tagged_down_prod = ak.prod(_heavy_tagged_down_sf, axis=1)

                untagged_heavy_numerator_up_prod = _heavy_untagged_up_sf * _heavy_untagged_btagEff
                untagged_heavy_up = ak.prod((1 - untagged_heavy_numerator_up_prod) / (1 - _heavy_untagged_btagEff), axis=1)

                untagged_heavy_numerator_down_prod = _heavy_untagged_down_sf * _heavy_untagged_btagEff
                untagged_heavy_down = ak.prod((1 - untagged_heavy_numerator_down_prod) / (1 - _heavy_untagged_btagEff), axis=1)

                untagged_light_numerator_up_prod = _light_untagged_up_sf * _light_untagged_btagEff
                untagged_light_up = ak.prod((1 - untagged_light_numerator_up_prod) / (1 - _light_untagged_btagEff), axis=1)

                untagged_light_numerator_down_prod = _light_untagged_down_sf * _light_untagged_btagEff
                untagged_light_down = ak.prod((1 - untagged_light_numerator_down_prod) / (1 - _light_untagged_btagEff), axis=1)

                w_btag_light_up = light_tagged_up_prod * untagged_light_up
                w_btag_light_down = light_tagged_down_prod * untagged_light_down
                w_btag_heavy_up = heavy_tagged_up_prod * untagged_heavy_up
                w_btag_heavy_down = heavy_tagged_down_prod * untagged_heavy_down

                light_variations[syst_name]["up"] = w_btag_light_up
                light_variations[syst_name]["down"] = w_btag_light_down
                heavy_variations[syst_name]["up"] = w_btag_heavy_up
                heavy_variations[syst_name]["down"] = w_btag_heavy_down

            w_btag = ak.values_astype(ak.ones_like(events.event), np.float32)
            light_up = [light_variations[syst_name]["up"] / w_btag_light_central for syst_name in btag_systematics]
            light_down = [light_variations[syst_name]["down"] / w_btag_light_central for syst_name in btag_systematics]
            heavy_up = [heavy_variations[syst_name]["up"] / w_btag_heavy_central for syst_name in btag_systematics]
            heavy_down = [heavy_variations[syst_name]["down"] / w_btag_heavy_central for syst_name in btag_systematics]

            weights.add_multivariation(
                name="btagSFlight",
                weight=w_btag,
                modifierNames=btag_systematics_modifiers,
                weightsUp=light_up,
                weightsDown=light_down,
                shift=False,
            )
            weights.add_multivariation(
                name="btagSFbc",
                weight=w_btag,
                modifierNames=btag_systematics_modifiers,
                weightsUp=heavy_up,
                weightsDown=heavy_down,
                shift=False,
            )
            return weights

    else:
        logger.error(f"\n You specified the Btagging SF for dataset with {dataset_name}. First compute the Btagging efficiency correctionlib for your analysis before proceeding. \n")
        exit()


def bTagMultiFixedWP(events, jets, weights, dataset_name, mva_name, wps, bTagEffFileName, btagEffDatasetName=None, is_correction=True, year="2017", is_Run2_v15=False, **kwargs):

    avail_years = ["2016preVFP", "2016postVFP", "2017", "2018", "2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024"]
    if year not in avail_years:
        logger.error(f"\n Only fixed WP Scale Factors for the year strings {avail_years} are already implemented! \n Exiting. \n")
        exit()

    avail_modes = ["L", "M", "T", "XT", "XXT"]
    for wp in wps:
        if wp not in avail_modes:
            logger.error(f"\n Only fixed WP Scale Factors for the mode strings {avail_modes} are already implemented! \n Exiting. \n")
            exit()

    eff_dataset_name = btagEffDatasetName if btagEffDatasetName is not None else dataset_name

    inputFilePath = "JSONs/"
    if bTagEffFileName is None:
        bTagEffFileName = "midRun3"

    Run2_btag_json = ""
    if is_Run2_v15:
        Run2_btag_json = "_v15"

    btageff_correction_configs = {
        "2016preVFP":{
            "file": os.path.join(
                inputFilePath , "bTagEff/2016preVFP_UL/" + bTagEffFileName + Run2_btag_json + ".json.gz"
            )
        },
        "2016postVFP":{
            "file": os.path.join(
                inputFilePath , "bTagEff/2016postVFP_UL/" + bTagEffFileName + Run2_btag_json + ".json.gz"
            )
        },
        "2017":{
            "file": os.path.join(
                inputFilePath , "bTagEff/2017_UL/" + bTagEffFileName + Run2_btag_json + ".json.gz"
            )
        },
        "2018":{
            "file": os.path.join(
                inputFilePath , "bTagEff/2018_UL/" + bTagEffFileName + Run2_btag_json + ".json.gz"
            )
        },
        "2022preEE":{
            "file": os.path.join(
                inputFilePath , "bTagEff/2022_Summer22/" + bTagEffFileName + Run2_btag_json + ".json.gz"
            )
        },
        "2022postEE":{
            "file": os.path.join(
                inputFilePath , "bTagEff/2022_Summer22EE/" + bTagEffFileName + Run2_btag_json + ".json.gz"
            )
        },
        "2023preBPix":{
            "file": os.path.join(
                inputFilePath , "bTagEff/2023_Summer23/" + bTagEffFileName + Run2_btag_json + ".json.gz"
            )
        },
        "2023postBPix":{
            "file": os.path.join(
                inputFilePath , "bTagEff/2023_Summer23BPix/" + bTagEffFileName + Run2_btag_json + ".json.gz"
            )
        },
        "2024":{
            "file": os.path.join(
                inputFilePath , "bTagEff/2024_Summer24/" + bTagEffFileName + Run2_btag_json + ".json.gz"
            )
        }
    }

    btageff_jsonpog_file = os.path.join(
        os.path.dirname(__file__), btageff_correction_configs[year]["file"]
    )

    try:
        btageff_clib = correctionlib.CorrectionSet.from_file(btageff_jsonpog_file)
        btageff_dict = ast.literal_eval(btageff_clib._data)

        avail_procs = [current_proc["key"] for current_proc in btageff_dict["corrections"][0]["data"]["content"]]
    except:
        logger.error("\n Error when reading the dataset name from the correction lib. \n")
        exit()

    if (eff_dataset_name in avail_procs):

        mva_name_to_btag_wp_name = {
            "particleNet": "particleNet_wp_values",
            "deepJet": "deepJet_wp_values",
            "robustParticleTransformer": "robustParticleTransformer_wp_values",
            "UParTAK4": "UParTAK4_wp_values"
        }

        mva_name_to_discriminator = {
            "particleNet": "btagPNetB",
            "deepJet": "btagDeepFlavB",
            "robustParticleTransformer": "btagRobustParTAK4B",
            "UParTAK4": "btagUParTAK4B"
        }

        mva_name_to_btag_sf_name = {
            "light": {
                "particleNet": "particleNet_light",
                "deepJet": "deepJet_light",
                "robustParticleTransformer": "robustParticleTransformer_light",
                "UParTAK4": "UParTAK4_light"
            },
            "comb": {
                "particleNet": "particleNet_comb",
                "deepJet": "deepJet_comb",
                "robustParticleTransformer": "robustParticleTransformer_comb",
                "UParTAK4": "UParTAK4_comb"
            }
        }

        btag_systematics = [
            "correlated",
            "uncorrelated"
        ]
        btag_systematics_modifiers = [
            "correlated",
            f"{year}",
        ]

        btag_correction_configs = {
            "2016preVFP": {
                "file": os.path.join(
                    inputFilePath , "bTagSF/2016preVFP_UL/btagging" + Run2_btag_json + ".json.gz"
                ),
                "wp": mva_name_to_btag_wp_name[mva_name],
                "sf_light": mva_name_to_btag_sf_name["light"][mva_name],
                "sf_comb": mva_name_to_btag_sf_name["comb"][mva_name]
            },
            "2016postVFP": {
                "file": os.path.join(
                    inputFilePath , "bTagSF/2016postVFP_UL/btagging" + Run2_btag_json + ".json.gz"
                ),
                "wp": mva_name_to_btag_wp_name[mva_name],
                "sf_light": mva_name_to_btag_sf_name["light"][mva_name],
                "sf_comb": mva_name_to_btag_sf_name["comb"][mva_name]
            },
            "2017": {
                "file": os.path.join(
                    inputFilePath , "bTagSF/2017_UL/btagging" + Run2_btag_json + ".json.gz"
                ),
                "wp": mva_name_to_btag_wp_name[mva_name],
                "sf_light": mva_name_to_btag_sf_name["light"][mva_name],
                "sf_comb": mva_name_to_btag_sf_name["comb"][mva_name]
            },
            "2018": {
                "file": os.path.join(
                    inputFilePath , "bTagSF/2018_UL/btagging" + Run2_btag_json + ".json.gz"
                ),
                "wp": mva_name_to_btag_wp_name[mva_name],
                "sf_light": mva_name_to_btag_sf_name["light"][mva_name],
                "sf_comb": mva_name_to_btag_sf_name["comb"][mva_name]
            },
            "2022preEE":{
                "file": os.path.join(
                    inputFilePath , "bTagSF/2022_Summer22/btagging" + Run2_btag_json + ".json.gz"
                ),
                "wp": mva_name_to_btag_wp_name[mva_name],
                "sf_light": mva_name_to_btag_sf_name["light"][mva_name],
                "sf_comb": mva_name_to_btag_sf_name["comb"][mva_name]
            },
            "2022postEE":{
                "file": os.path.join(
                    inputFilePath , "bTagSF/2022_Summer22EE/btagging" + Run2_btag_json + ".json.gz"
                ),
                "wp": mva_name_to_btag_wp_name[mva_name],
                "sf_light": mva_name_to_btag_sf_name["light"][mva_name],
                "sf_comb": mva_name_to_btag_sf_name["comb"][mva_name]
            },
            "2023preBPix":{
                "file": os.path.join(
                    inputFilePath , "bTagSF/2023_Summer23/btagging" + Run2_btag_json + ".json.gz"
                ),
                "wp": mva_name_to_btag_wp_name[mva_name],
                "sf_light": mva_name_to_btag_sf_name["light"][mva_name],
                "sf_comb": mva_name_to_btag_sf_name["comb"][mva_name]
            },
            "2023postBPix":{
                "file": os.path.join(
                    inputFilePath , "bTagSF/2023_Summer23BPix/btagging" + Run2_btag_json + ".json.gz"
                ),
                "wp": mva_name_to_btag_wp_name[mva_name],
                "sf_light": mva_name_to_btag_sf_name["light"][mva_name],
                "sf_comb": mva_name_to_btag_sf_name["comb"][mva_name]
            },
            "2024":{
                "file": os.path.join(
                    inputFilePath , "bTagSF/2024_Summer24/btagging" + Run2_btag_json + ".json.gz"
                ),
                "wp": mva_name_to_btag_wp_name[mva_name],
                "sf_light": mva_name_to_btag_sf_name["light"][mva_name],
                "sf_comb": mva_name_to_btag_sf_name["comb"][mva_name]
            },
        }

        btagsf_jsonpog_file = os.path.join(
            os.path.dirname(__file__), btag_correction_configs[year]["file"]
        )

        # Import evaluators
        light_evaluator = correctionlib.CorrectionSet.from_file(btagsf_jsonpog_file)[btag_correction_configs[year]["sf_light"]]
        heavy_btagsf_jsonpog_file = btagsf_jsonpog_file
        if is_Run2_v15 and ("2016preVFP" in year or "2016postVFP" in year or "2017" in year or "2018" in year):
            heavy_btagsf_jsonpog_file = os.path.join(
                os.path.dirname(__file__), inputFilePath, "bTagSF/2024_Summer24/btagging.json.gz"
            )
            logger.warning("Run2 v15 missing comb SF, fallback to 2024_Summer24 comb SF.")
        heavy_evaluator = correctionlib.CorrectionSet.from_file(heavy_btagsf_jsonpog_file)[btag_correction_configs[year]["sf_comb"]]
        btageff_evaluator = correctionlib.CorrectionSet.from_file(btageff_jsonpog_file)["btagging_efficiencies"]

        # official order, from loose to tight
        wp_order = ["L", "M", "T", "XT", "XXT"]
        wps = [wp for wp in wp_order if wp in wps]
        wp_thr = {
            wp: correctionlib.CorrectionSet.from_file(btagsf_jsonpog_file)[btag_correction_configs[year]["wp"]].evaluate(wp)
            for wp in wps
        }

        # Era-dependent gen-jet pT threshold:
        # - Run2 (2016–2018): pT > 30 GeV (Run2 recommendation)
        # - Run3 (2022–2025): pT > 20 GeV (Run3 recommendation https://btv-wiki.docs.cern.ch/ScaleFactors/#important-notes)
        if year in ["2016preVFP", "2016postVFP", "2017", "2018"]:
            gen_jet_min_pt = 30
        else:
            gen_jet_min_pt = 20

        # Removing jets with eta beyond 2.5 (No bining exist in input JSON file for such jets)
        relevant_jets = events["sel_jets"][
            ((events["sel_jets"].pt) > gen_jet_min_pt)
            & (np.abs(events["sel_jets"].eta) < 2.5)
        ]

        if jets is not None:
            logger.info("using the dedicated jets input for b-tagging SF corrections and systematics")
            relevant_jets = jets[
                ((jets.pt) > gen_jet_min_pt)
                & (np.abs(jets.eta) < 2.5)
            ]

        # Split jets into "tagged J, not J+1" regions
        # region : (wp_pass, wp_fail)
        jets_by_region = {}
        jets_by_region[(None, wps[0])] = relevant_jets[relevant_jets[mva_name_to_discriminator[mva_name]] < wp_thr[wps[0]]]
        for i in range(len(wps)):
            if i + 1 < len(wps):
                jets_by_region[(wps[i], wps[i + 1])] = relevant_jets[(relevant_jets[mva_name_to_discriminator[mva_name]] >= wp_thr[wps[i]]) & (relevant_jets[mva_name_to_discriminator[mva_name]] < wp_thr[wps[i + 1]])]
            else:
                jets_by_region[(wps[i], None)] = relevant_jets[relevant_jets[mva_name_to_discriminator[mva_name]] >= wp_thr[wps[i]]]

        # Split jetcollection in heavy (hFlav == 4, 5) and in light flavor (hFlav == 0)
        light_jets_by_region = {}
        heavy_jets_by_region = {}
        for key in jets_by_region:
            light_jets_by_region[key] = jets_by_region[key][jets_by_region[key].hFlav == 0]
            heavy_jets_by_region[key] = jets_by_region[key][jets_by_region[key].hFlav >= 4]

        # only calculate correction to nominal weight
        # we will evaluate the scale factors relative to all jets used in the analysis to be multiplied

        light_jet_by_region_pt = {}
        light_jet_by_region_eta = {}
        light_jet_by_region_hFlav = {}
        flat_light_jet_by_region_pt = {}
        flat_light_jet_by_region_eta = {}
        flat_light_jet_by_region_hFlav = {}
        light_jet_by_region_counts = {}

        heavy_jet_by_region_pt = {}
        heavy_jet_by_region_eta = {}
        heavy_jet_by_region_hFlav = {}
        flat_heavy_jet_by_region_pt = {}
        flat_heavy_jet_by_region_eta = {}
        flat_heavy_jet_by_region_hFlav = {}
        heavy_jet_by_region_counts = {}

        for key in jets_by_region:
            # light jets
            light_jet_by_region_pt[key] = light_jets_by_region[key].pt
            light_jet_by_region_eta[key] = np.abs(light_jets_by_region[key].eta)
            light_jet_by_region_hFlav[key] = light_jets_by_region[key].hFlav

            # Convert the jets in one dimension array and store the orignal structure of the ak array in counts
            flat_light_jet_by_region_pt[key] = ak.flatten(light_jet_by_region_pt[key])
            flat_light_jet_by_region_eta[key] = ak.flatten(light_jet_by_region_eta[key])
            flat_light_jet_by_region_hFlav[key] = ak.flatten(light_jet_by_region_hFlav[key])
            light_jet_by_region_counts[key] = ak.num(light_jet_by_region_hFlav[key])

            # heavy tagged jets
            heavy_jet_by_region_pt[key] = heavy_jets_by_region[key].pt
            heavy_jet_by_region_eta[key] = np.abs(heavy_jets_by_region[key].eta)
            heavy_jet_by_region_hFlav[key] = heavy_jets_by_region[key].hFlav

            # Convert the jets in one dimension array and store the orignal structure of the ak array in counts
            flat_heavy_jet_by_region_pt[key] = ak.flatten(heavy_jet_by_region_pt[key])
            flat_heavy_jet_by_region_eta[key] = ak.flatten(heavy_jet_by_region_eta[key])
            flat_heavy_jet_by_region_hFlav[key] = ak.flatten(heavy_jet_by_region_hFlav[key])
            heavy_jet_by_region_counts[key] = ak.num(heavy_jet_by_region_hFlav[key])

        # only calculate correction to nominal weight
        # replace by accessing partial weight!
        _light_by_region_central_sf_pass = {}
        _light_by_region_central_sf_fail = {}
        _heavy_by_region_central_sf_pass = {}
        _heavy_by_region_central_sf_fail = {}
        _light_by_region_btagEff_pass = {}
        _light_by_region_btagEff_fail = {}
        _heavy_by_region_btagEff_pass = {}
        _heavy_by_region_btagEff_fail = {}

        # Evaluate the scale factors per jet and unflatten the scale factors in original structure
        ##################
        ### LIGHT JETS ###
        ##################
        for key in jets_by_region:
            wp_pass, wp_fail = key
            if wp_pass is not None:
                _light_by_region_central_sf_pass[key], _light_by_region_btagEff_pass[key] = evaluate_btag_sf_eff_multiwp(
                    "central",
                    wp_pass,
                    flat_light_jet_by_region_hFlav[key],
                    flat_light_jet_by_region_eta[key],
                    flat_light_jet_by_region_pt[key],
                    light_jet_by_region_counts[key],
                    light_evaluator,
                    btageff_evaluator,
                    eff_dataset_name
                )

            if wp_fail is not None:
                _light_by_region_central_sf_fail[key], _light_by_region_btagEff_fail[key] = evaluate_btag_sf_eff_multiwp(
                    "central",
                    wp_fail,
                    flat_light_jet_by_region_hFlav[key],
                    flat_light_jet_by_region_eta[key],
                    flat_light_jet_by_region_pt[key],
                    light_jet_by_region_counts[key],
                    light_evaluator,
                    btageff_evaluator,
                    eff_dataset_name
                )

        ##################
        ### HEAVY JETS ###
        ##################
        for key in jets_by_region:
            wp_pass, wp_fail = key

            if wp_pass is not None:
                _heavy_by_region_central_sf_pass[key] = ak.unflatten(
                    heavy_evaluator.evaluate(
                        "central",
                        wp_pass,
                        flat_heavy_jet_by_region_hFlav[key],
                        flat_heavy_jet_by_region_eta[key],
                        flat_heavy_jet_by_region_pt[key],
                    ),
                    heavy_jet_by_region_counts[key]
                )
                _heavy_by_region_btagEff_pass[key] = ak.unflatten(
                    btageff_evaluator.evaluate(
                        eff_dataset_name,
                        wp_pass,
                        flat_heavy_jet_by_region_hFlav[key],
                        flat_heavy_jet_by_region_pt[key]
                    ),
                    heavy_jet_by_region_counts[key]
                )

            if wp_fail is not None:
                _heavy_by_region_central_sf_fail[key] = ak.unflatten(
                    heavy_evaluator.evaluate(
                        "central",
                        wp_fail,
                        flat_heavy_jet_by_region_hFlav[key],
                        flat_heavy_jet_by_region_eta[key],
                        flat_heavy_jet_by_region_pt[key],
                    ),
                    heavy_jet_by_region_counts[key]
                )
                _heavy_by_region_btagEff_fail[key] = ak.unflatten(
                    btageff_evaluator.evaluate(
                        eff_dataset_name,
                        wp_fail,
                        flat_heavy_jet_by_region_hFlav[key],
                        flat_heavy_jet_by_region_pt[key]
                    ),
                    heavy_jet_by_region_counts[key]
                )

        ################################
        ### Calculate w_btag_central ###
        ################################
        w_btag_central = ak.ones_like(events.event, dtype=float)

        w_btag_light_central = compute_btag_multiwp(
            jets_by_region,
            _light_by_region_central_sf_pass,
            _light_by_region_central_sf_fail,
            _light_by_region_btagEff_pass,
            _light_by_region_btagEff_fail,
            is_central=True
        )
        w_btag_heavy_central = compute_btag_multiwp(
            jets_by_region,
            _heavy_by_region_central_sf_pass,
            _heavy_by_region_central_sf_fail,
            _heavy_by_region_btagEff_pass,
            _heavy_by_region_btagEff_fail,
            is_central=True
        )

        w_btag_central = w_btag_central * (w_btag_light_central * w_btag_heavy_central)

        def evaluate_btag_sf_multiwp_syst(
            variation, wp,
            flat_light_jet_hFlav, flat_light_jet_eta, flat_light_jet_pt, light_jet_counts,
            flat_heavy_jet_hFlav, flat_heavy_jet_eta, flat_heavy_jet_pt, heavy_jet_counts
        ):
            _light_sf = ak.unflatten(
                light_evaluator.evaluate(
                    variation,
                    wp,
                    flat_light_jet_hFlav,
                    flat_light_jet_eta,
                    flat_light_jet_pt,
                ),
                light_jet_counts,
            )
            _heavy_sf = ak.unflatten(
                heavy_evaluator.evaluate(
                    variation,
                    wp,
                    flat_heavy_jet_hFlav,
                    flat_heavy_jet_eta,
                    flat_heavy_jet_pt,
                ),
                heavy_jet_counts,
            )
            return _light_sf, _heavy_sf

        ###############################################################
        ### Calculate the SF corrections and systematics variations ###
        ###############################################################
        if is_correction:
            w_btag = w_btag_central

            w_btag_up = [None for _ in btag_systematics]
            w_btag_down = [None for _ in btag_systematics]

            weights.add_multivariation(
                name="bTagMultiFixedWP",
                weight=w_btag,
                modifierNames=btag_systematics,
                weightsUp=w_btag_up,
                weightsDown=w_btag_down,
                shift=False,
            )

            return weights

        else:
            light_variations = {}
            heavy_variations = {}

            for syst_name in btag_systematics:
                light_variations[syst_name] = {}
                heavy_variations[syst_name] = {}

                _light_by_region_up_sf_pass = {}
                _light_by_region_up_sf_fail = {}
                _heavy_by_region_up_sf_pass = {}
                _heavy_by_region_up_sf_fail = {}
                _light_by_region_down_sf_pass = {}
                _light_by_region_down_sf_fail = {}
                _heavy_by_region_down_sf_pass = {}
                _heavy_by_region_down_sf_fail = {}

                # Evluate the scale factore per jet and unflatten the scale fatores in original structure
                for key in jets_by_region:
                    wp_pass, wp_fail = key
                    if wp_pass is not None:
                        _light_by_region_up_sf_pass[key], _heavy_by_region_up_sf_pass[key] = evaluate_btag_sf_multiwp_syst(
                            "up_" + syst_name,
                            wp_pass,
                            flat_light_jet_by_region_hFlav[key], flat_light_jet_by_region_eta[key], flat_light_jet_by_region_pt[key], light_jet_by_region_counts[key],
                            flat_heavy_jet_by_region_hFlav[key], flat_heavy_jet_by_region_eta[key], flat_heavy_jet_by_region_pt[key], heavy_jet_by_region_counts[key]
                        )
                        _light_by_region_down_sf_pass[key], _heavy_by_region_down_sf_pass[key] = evaluate_btag_sf_multiwp_syst(
                            "down_" + syst_name,
                            wp_pass,
                            flat_light_jet_by_region_hFlav[key], flat_light_jet_by_region_eta[key], flat_light_jet_by_region_pt[key], light_jet_by_region_counts[key],
                            flat_heavy_jet_by_region_hFlav[key], flat_heavy_jet_by_region_eta[key], flat_heavy_jet_by_region_pt[key], heavy_jet_by_region_counts[key]
                        )
                    if wp_fail is not None:
                        _light_by_region_up_sf_fail[key], _heavy_by_region_up_sf_fail[key] = evaluate_btag_sf_multiwp_syst(
                            "up_" + syst_name,
                            wp_fail,
                            flat_light_jet_by_region_hFlav[key], flat_light_jet_by_region_eta[key], flat_light_jet_by_region_pt[key], light_jet_by_region_counts[key],
                            flat_heavy_jet_by_region_hFlav[key], flat_heavy_jet_by_region_eta[key], flat_heavy_jet_by_region_pt[key], heavy_jet_by_region_counts[key]
                        )
                        _light_by_region_down_sf_fail[key], _heavy_by_region_down_sf_fail[key] = evaluate_btag_sf_multiwp_syst(
                            "down_" + syst_name,
                            wp_fail,
                            flat_light_jet_by_region_hFlav[key], flat_light_jet_by_region_eta[key], flat_light_jet_by_region_pt[key], light_jet_by_region_counts[key],
                            flat_heavy_jet_by_region_hFlav[key], flat_heavy_jet_by_region_eta[key], flat_heavy_jet_by_region_pt[key], heavy_jet_by_region_counts[key]
                        )

                ################################
                ### Calculate w_btag_up/down ###
                ################################
                w_btag_light_up = compute_btag_multiwp(
                    jets_by_region,
                    _light_by_region_up_sf_pass,
                    _light_by_region_up_sf_fail,
                    _light_by_region_btagEff_pass,
                    _light_by_region_btagEff_fail,
                    is_central=False
                )
                w_btag_light_down = compute_btag_multiwp(
                    jets_by_region,
                    _light_by_region_down_sf_pass,
                    _light_by_region_down_sf_fail,
                    _light_by_region_btagEff_pass,
                    _light_by_region_btagEff_fail,
                    is_central=False
                )
                w_btag_heavy_up = compute_btag_multiwp(
                    jets_by_region,
                    _heavy_by_region_up_sf_pass,
                    _heavy_by_region_up_sf_fail,
                    _heavy_by_region_btagEff_pass,
                    _heavy_by_region_btagEff_fail,
                    is_central=False
                )
                w_btag_heavy_down = compute_btag_multiwp(
                    jets_by_region,
                    _heavy_by_region_down_sf_pass,
                    _heavy_by_region_down_sf_fail,
                    _heavy_by_region_btagEff_pass,
                    _heavy_by_region_btagEff_fail,
                    is_central=False
                )

                light_variations[syst_name]["up"] = w_btag_light_up
                light_variations[syst_name]["down"] = w_btag_light_down
                heavy_variations[syst_name]["up"] = w_btag_heavy_up
                heavy_variations[syst_name]["down"] = w_btag_heavy_down

            w_btag = ak.values_astype(ak.ones_like(events.event), np.float32)
            light_up = [light_variations[syst_name]["up"] / w_btag_light_central for syst_name in btag_systematics]
            light_down = [light_variations[syst_name]["down"] / w_btag_light_central for syst_name in btag_systematics]
            heavy_up = [heavy_variations[syst_name]["up"] / w_btag_heavy_central for syst_name in btag_systematics]
            heavy_down = [heavy_variations[syst_name]["down"] / w_btag_heavy_central for syst_name in btag_systematics]

            weights.add_multivariation(
                name="btagSFlight",
                weight=w_btag,
                modifierNames=btag_systematics_modifiers,
                weightsUp=light_up,
                weightsDown=light_down,
                shift=False,
            )
            weights.add_multivariation(
                name="btagSFbc",
                weight=w_btag,
                modifierNames=btag_systematics_modifiers,
                weightsUp=heavy_up,
                weightsDown=heavy_down,
                shift=False,
            )
            return weights

    else:
        logger.error(f"\n You specified the Btagging SF for dataset with {dataset_name}. First compute the Btagging efficiency correctionlib for your analysis before proceeding. \n")
        exit()


def cTagSF(events, weights, is_correction=True, year="2017", **kwargs):
    """
    Add c-tagging reshaping SFs as from /https://github.com/higgs-charm/flashgg/blob/dev/cH_UL_Run2_withBDT/Systematics/scripts/applyCTagCorrections.py
    BTV scale factor Wiki: https://btv-wiki.docs.cern.ch/ScaleFactors/
    events must contain jet objects, moreover evaluation of SFs works by calculating the scale factors for all the jets in the event,
    to do this in columnar style the only thing I could think of was to pad the jet collection to the max(n_jets) keep track of the "fake jets" introduced
    by this procedure and fill these position wit 1s before actually setting the weights in the collection. If someone has better ideas I'm open for suggestions
    """

    # era/year defined as parameter of the function, only Run2 is implemented up to now
    avail_years = ["2016preVFP", "2016postVFP", "2017", "2018"]
    if year not in avail_years:
        print(f"\n WARNING: only scale corrections for the year strings {avail_years} are already implemented! \n Exiting. \n")
        exit()

    ctag_systematics = [
        "Extrap",
        "Interp",
        "LHEScaleWeight_muF",
        "LHEScaleWeight_muR",
        "PSWeightFSR",
        "PSWeightISR",
        "PUWeight",
        "Stat",
        "XSec_BRUnc_DYJets_b",
        "XSec_BRUnc_DYJets_c",
        "XSec_BRUnc_WJets_c",
        "jer",
        "jesTotal",
    ]

    ctag_correction_configs = {
        "2016preVFP": {
            "file": os.path.join(
                os.path.dirname(__file__), "JSONs/cTagSF/2016/ctagging_2016preVFP.json.gz"
            ),
            "method": "deepJet_shape",
            "systs": ctag_systematics,
        },
        "2016postVFP": {
            "file": os.path.join(
                os.path.dirname(__file__), "JSONs/cTagSF/2016/ctagging_2016postVFP.json.gz"
            ),
            "method": "deepJet_shape",
            "systs": ctag_systematics,
        },
        "2017": {
            "file": os.path.join(
                os.path.dirname(__file__), "JSONs/cTagSF/2017/ctagging_2017.json.gz"
            ),
            "method": "deepJet_shape",
            "systs": ctag_systematics,
        },
        "2018": {
            "file": os.path.join(
                os.path.dirname(__file__), "JSONs/cTagSF/2018/ctagging_2018.json.gz"
            ),
            "method": "deepJet_shape",
            "systs": ctag_systematics,
        },
    }

    jsonpog_file = os.path.join(
        os.path.dirname(__file__), ctag_correction_configs[year]["file"]
    )
    evaluator = correctionlib.CorrectionSet.from_file(jsonpog_file)[
        ctag_correction_configs[year]["method"]
    ]

    events["n_jets"] = ak.num(events["sel_jets"])
    max_n_jet = int(ak.max(events["n_jets"], mask_identity=False, initial=0))

    dummy_sf = ak.ones_like(events["event"])

    if is_correction:
        # only calculate correction to nominal weight
        # we will append the scale factors relative to all jets to be multiplied
        _sf = []
        # we need a seres of masks to remember where there were no jets
        masks = []
        # to calculate the SFs we have to distinguish for different number of jets
        for i in range(max_n_jet):
            masks.append(events["n_jets"] > i)

            # I select the nth jet column
            nth_jet_hFlav = choose_jet(events["sel_jets"].hFlav, i, 0)
            nth_jet_DeepFlavour_CvsL = choose_jet(
                events["sel_jets"].btagDeepFlav_CvL, i, 0
            )
            nth_jet_DeepFlavour_CvsB = choose_jet(
                events["sel_jets"].btagDeepFlav_CvB, i, 0
            )
            _sf.append(
                evaluator.evaluate(
                    "central",
                    nth_jet_hFlav,
                    nth_jet_DeepFlavour_CvsL,
                    nth_jet_DeepFlavour_CvsB,
                )
            )

            # and fill the places where we had dummies with ones
            _sf[i] = ak.where(
                masks[i],
                _sf[i],
                dummy_sf,
            )

        sfup, sfdown = None, None
        # here we multiply all the sf for different jets in the event
        sf = dummy_sf
        for nth in _sf:
            sf = sf * nth

        sfs_up = [ak.values_astype(dummy_sf, np.float32) for _ in ctag_systematics]
        sfs_down = [ak.values_astype(dummy_sf, np.float32) for _ in ctag_systematics]

        weights.add_multivariation(
            name="cTagSF_corr",
            weight=sf,
            modifierNames=ctag_systematics,
            weightsUp=sfs_up,
            weightsDown=sfs_down,
        )

    else:
        # only calculate correction to nominal weight
        # we will append the scale factors relative to all jets to be multiplied
        _sf = []
        # we need a seres of masks to remember where there were no jets
        masks = []
        # to calculate the SFs we have to distinguish for different number of jets
        for i in range(max_n_jet):
            masks.append(events["n_jets"] > i)

            # I select the nth jet column
            nth_jet_hFlav = choose_jet(events["sel_jets"].hFlav, i, 0)
            nth_jet_DeepFlavour_CvsL = choose_jet(
                events["sel_jets"].btagDeepFlav_CvL, i, 0
            )
            nth_jet_DeepFlavour_CvsB = choose_jet(
                events["sel_jets"].btagDeepFlav_CvB, i, 0
            )
            _sf.append(
                evaluator.evaluate(
                    "central",
                    nth_jet_hFlav,
                    nth_jet_DeepFlavour_CvsL,
                    nth_jet_DeepFlavour_CvsB,
                )
            )

            # and fill the places where we had dummies with ones
            _sf[i] = ak.where(
                masks[i],
                _sf[i],
                dummy_sf,
            )

        # here we multiply all the sf for different jets in the event
        sf = dummy_sf
        for nth in _sf:
            sf = sf * nth

        variations = {}
        for syst_name in ctag_correction_configs[year]["systs"]:
            # we will append the scale factors relative to all jets to be multiplied
            _sfup = []
            _sfdown = []
            variations[syst_name] = {}
            for i in range(max_n_jet):
                # I select the nth jet column
                nth_jet_hFlav = choose_jet(events["sel_jets"].hFlav, i, 0)
                nth_jet_DeepFlavour_CvsL = choose_jet(
                    events["sel_jets"].btagDeepFlav_CvL, i, 0
                )
                nth_jet_DeepFlavour_CvsB = choose_jet(
                    events["sel_jets"].btagDeepFlav_CvB, i, 0
                )

                _sfup.append(
                    evaluator.evaluate(
                        "up_" + syst_name,
                        nth_jet_hFlav,
                        nth_jet_DeepFlavour_CvsL,
                        nth_jet_DeepFlavour_CvsB,
                    )
                )

                _sfdown.append(
                    evaluator.evaluate(
                        "down_" + syst_name,
                        nth_jet_hFlav,
                        nth_jet_DeepFlavour_CvsL,
                        nth_jet_DeepFlavour_CvsB,
                    )
                )

                # and fill the places where we had dummies with ones
                _sfup[i] = ak.where(
                    masks[i],
                    _sfup[i],
                    dummy_sf,
                )
                _sfdown[i] = ak.where(
                    masks[i],
                    _sfdown[i],
                    dummy_sf,
                )
            # here we multiply all the sf for different jets in the event
            sfup = dummy_sf
            sfdown = dummy_sf
            for i in range(len(_sf)):
                sfup = sfup * _sfup[i]
                sfdown = sfdown * _sfdown[i]

            variations[syst_name]["up"] = sfup
            variations[syst_name]["down"] = sfdown

        # coffea weights.add_multivariation() wants a list of arrays for the multiple up and down variations
        sfs_up = [variations[syst_name]["up"] / sf for syst_name in ctag_systematics]
        sfs_down = [
            variations[syst_name]["down"] / sf for syst_name in ctag_systematics
        ]

        weights.add_multivariation(
            name="cTagSF",
            weight=dummy_sf,
            modifierNames=ctag_systematics,
            weightsUp=sfs_up,
            weightsDown=sfs_down,
            shift=False,
        )

    return weights


def cTagSF_WPs(events, weights, meta, is_correction=True, year="2017", n_toys=1000, **kwargs):
    """
    Add c-tagging reshaping SFs as from /https://github.com/higgs-charm/flashgg/blob/dev/cH_UL_Run2_withBDT/Systematics/scripts/applyCTagCorrections.py
    BTV scale factor Wiki: https://btv-wiki.docs.cern.ch/ScaleFactors/
    events must contain jet objects, moreover evaluation of SFs works by calculating the scale factors for all the jets in the event,
    to do this in columnar style the only thing I could think of was to pad the jet collection to the max(n_jets) keep track of the "fake jets" introduced
    by this procedure and fill these position wit 1s before actually setting the weights in the collection. If someone has better ideas I'm open for suggestions
    """
    logger.warning("Applying PNet c-tagging SFs")
    # era/year defined as parameter of the function, only Run2 is implemented up to now
    avail_years = ["2016preVFP", "2016postVFP", "2017", "2018", "2022preEE", "2022postEE", "2023preBPix", "2023postBPix"]
    if year not in avail_years:
        print(f"\n WARNING: only cTagSF corrections for the year strings {avail_years} are already implemented! \n Exiting. \n")
        exit()

    ctag_systematics = [
        "Stat",
        # "LHEScaleWeight_muF_ttbar",
        # "LHEScaleWeight_muF_singlet",
        # "LHEScaleWeight_muF_wjets",
        # "LHEScaleWeight_muF_zjets",
        # "LHEScaleWeight_muF_diboson",
        # "LHEScaleWeight_muR_ttbar",
        # "LHEScaleWeight_muR_singlet",
        # "LHEScaleWeight_muR_wjets",
        # "LHEScaleWeight_muR_zjets",
        # "LHEScaleWeight_muR_diboson",
        # "LHEScaleWeight_aS_ttbar",
        # "LHEScaleWeight_aS_singlet",
        # "LHEScaleWeight_aS_wjets",
        # "LHEScaleWeight_aS_zjets",
        # "LHEScaleWeight_aS_diboson",
        # "LHEScaleWeight_PDF_ttbar",
        # "LHEScaleWeight_PDF_singlet",
        # "LHEScaleWeight_PDF_wjets",
        # "LHEScaleWeight_PDF_zjets",
        # "LHEScaleWeight_PDF_diboson",
        # "PSWeightISR_ttbar",
        # "PSWeightISR_singlet",
        # "PSWeightISR_wjets",
        # "PSWeightISR_zjets",
        # "PSWeightISR_diboson",
        # "PSWeightFSR_ttbar",
        # "PSWeightFSR_singlet",
        # "PSWeightFSR_wjets",
        # "PSWeightFSR_zjets",
        # "PSWeightFSR_diboson",
        # "TTWeight_ttbar",
        "XSec_WJets_c",
        "XSec_WJets_b",
        "XSec_ZJets_c",
        "XSec_ZJets_b",
        "XSec_ttbar",
        "XSec_singlet_tW",
        "XSec_singlet_tCh",
        "JES",
        "JER",
        "PUWeight",
        "Ele_Reco",
        "Ele_ID",
        "Ele_Trigger",
        "Mu_ID",
        "Mu_Iso",
        "Mu_Trigger",
    ]

    # if self._opts['split_stat_unc']:
    #     flavors = ['flavB', 'flavC', 'flavL']
    #     tag_categories = ['C0', 'C1', 'C2', 'C3', 'C4', 'B0', 'B1', 'B2', 'B3', 'B4']
    #     for flav in flavors:
    #         for tag in tag_categories:
    #             ctag_systematics.append(f'Stat_{flav}_{tag}')

    ctag_correction_configs = {
        "2016preVFP": {
            "file": os.path.join(
                os.path.dirname(__file__), "JSONs/cTagSF/2016/flavTaggingSF_2016preVFP_UL.json.gz"
            ),
            "method": "particleNetAK4_shape",
            "systs": ctag_systematics,
        },
        "2016postVFP": {
            "file": os.path.join(
                os.path.dirname(__file__), "JSONs/cTagSF/2016/flavTaggingSF_2016postVFP_UL.json.gz"
            ),
            "method": "particleNetAK4_shape",
            "systs": ctag_systematics,
        },
        "2017": {
            "file": os.path.join(
                os.path.dirname(__file__), "JSONs/cTagSF/2017/flavTaggingSF_2017_UL.json.gz"
            ),
            "method": "particleNetAK4_shape",
            "systs": ctag_systematics,
        },
        "2018": {
            "file": os.path.join(
                os.path.dirname(__file__), "JSONs/cTagSF/2018/flavTaggingSF_2018_UL.json.gz"
            ),
            "method": "particleNetAK4_shape",
            "systs": ctag_systematics,
        },
        "2022preEE": {
            "file": os.path.join(
                os.path.dirname(__file__), "JSONs/cTagSF/2022_Summer22/flavTaggingSF_2022preEE.json.gz"
            ),
            "method": "ParticleNetAK4_pseudocontinuous",
            "systs": ctag_systematics,
        },
        "2022postEE": {
            "file": os.path.join(
                os.path.dirname(__file__), "JSONs/cTagSF/2022_Summer22EE/flavTaggingSF_2022postEE.json.gz"
            ),
            "method": "ParticleNetAK4_pseudocontinuous",
            "systs": ctag_systematics,
        },
        "2023preBPix": {
            "file": os.path.join(
                os.path.dirname(__file__), "JSONs/cTagSF/2023_Summer23/flavTaggingSF_2023preBPix.json.gz"
            ),
            "method": "ParticleNetAK4_pseudocontinuous",
            "systs": ctag_systematics,
        },
        "2023postBPix": {
            "file": os.path.join(
                os.path.dirname(__file__), "JSONs/cTagSF/2023_Summer23BPix/flavTaggingSF_2023postBPix.json.gz"
            ),
            "method": "ParticleNetAK4_pseudocontinuous",
            "systs": ctag_systematics,
        },
    }

    jsonpog_file = os.path.join(
        os.path.dirname(__file__), ctag_correction_configs[year]["file"]
    )
    evaluator = correctionlib.CorrectionSet.from_file(jsonpog_file)[
        ctag_correction_configs[year]["method"]
    ]

    events["n_jets"] = ak.num(events["sel_jets"])
    max_n_jet = int(ak.max(events["n_jets"], mask_identity=False, initial=0))
    dummy_sf = ak.ones_like(events["event"])

    if is_correction:
        # only calculate correction to nominal weight
        # we will append the scale factors relative to all jets to be multiplied
        _sf = []
        # we need a seres of masks to remember where there were no jets
        masks = []
        # to calculate the SFs we have to distinguish for different number of jets
        for i in range(max_n_jet):
            masks.append(events["n_jets"] > i)

            # I select the nth jet column
            nth_jet_hFlav = choose_jet(events["sel_jets"].hFlav, i, 0)
            nth_jet_abs_eta = choose_jet(
                abs(events["sel_jets"].eta), i, -999.
            )
            nth_jet_pt = choose_jet(
                events["sel_jets"].pt, i, -999.
            )

            # attach ParticleNet scores
            nth_jet_pn_b_plus_c = choose_jet(events["sel_jets"].pn_b_plus_c, i, -1)
            nth_jet_pn_b_vs_c = choose_jet(events["sel_jets"].pn_b_vs_c, i, -1)

            # evaluate the working point
            # ParticleNetAK4 -- exclusive b- and c-tagging categories
            # 5x: b-tagged; 4x: c-tagged;
            # 0: light
            # -1: untagged
            wp = evaluate_ctag_wp(meta["HPC_ctag_WPs"]["wps"], nth_jet_pn_b_plus_c, nth_jet_pn_b_vs_c)
            valid_wp = (wp != -1)
            wp_evaluate = ak.where(valid_wp, wp, 0)  # for untagged jets we set wp to 0, but we will fill the SF with 1 later
            _sf.append(
                evaluator.evaluate(
                    "central",
                    nth_jet_hFlav,
                    wp_evaluate,
                    nth_jet_abs_eta,
                    nth_jet_pt,
                )
            )

            # and fill the places where we had dummies with ones
            _sf[i] = ak.where(
                masks[i] & valid_wp,
                _sf[i],
                dummy_sf,
            )

        sfup, sfdown = None, None
        # here we multiply all the sf for different jets in the event
        sf = dummy_sf
        for nth in _sf:
            sf = sf * nth

        sfs_up = [ak.values_astype(dummy_sf, np.float32) for _ in ctag_systematics]
        sfs_down = [ak.values_astype(dummy_sf, np.float32) for _ in ctag_systematics]

        weights.add_multivariation(
            name="cTagSF_corr",
            weight=sf,
            modifierNames=ctag_systematics,
            weightsUp=sfs_up,
            weightsDown=sfs_down,
        )

    else:
        # only calculate correction to nominal weight
        # we will append the scale factors relative to all jets to be multiplied
        _sf = []
        # we need a seres of masks to remember where there were no jets
        masks = []
        # to calculate the SFs we have to distinguish for different number of jets
        for i in range(max_n_jet):
            masks.append(events["n_jets"] > i)

            # I select the nth jet column
            nth_jet_hFlav = choose_jet(events["sel_jets"].hFlav, i, 0)
            nth_jet_abs_eta = choose_jet(
                abs(events["sel_jets"].eta), i, -999.
            )
            nth_jet_pt = choose_jet(
                events["sel_jets"].pt, i, -999.
            )

            # attach ParticleNet scores
            nth_jet_pn_b_plus_c = choose_jet(events["sel_jets"].pn_b_plus_c, i, -1)
            nth_jet_pn_b_vs_c = choose_jet(events["sel_jets"].pn_b_vs_c, i, -1)

            # evaluate the working point
            wp = evaluate_ctag_wp(meta["HPC_ctag_WPs"]["wps"], nth_jet_pn_b_plus_c, nth_jet_pn_b_vs_c)
            valid_wp = (wp != -1)
            wp_evaluate = ak.where(valid_wp, wp, 0)  # for untagged jets we set wp to 0, but we will fill the SF with 1 later

            _sf.append(
                evaluator.evaluate(
                    "central",
                    nth_jet_hFlav,
                    wp_evaluate,
                    nth_jet_abs_eta,
                    nth_jet_pt,
                )
            )

            # and fill the places where we had dummies with ones
            _sf[i] = ak.where(
                masks[i] & valid_wp,
                _sf[i],
                dummy_sf,
            )

        # here we multiply all the sf for different jets in the event
        sf = dummy_sf
        for nth in _sf:
            sf = sf * nth

        variations = {}
        for syst_name in ctag_correction_configs[year]["systs"]:
            # we will append the scale factors relative to all jets to be multiplied
            _sfup = []
            _sfdown = []
            variations[syst_name] = {}

            for i in range(max_n_jet):
                masks.append(events["n_jets"] > i)

                # I select the nth jet column
                nth_jet_hFlav = choose_jet(events["sel_jets"].hFlav, i, 0)
                nth_jet_abs_eta = choose_jet(
                    abs(events["sel_jets"].eta), i, -999.
                )
                nth_jet_pt = choose_jet(
                    events["sel_jets"].pt, i, -999.
                )

                # attach ParticleNet scores
                nth_jet_pn_b_plus_c = choose_jet(events["sel_jets"].pn_b_plus_c, i, -1)
                nth_jet_pn_b_vs_c = choose_jet(events["sel_jets"].pn_b_vs_c, i, -1)

                # evaluate the working point
                wp = evaluate_ctag_wp(meta["HPC_ctag_WPs"]["wps"], nth_jet_pn_b_plus_c, nth_jet_pn_b_vs_c)
                valid_wp = (wp != -1)
                wp_evaluate = ak.where(valid_wp, wp, 0)  # for untagged jets we set wp to 0, but we will fill the SF with 1 later

                if "Stat" not in syst_name:
                    _sfup.append(
                        evaluator.evaluate(
                            "up_" + syst_name,
                            nth_jet_hFlav,
                            wp_evaluate,
                            nth_jet_abs_eta,
                            nth_jet_pt,
                        )
                    )

                    _sfdown.append(
                        evaluator.evaluate(
                            "down_" + syst_name,
                            nth_jet_hFlav,
                            wp_evaluate,
                            nth_jet_abs_eta,
                            nth_jet_pt,
                        )
                    )

                else:
                    sf_central = evaluator.evaluate("central", nth_jet_hFlav, wp_evaluate, nth_jet_abs_eta, nth_jet_pt)
                    sf_stat_up = evaluator.evaluate(f'up_{syst_name}', nth_jet_hFlav, wp_evaluate, nth_jet_abs_eta, nth_jet_pt)
                    sf_stat_dn = evaluator.evaluate(f'down_{syst_name}', nth_jet_hFlav, wp_evaluate, nth_jet_abs_eta, nth_jet_pt)
                    err = (np.abs(sf_stat_up - sf_central) + np.abs(sf_central - sf_stat_dn)) / 2
                    np.random.seed(np.random.randint(0, 2**32))
                    sf_toys = np.random.normal(sf_central[:, None], err[:, None], (len(nth_jet_hFlav), n_toys))

                    # here
                    wgt_toys = np.clip(sf_toys, 0.3, 3)
                    wgt_stat_dn, wgt_stat_up = np.percentile(wgt_toys, q=[16, 84])

                    _sfup.append(wgt_stat_up)
                    _sfdown.append(wgt_stat_dn)

                # and fill the places where we had dummies with ones
                _sfup[i] = ak.where(
                    masks[i] & valid_wp,
                    _sfup[i],
                    dummy_sf,
                )
                _sfdown[i] = ak.where(
                    masks[i] & valid_wp,
                    _sfdown[i],
                    dummy_sf,
                )
            # here we multiply all the sf for different jets in the event
            sfup = dummy_sf
            sfdown = dummy_sf
            for i in range(len(_sf)):
                sfup = sfup * _sfup[i]
                sfdown = sfdown * _sfdown[i]

            variations[syst_name]["up"] = sfup
            variations[syst_name]["down"] = sfdown

        # coffea weights.add_multivariation() wants a list of arrays for the multiple up and down variations
        sfs_up = [variations[syst_name]["up"] / sf for syst_name in ctag_systematics]
        sfs_down = [
            variations[syst_name]["down"] / sf for syst_name in ctag_systematics
        ]

        weights.add_multivariation(
            name="cTagSF",
            weight=dummy_sf,
            modifierNames=ctag_systematics,
            weightsUp=sfs_up,
            weightsDown=sfs_down,
            shift=False,
        )

    return weights


def Zpt(
    events,
    weights,
    logger,
    dataset_name,
    is_correction=True,
    year="2022postEE",
    **kwargs,
):
    """
    Z pt reweighting
    """
    systematic = "Z pt reweighting"

    json_dict = {
        "2016postVFP_UL": os.path.join(
            os.path.dirname(__file__),
            "./JSONs/my_Zpt_reweighting.json.gz",
        ),
        "2016preVFP_UL": os.path.join(
            os.path.dirname(__file__),
            "./JSONs/my_Zpt_reweighting.json.gz",
        ),
        "2017_UL": os.path.join(
            os.path.dirname(__file__),
            "./JSONs/my_Zpt_reweighting.json.gz",
        ),
        "2018_UL": os.path.join(
            os.path.dirname(__file__),
            "./JSONs/my_Zpt_reweighting.json.gz",
        ),
        "2022postEE": os.path.join(
            os.path.dirname(__file__),
            "./JSONs/my_Zpt_reweighting.json.gz",
        ),
        "2023": os.path.join(
            os.path.dirname(__file__),
            "./JSONs/my_Zpt_reweighting.json.gz",
        ),
    }
    key_map = {
        "2016postVFP_UL": "Zpt_reweight",
        "2016preVFP_UL": "Zpt_reweight",
        "2017_UL": "Zpt_reweight",
        "2018_UL": "Zpt_reweight",
        "2022postEE": "Zpt_reweight",
        "2023": "Zpt_reweight",
    }

    # inputs
    input_value = {
        "Zpt": events.mmy_pt,
    }
    cset = correctionlib.CorrectionSet.from_file(json_dict[year])
    # list(cset) # get keys in cset
    sf = cset[key_map[year]]

    logger.debug(f"{systematic}:{key_map[year]}, year: {year} ===> {dataset_name}")
    if is_correction:
        nom = sf.evaluate(input_value["Zpt"])
        weights.add(name="ZptWeight_corr", weight=nom)
    else:
        nom = sf.evaluate(input_value["Zpt"])
        up = sf.evaluate(input_value["Zpt"])
        down = sf.evaluate(input_value["Zpt"])
        weights.add(
            name="ZptWeight",
            weight=ak.ones_like(nom),
            weightUp=up / nom,
            weightDown=down / nom,
        )

    return weights


def Higgs_plus_HF_syst(events, weights, flav="b", pt_min=25, rel_unc=0.5, **kwargs):
    """
    Apply a flat systematic uncertainty for ggH or VBF events with Higgs plus heavy-flavor (b or c) jets.

    This function assigns a relative weight variation to events containing at least one
    generated heavy-flavor jet (identified via hadronFlavour == 5 for b or 4 for c) with
    transverse momentum above `pt_min` and pseudorapidity |eta| < 2.5. Events without such jets
    receive no uncertainty.

    Parameters
    ----------
    events : ak.Array
        The event record, typically a NanoAOD-format awkward array.
    weights : coffea.analysis_tools.Weights
        The Weights object to which the systematic weights will be added.
    flav : {"b", "c"}, optional
        The heavy-flavor type to consider ("b" for b-jets, "c" for c-jets).
    pt_min : float, optional
        Minimum transverse momentum (pT) threshold for heavy-flavor jet selection in GeV.
    rel_unc : float, optional
        Relative uncertainty to apply (e.g., 0.5 for ±50%).

    Returns
    -------
    weights : coffea.analysis_tools.Weights
        The modified Weights object with the added systematic uncertainty named
        "Higgs_plus_b_syst" or "Higgs_plus_c_syst", depending on the chosen flavor.

    Raises
    ------
    ValueError
        If `flav` is not one of "b" or "c".
    """

    logger.info(
        f"Applying Higgs plus {flav} systematic uncertainty with rel_unc={rel_unc}.\
        Make sure you apply it only on ggH or VBF samples."
    )

    genJets = get_genJets(events=events, pt_cut=pt_min, eta_cut=2.5)

    if flav == "b":
        SF_name = "Higgs_plus_b_syst"
        flav = 5
    elif flav == "c":
        SF_name = "Higgs_plus_c_syst"
        flav = 4
    else:
        raise ValueError("flav must be either 'b' or 'c'")

    num_HF_jets = ak.sum((genJets.hadronFlavour == flav), axis=-1)

    up = ak.where(num_HF_jets > 0, 1 + rel_unc, 1.0)
    down = ak.where(num_HF_jets > 0, 1 - rel_unc, 1.0)

    weights.add(name=SF_name, weight=ak.ones_like(up), weightUp=up, weightDown=down)

    return weights


def electronSFs(
    electrons,
    weights,
    year,
    sf_key,
    is_correction=True,
    return_jagged=False,
    variation="nominal",
    **kwargs,
):
    """
    Electron identification or reconstruction scale factors for Run 3 (2022/2023/2024).
    Documentation: https://twiki.cern.ch/twiki/bin/view/CMS/EgammSFandSSRun3
    Can either return jagged per-electron SFs for a given variation or add event-level
    weights to a coffea Weights container. In the latter case, the SF corresponds
    to the product over all electron SFs in the event.
    Take note that this is only correct if that number of electrons is required in the event selection.

    Parameters
    ----------
    electrons : ak.Array
        Needs: pt, eta, and (for 2023) phi.
    weights : coffea.analysis_tools.Weights or None
        Is modified unless return_jagged=True.
    year : {"2022preEE","2022postEE","2023preBPix","2023postBPix", "2024"}
    sf_key : str
        ID WP ("Loose","Medium","Tight","wp90iso","wp80iso") or "Reco".
        In the "Reco" case, the SFs are obtained in three pt slices and combined into one SF.
    is_correction : bool, default True
        If True, add central event-level weight only. If False, add up/down variations.
    return_jagged : bool, default False
        If True, return jagged per-electron SFs for `variation`.
    variation : {"nominal","up","down"}, default "nominal"
        Used only when return_jagged=True.

    Returns
    -------
    Weights or ak.Array
    """
    avail_years = ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024"]
    if year not in avail_years:
        logger.error(f"Only electron corrections for {avail_years} implemented!")
        raise ValueError(f"Year '{year}' not supported for electron corrections.")

    path_json = os.path.join(os.path.dirname(__file__), f"JSONs/POG/EGM/{year}/electron.json.gz")
    evaluator = correctionlib.CorrectionSet.from_file(path_json)["Electron-ID-SF"]
    era_label = {
        "2022preEE": "2022Re-recoBCD",
        "2022postEE": "2022Re-recoE+PromptFG",
        "2023preBPix": "2023PromptC",
        "2023postBPix": "2023PromptD",
        "2024": "2024Prompt",
    }[year]

    # Flatten per-electron inputs
    counts = ak.num(electrons.pt)
    eta = ak.flatten(electrons.eta)
    pt = ak.flatten(electrons.pt)
    phi = ak.flatten(electrons.phi)  # used in 2023

    def _eval(var_key: str, key: str, eta_in, pt_in, phi_in):
        """Evaluate a single correction key (ID WP or one reco-slice key) on flat arrays."""
        if "2023" in year:
            return evaluator.evaluate(era_label, var_key, key, eta_in, pt_in, phi_in)
        else:
            return evaluator.evaluate(era_label, var_key, key, eta_in, pt_in)

    def _eval_id(var_key: str):
        """
        Non-reco case: evaluate once on full arrays.
        """
        return _eval(var_key, sf_key, eta, pt, phi)

    def _eval_reco(var_key: str):
        """Evaluate the three pt slices and scatter back into one flat array."""
        mask_lt20 = pt < 20
        mask_20_75 = (pt >= 20) & (pt < 75)
        mask_ge75 = pt >= 75

        sf_lt20 = _eval(var_key, "RecoBelow20", eta[mask_lt20], pt[mask_lt20], phi[mask_lt20])
        sf_20_75 = _eval(var_key, "Reco20to75", eta[mask_20_75], pt[mask_20_75], phi[mask_20_75])
        sf_ge75 = _eval(var_key, "RecoAbove75", eta[mask_ge75], pt[mask_ge75], phi[mask_ge75])

        sf = np.ones(len(pt), dtype=float)
        sf[np.asarray(mask_lt20)] = ak.to_numpy(sf_lt20)
        sf[np.asarray(mask_20_75)] = ak.to_numpy(sf_20_75)
        sf[np.asarray(mask_ge75)] = ak.to_numpy(sf_ge75)

        return ak.Array(sf)

    def _eval_dispatch(var_key: str):
        if sf_key == "Reco":
            return _eval_reco(var_key)
        else:
            return _eval_id(var_key)

    # Return jagged per-electron SFs if requested
    if return_jagged:
        key = "sf" if variation == "nominal" else f"sf{variation}"
        sf_flat = _eval_dispatch(key)
        return ak.unflatten(sf_flat, counts)

    # Event-level weights (product over electrons)
    sf_nom = ak.unflatten(_eval_dispatch("sf"), counts)
    prod_nom = ak.prod(sf_nom, axis=1)

    if is_correction:
        name = "ElectronRecoSF_corr" if sf_key == "reco" else f"ElectronId{sf_key}SF_corr"
        weights.add(name=name, weight=prod_nom, weightUp=None, weightDown=None)
    else:
        sf_up = ak.unflatten(_eval_dispatch("sfup"), counts)
        sf_dn = ak.unflatten(_eval_dispatch("sfdown"), counts)
        prod_up = ak.prod(sf_up, axis=1)
        prod_dn = ak.prod(sf_dn, axis=1)
        name = "ElectronRecoSF" if sf_key == "reco" else f"ElectronId{sf_key}SF"
        weights.add(name=name, weight=np.ones(len(prod_nom)), weightUp=prod_up, weightDown=prod_dn)

    return weights


def electron_reco_sf_for_Zee_val_photons(photons, weights, year, is_correction=True, **kwargs):
    """
    Electron reconstruction SFs for Z→ee validation photons.

    Builds a (N_events, 2) jagged array from `photons.pho_lead` and
    `photons.pho_sublead` and forwards it to `electronSFs(..., sf_key="Reco")` as electrons.
    Intended for use with the `--validate-with-electrons` mode in the processors.

    Parameters
    ----------
    photons : ak.Array
        Diphoton object as in processors, must have `pho_lead` and `pho_sublead`.
    weights : coffea.analysis_tools.Weights
    year : {"2022preEE","2022postEE","2023preBPix","2023postBPix"}
    is_correction : bool, default True
    **kwargs : ignored

    Returns
    -------
    coffea.analysis_tools.Weights
    """
    photons_jagged = ak.concatenate(
        [ak.singletons(photons.pho_lead), ak.singletons(photons.pho_sublead)], axis=1
    )
    return electronSFs(
        electrons=photons_jagged,
        weights=weights,
        year=year,
        is_correction=is_correction,
        sf_key="Reco",
    )


def muonSFs(muons, weights, year="2022preEE",
            SF_name="NUM_TightID_DEN_TrackerMuons",
            is_correction=True, return_jagged=False, variation="nominal", **kwargs):
    """
    Can either return jagged per-muon SFs for a given variation or add event-level
    weights to a coffea Weights container. In the latter case, the SF corresponds
    to the product over all muon SFs in the event.
    Take note that this is only correct if that number of muons is required in the event selection.
    For IDs, the low-pt (<15 GeV) SFs are used together with the medium-pt ones.

    Parameters
    ----------
    muons : ak.Array
        Awkward array with fields at least: pt, eta.
    weights : coffea.analysis_tools.Weights or None
        Weights container to which the event-level weight is added.
        If return_jagged=True, this is ignored and may be None.
    year : str, default "2022preEE"
        One of {"2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024"}.
        Selects the corresponding POG JSON file.
    SF_name : str, default "NUM_TightID_DEN_TrackerMuons"
        Name of the correction within the muon POG JSON, for example
        "NUM_MediumID_DEN_TrackerMuons", "NUM_TightPFIso_DEN_MediumID",
        "NUM_LoosePFIso_DEN_MediumID", etc.
    is_correction : bool, default True
        If True, add the central event-level weight equal to the product of
        per-muon nominal SFs over all muons in the event.
        If False, add up/down variations as ratios to nominal (weightUp and
        weightDown are product(up)/product(nominal) and product(down)/product(nominal)).
    return_jagged : bool, default False
        If True, do not touch `weights`; instead return a jagged ak.Array of
        per-muon SFs for the requested `variation`.
    variation : {"nominal", "up", "down"}, default "nominal"
        Which variation to evaluate when return_jagged=True.
    **kwargs
        Unused; accepted for a uniform call signature.

    Returns
    -------
    weights or ak.Array
        - If return_jagged=True: jagged ak.Array of per-muon SFs for the
          requested variation (covers all selected muons; low-pt uses JPsi IDs).
        - Otherwise: the modified Weights object with an event-level weight
          added. For events with no muons, the event-level product is 1.
    """
    avail_years = ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024"]
    if year not in avail_years:
        logger.error(f"Only muon corrections for {avail_years} implemented!")
        raise ValueError(f"Year '{year}' not supported for muon corrections.")

    # shoose base dir for JSONs
    if year == "2022preEE":
        base_dir = os.path.join(os.path.dirname(__file__), "JSONs/POG/MUO/2022_Summer22")
    elif year == "2022postEE":
        base_dir = os.path.join(os.path.dirname(__file__), "JSONs/POG/MUO/2022_Summer22EE")
    elif year == "2023preBPix":
        base_dir = os.path.join(os.path.dirname(__file__), "JSONs/POG/MUO/2023_Summer23")
    elif year == "2023postBPix":
        base_dir = os.path.join(os.path.dirname(__file__), "JSONs/POG/MUO/2023_Summer23BPix")
    else:
        base_dir = os.path.join(os.path.dirname(__file__), "JSONs/POG/MUO/2024")

    json_file = os.path.join(base_dir, "muon_Z.json.gz")
    json_file_low_pt = os.path.join(base_dir, "muon_JPsi.json.gz")  # IDs only, for pt < 15 GeV

    # Low-pt IDs available in the JPsi file
    low_pt_id_keys = {
        "NUM_SoftID_DEN_TrackerMuons",
        "NUM_LooseID_DEN_TrackerMuons",
        "NUM_MediumID_DEN_TrackerMuons",
        "NUM_TightID_DEN_TrackerMuons",
    }
    use_low_pt_file = (SF_name in low_pt_id_keys)

    evaluator = correctionlib.CorrectionSet.from_file(json_file)[SF_name]
    evaluator_low_pt = None
    if use_low_pt_file:
        evaluator_low_pt = correctionlib.CorrectionSet.from_file(json_file_low_pt)[SF_name]

    counts = ak.num(muons.pt)
    abseta = ak.flatten(np.abs(muons.eta))
    pt = ak.flatten(muons.pt)

    low_pt_mask = (pt < 15.0)

    # map variation to evaluator keys
    var_map = {"nominal": "nominal",
               "up": "systup",
               "down": "systdown"}
    key = var_map[variation]

    # start with ones, then fill per region
    sf_flat = np.ones_like(ak.to_numpy(pt), dtype=float)

    # main region evaluation (pt >= 15 GeV via *_Z.json)
    if ak.any(~low_pt_mask):
        vals = evaluator.evaluate(abseta[~low_pt_mask], pt[~low_pt_mask], key)
        sf_flat[ak.to_numpy(~low_pt_mask)] = vals

    # low-pt region (pt < 15 GeV): only if ID key is requested; otherwise keep 1.0
    if use_low_pt_file and ak.any(low_pt_mask):
        vals_low = evaluator_low_pt.evaluate(abseta[low_pt_mask], pt[low_pt_mask], key)
        sf_flat[ak.to_numpy(low_pt_mask)] = vals_low

    sf_jagged = ak.unflatten(sf_flat, counts)
    if return_jagged:
        return sf_jagged
    else:
        # event-level product over all selected muons
        prod_nom = ak.prod(sf_jagged, axis=1)

        if is_correction:
            weights.add(name=SF_name + "_corr", weight=prod_nom, weightUp=None, weightDown=None)
        else:
            # up
            sf_flat_up = sf_flat.copy()
            if ak.any(~low_pt_mask):
                vals_up = evaluator.evaluate(abseta[~low_pt_mask], pt[~low_pt_mask], "systup")
                sf_flat_up[ak.to_numpy(~low_pt_mask)] = vals_up
            if use_low_pt_file and ak.any(low_pt_mask):
                vals_low_up = evaluator_low_pt.evaluate(abseta[low_pt_mask], pt[low_pt_mask], "systup")
                sf_flat_up[ak.to_numpy(low_pt_mask)] = vals_low_up
            prod_up = ak.prod(ak.unflatten(sf_flat_up, counts), axis=1)

            # down
            sf_flat_dn = sf_flat.copy()
            if ak.any(~low_pt_mask):
                vals_dn = evaluator.evaluate(abseta[~low_pt_mask], pt[~low_pt_mask], "systdown")
                sf_flat_dn[ak.to_numpy(~low_pt_mask)] = vals_dn
            if use_low_pt_file and ak.any(low_pt_mask):
                vals_low_dn = evaluator_low_pt.evaluate(abseta[low_pt_mask], pt[low_pt_mask], "systdown")
                sf_flat_dn[ak.to_numpy(low_pt_mask)] = vals_low_dn
            prod_dn = ak.prod(ak.unflatten(sf_flat_dn, counts), axis=1)

            # store as ratios to nominal
            weights.add(
                name=SF_name,
                weight=np.ones(len(prod_nom)),
                weightUp=prod_up / ak.to_numpy(prod_nom),
                weightDown=prod_dn / ak.to_numpy(prod_nom),
            )
        return weights


def atLeast1LeptonIdSF(
    electrons,
    muons,
    weights,
    year,
    ele_SF_names=("wp90iso", "Reco"),
    mu_SF_names=("NUM_MediumID_DEN_TrackerMuons", "NUM_TightPFIso_DEN_MediumID"),
    name_base="atLeast1LeptonIdSF",
    is_correction=True,
    **kwargs,
):
    """
    Event-level SF for analyses requiring ">= 1 lepton (e or mu)" using OR logic and per-lepton SFs.

    This function builds an event weight appropriate for selections that pass when
    at least one lepton is identified. It combines per-lepton scale factors (SFs)
    across components (e.g. ID, ISO) into a per-lepton total SF, and then applies
    the OR probability:
        w = [1 - (prod_e (1 - SF_e * epsilon_e_MC)) * (prod_mu (1 - SF_mu * epsilon_mu_MC))]
            / [1 - (prod_e (1 - epsilon_e_MC))     * (prod_mu (1 - epsilon_mu_MC))]

    where epsilon_*_MC are average MC efficiencies for the working points used.
    For systematics, it creates one nuisance parameter per SF component (electron
    ID WP and each muon SF name), varying a single component up/down while keeping
    all other components at nominal, and stores the variations as ratios to the
    central weight.

    Parameters
    ----------
    electrons : ak.Array
        Awkward Array of selected electrons used by the category. Must provide
        at least `pt` and `eta` (and `phi` if your electron SFs require it).
    muons : ak.Array
        Awkward Array of selected muons used by the category. Must provide
        at least `pt` and `eta`.
    weights : coffea.analysis_tools.Weights
        Weights container to which the event-level weight and NP variations
        are added.
    year : str
        Data-taking period string understood by the underlying SF evaluators,
        e.g. "2022preEE", "2022postEE", "2023preBPix", "2023postBPix".
    ele_SF_names: str, default ("wp90iso", "Reco")
        Electron SF component(s) to multiply per electron. Can be a single key
        (e.g. "wp90iso" or "Reco") or a list/tuple like ["Reco","wp90iso"].
    mu_SF_names : tuple[str] or list[str], default ("NUM_MediumID_DEN_TrackerMuons", "NUM_TightPFIso_DEN_MediumID")
        Iterable of muon SF component names to multiply per muon (for example
        an ID component and an ISO component). Each entry must match a key in
        the muon POG JSON used by `muonSFs`.
    name_base : str, default "atLeast1LeptonIdSF"
        Base name for the weight(s) added to the Weights container. The central
        weight is stored as `{name_base}`. Per-component NPs are stored as
        `{name_base}_ele_<comp>` and `{name_base}_mu_<comp>`.
    is_correction : bool, default True
        If True, add only the central event-level weight. If False, do not add
        a central weight here; instead add one nuisance parameter per component,
        each with Up/Down variations stored as ratios to the central weight.

    Returns
    -------
    coffea.analysis_tools.Weights
        The modified Weights container. In correction mode, a weight named
        `{name_base}` is added. In systematics mode, one NP per component is
        added with names `{name_base}_ele_<comp>` and `{name_base}_mu_<comp>`.

    Notes
    -----
    - This routine relies on two helpers:
        * `electronSFs(..., return_jagged=True, variation in {"nominal","up","down"})`
        * `muonSFs(..., return_jagged=True, variation in {"nominal","up","down"})`
      which must return jagged per-lepton SFs for the requested variation.
    - MC efficiencies are approximated by average constants, taken from POG material.
      Replace these with maps if you have kinematic-dependent efficiencies.
    - Events with no leptons receive a weight of 1.0.
    - The OR logic is appropriate for categories that require at least one lepton.
      Do not also multiply per-lepton SF products independently in the same
      category, or you will double count.
    - Systematic variations:
        * Electron components: the function varies the electron ID WP component.
        * Muon components: the function varies each entry in `mu_SF_names`.
        * Each NP is added with central=1 and Up/Down equal to the ratio of the
          varied event weight to the central event weight computed here.
    """

    # we need MC efficiencies, these are functions of kinematics, but we dont have that easily accessible
    # so we use average values
    # electron effs from https://twiki.cern.ch/twiki/bin/view/CMS/CutBasedElectronIdentificationRun3
    # reco: https://cds.cern.ch/record/2747266/files/fulltext.pdf Fig. 4 (Run-2)
    def _ele_eff_from_wp(sf_key: str) -> float:
        m = {"wp90iso": 0.90, "wp80iso": 0.80, "Loose": 0.90, "Medium": 0.80, "Tight": 0.70, "Reco": 0.96}
        return m.get(sf_key, 0.90)

    # same for muons
    # reference: https://muon-wiki.docs.cern.ch/guidelines/corrections/#medium-pt-id-efficiencies
    def _mu_eff_from_wp(name: str) -> float:
        if "NUM_MediumID_DEN_TrackerMuons" in name:
            return 0.985  # between 0.98 and 0.99 mostly
        elif "NUM_TightID_DEN_TrackerMuons" in name:
            return 0.97
        elif "NUM_LooseID_DEN_TrackerMuons" in name:
            return 0.995
        # ISO (conditional on ID WPs)
        if "NUM_TightPFIso_DEN_MediumID" in name:
            return 0.95
        elif "NUM_LoosePFIso_DEN_MediumID" in name:
            return 0.97
        # other cases, just return a default value
        logger.warning(f"Muon SF name '{name}' not recognized, using default efficiency of 0.97.")
        return 0.97

    # Normalize to tuples if only one string is passed
    electron_component_names = (ele_SF_names if isinstance(ele_SF_names, (list, tuple))
                                else (ele_SF_names,))
    muon_component_names = (mu_SF_names if isinstance(mu_SF_names, (list, tuple))
                            else (mu_SF_names,))

    efficiency_ele = 1.0
    for name in electron_component_names:
        efficiency_ele *= _ele_eff_from_wp(name)
    efficiency_mu = 1.0
    for name in muon_component_names:
        efficiency_mu *= _mu_eff_from_wp(name)

    # for each lepton, multiply the components with these helper functions
    def _ele_total_sf(variation: str):
        sf_tot = None
        for name in electron_component_names:
            _sf = electronSFs(electrons, weights=None, year=year, sf_key=name, return_jagged=True, variation=variation)
            sf_tot = _sf if sf_tot is None else (sf_tot * _sf)
        return sf_tot if sf_tot is not None else ak.ones_like(electrons.pt)

    def _mu_total_sf(variation: str):
        sf_tot = None
        for name in muon_component_names:
            _sf = muonSFs(muons, weights=None, year=year, SF_name=name, return_jagged=True, variation=variation)
            sf_tot = _sf if sf_tot is None else (sf_tot * _sf)
        # if there are no components, fall back to ones
        return sf_tot if sf_tot is not None else ak.ones_like(muons.pt)

    el_sf_nom = _ele_total_sf("nominal")
    mu_sf_nom = _mu_total_sf("nominal")

    # build OR ratio
    # Denominator (MC): prod(1 - eps_MC_total) per flavour, then multiply flavours
    # ak.prod over jagged axis returns 1.0 for empty lists
    el_fail_mc = ak.prod(1.0 - efficiency_ele * ak.ones_like(el_sf_nom), axis=1)
    mu_fail_mc = ak.prod(1.0 - efficiency_mu * ak.ones_like(mu_sf_nom), axis=1)
    F_mc = el_fail_mc * mu_fail_mc

    # data: replace eps_MC by SF_total * eps_MC
    el_fail_data_nom = ak.prod(1.0 - el_sf_nom * efficiency_ele, axis=1)
    mu_fail_data_nom = ak.prod(1.0 - mu_sf_nom * efficiency_mu, axis=1)
    F_data_nom = el_fail_data_nom * mu_fail_data_nom

    denom = 1.0 - F_mc
    numer = 1.0 - F_data_nom
    eps_safe = 1e-12
    w_nom = numer / np.clip(denom, eps_safe, None)

    # Neutralize events with no leptons at all
    has_any_lep = (ak.num(electrons.pt) + ak.num(muons.pt)) > 0
    w_nom = np.where(ak.to_numpy(has_any_lep), ak.to_numpy(w_nom), 1.0)

    if is_correction:
        weights.add(name=f"{name_base}", weight=w_nom)
        return weights
    else:

        # Cache nominal per-component SFs (so we don't recompute them in each NP)
        electron_component_sfs_nominal = {}
        for comp_name in electron_component_names:
            electron_component_sfs_nominal[comp_name] = electronSFs(
                electrons, weights=None, year=year, sf_key=comp_name,
                return_jagged=True, variation="nominal"
            )

        muon_component_sfs_nominal = {}
        for comp_name in muon_component_names:
            muon_component_sfs_nominal[comp_name] = muonSFs(
                muons, weights=None, year=year, SF_name=comp_name,
                return_jagged=True, variation="nominal"
            )

        # Build a single list of (flavor, component_name) to vary
        components_to_vary = []
        for name in electron_component_names:
            components_to_vary.append(("ele", name))
        for name in muon_component_names:
            components_to_vary.append(("mu", name))

        # Safe arrays for ratios and neutralization
        w_nom_safe = np.clip(w_nom, eps_safe, None)
        no_lepton_mask = ~ak.to_numpy(has_any_lep)

        # Unified loop: vary ONE component at a time, others stay nominal
        for flavor, comp_name in components_to_vary:

            if flavor == "ele":
                # --- electron component up/down ---
                el_comp_up = electronSFs(
                    electrons, weights=None, year=year, sf_key=comp_name,
                    return_jagged=True, variation="up"
                )
                el_comp_dn = electronSFs(
                    electrons, weights=None, year=year, sf_key=comp_name,
                    return_jagged=True, variation="down"
                )

                # Total electron SF where ONLY this component is varied; others nominal
                el_total_sf_up = el_comp_up
                el_total_sf_dn = el_comp_dn
                for other in electron_component_names:
                    if other == comp_name:
                        continue
                    el_total_sf_up = el_total_sf_up * electron_component_sfs_nominal[other]
                    el_total_sf_dn = el_total_sf_dn * electron_component_sfs_nominal[other]

                # Event OR weight with electrons varied, muons fixed to nominal
                el_fail_data_up = ak.prod(1.0 - el_total_sf_up * efficiency_ele, axis=1)
                el_fail_data_dn = ak.prod(1.0 - el_total_sf_dn * efficiency_ele, axis=1)
                w_up = (1.0 - (el_fail_data_up * mu_fail_data_nom)) / np.clip(denom, eps_safe, None)
                w_dn = (1.0 - (el_fail_data_dn * mu_fail_data_nom)) / np.clip(denom, eps_safe, None)

            elif flavor == "mu":
                mu_comp_up = muonSFs(
                    muons, weights=None, year=year, SF_name=comp_name,
                    return_jagged=True, variation="up"
                )
                mu_comp_dn = muonSFs(
                    muons, weights=None, year=year, SF_name=comp_name,
                    return_jagged=True, variation="down"
                )

                # Total muon SF where ONLY this component is varied; others nominal
                mu_total_sf_up = mu_comp_up
                mu_total_sf_dn = mu_comp_dn
                for other in muon_component_names:
                    if other == comp_name:
                        continue
                    mu_total_sf_up = mu_total_sf_up * muon_component_sfs_nominal[other]
                    mu_total_sf_dn = mu_total_sf_dn * muon_component_sfs_nominal[other]

                # Event OR weight with muons varied, electrons fixed to nominal
                mu_fail_data_up = ak.prod(1.0 - mu_total_sf_up * efficiency_mu, axis=1)
                mu_fail_data_dn = ak.prod(1.0 - mu_total_sf_dn * efficiency_mu, axis=1)
                w_up = (1.0 - (el_fail_data_nom * mu_fail_data_up)) / np.clip(denom, eps_safe, None)
                w_dn = (1.0 - (el_fail_data_nom * mu_fail_data_dn)) / np.clip(denom, eps_safe, None)

            # Ratios and neutralization
            r_up = ak.to_numpy(w_up) / w_nom_safe
            r_dn = ak.to_numpy(w_dn) / w_nom_safe
            r_up[no_lepton_mask] = 1.0
            r_dn[no_lepton_mask] = 1.0

            weights.add(
                name=f"{name_base}_{flavor}_{comp_name}",
                weight=np.ones_like(w_nom),
                weightUp=r_up, weightDown=r_dn
            )

        return weights


def TriggerSF_LM(photons, weights, year="2022postEE", is_correction=True, **kwargs):
    """
    Lowmass Trigger SFs of Run3
    """

    # Currently only have Run3 2022 corrections available, for other Run3 years, using 2022 SFs temporarily
    avail_years = [
        "2022preEE",
        "2022postEE",
        "2023preBPix",
        "2023postBPix",
        "2024",
        "2025",
    ]
    if year not in avail_years:
        logger.warning(
            f"\n WARNING: only TriggerSF corrections for the year strings {avail_years} are already implemented! \n Exiting. \n"
        )
        exit()

    if year in [
        "2022preEE",
        "2022postEE",
        "2023preBPix",
        "2023postBPix",
        "2024",
        "2025",
    ]:
        if year not in ["2022preEE", "2022postEE"]:
            logger.warning(
                f"Using 2022postEE LowmassTrigger SFs for {year} as a placeholder until {year} SFs are available!"
            )
            _year = "2022postEE"
        else:
            _year = year
        json_file_lead = os.path.join(
            os.path.dirname(__file__),
            f"JSONs/TriggerSF_LM/{_year}/Lowmass_seeded_leg_efficinecy.json",
        )
        json_file_sublead = os.path.join(
            os.path.dirname(__file__),
            f"JSONs/TriggerSF_LM/{_year}/Lowmass_unseeded_leg_efficinecy.json",
        )

    evaluator_lead = correctionlib.CorrectionSet.from_file(json_file_lead)["TriggerSF"]
    evaluator_sublead = correctionlib.CorrectionSet.from_file(json_file_sublead)[
        "TriggerSF"
    ]

    if year in [
        "2022preEE",
        "2022postEE",
        "2023preBPix",
        "2023postBPix",
        "2024",
        "2025",
    ]:
        # If flow corrections are applied, we use the raw (uncorrected) r9 for the trigger SF evaluation
        if hasattr(photons["pho_lead"], "raw_r9"):
            sf_lead_p_lead = evaluator_lead.evaluate(
                "nominal",
                abs(photons["pho_lead"].ScEta),
                photons["pho_lead"].raw_r9,
                photons["pho_lead"].pt,
            )
            sf_lead_p_sublead = evaluator_lead.evaluate(
                "nominal",
                abs(photons["pho_sublead"].ScEta),
                photons["pho_sublead"].raw_r9,
                photons["pho_sublead"].pt,
            )
            sf_sublead_p_lead = evaluator_sublead.evaluate(
                "nominal",
                abs(photons["pho_lead"].ScEta),
                photons["pho_lead"].raw_r9,
                photons["pho_lead"].pt,
            )
            sf_sublead_p_sublead = evaluator_sublead.evaluate(
                "nominal",
                abs(photons["pho_sublead"].ScEta),
                photons["pho_sublead"].raw_r9,
                photons["pho_sublead"].pt,
            )
        else:
            sf_lead_p_lead = evaluator_lead.evaluate(
                "nominal",
                abs(photons["pho_lead"].ScEta),
                photons["pho_lead"].r9,
                photons["pho_lead"].pt,
            )
            sf_lead_p_sublead = evaluator_lead.evaluate(
                "nominal",
                abs(photons["pho_sublead"].ScEta),
                photons["pho_sublead"].r9,
                photons["pho_sublead"].pt,
            )
            sf_sublead_p_lead = evaluator_sublead.evaluate(
                "nominal",
                abs(photons["pho_lead"].ScEta),
                photons["pho_lead"].r9,
                photons["pho_lead"].pt,
            )
            sf_sublead_p_sublead = evaluator_sublead.evaluate(
                "nominal",
                abs(photons["pho_sublead"].ScEta),
                photons["pho_sublead"].r9,
                photons["pho_sublead"].pt,
            )

        if is_correction:
            # only calculate correction to nominal weight
            sf = (
                sf_lead_p_lead * sf_sublead_p_sublead
                + sf_lead_p_sublead * sf_sublead_p_lead
                - sf_lead_p_lead * sf_lead_p_sublead
            )

            sfup, sfdown = None, None

        else:
            # only calculate systs
            sf = np.ones(len(weights._weight))
            # get nominal SF to divide it out
            _sf = (
                sf_lead_p_lead * sf_sublead_p_sublead
                + sf_lead_p_sublead * sf_sublead_p_lead
                - sf_lead_p_lead * sf_lead_p_sublead
            )

            # up SF
            sfup_lead_p_lead = evaluator_lead.evaluate(
                "up",
                abs(photons["pho_lead"].ScEta),
                photons["pho_lead"].r9,
                photons["pho_lead"].pt,
            )
            sfup_lead_p_sublead = evaluator_lead.evaluate(
                "up",
                abs(photons["pho_sublead"].ScEta),
                photons["pho_sublead"].r9,
                photons["pho_sublead"].pt,
            )
            sfup_sublead_p_lead = evaluator_sublead.evaluate(
                "up",
                abs(photons["pho_lead"].ScEta),
                photons["pho_lead"].r9,
                photons["pho_lead"].pt,
            )
            sfup_sublead_p_sublead = evaluator_sublead.evaluate(
                "up",
                abs(photons["pho_sublead"].ScEta),
                photons["pho_sublead"].r9,
                photons["pho_sublead"].pt,
            )
            sfup = (
                sfup_lead_p_lead * sfup_sublead_p_sublead
                + sfup_lead_p_sublead * sfup_sublead_p_lead
                - sfup_lead_p_lead * sfup_lead_p_sublead
            ) / _sf

            # down SF
            sfdown_lead_p_lead = evaluator_lead.evaluate(
                "down",
                abs(photons["pho_lead"].ScEta),
                photons["pho_lead"].r9,
                photons["pho_lead"].pt,
            )
            sfdown_lead_p_sublead = evaluator_lead.evaluate(
                "down",
                abs(photons["pho_sublead"].ScEta),
                photons["pho_sublead"].r9,
                photons["pho_sublead"].pt,
            )
            sfdown_sublead_p_lead = evaluator_sublead.evaluate(
                "down",
                abs(photons["pho_lead"].ScEta),
                photons["pho_lead"].r9,
                photons["pho_lead"].pt,
            )
            sfdown_sublead_p_sublead = evaluator_sublead.evaluate(
                "down",
                abs(photons["pho_sublead"].ScEta),
                photons["pho_sublead"].r9,
                photons["pho_sublead"].pt,
            )
            sfdown = (
                sfdown_lead_p_lead * sfdown_sublead_p_sublead
                + sfdown_lead_p_sublead * sfdown_sublead_p_lead
                - sfdown_lead_p_lead * sfdown_lead_p_sublead
            ) / _sf

    name = "TriggerSF_corr" if is_correction else "TriggerSF"
    weights.add(name=name, weight=sf, weightUp=sfup, weightDown=sfdown)

    return weights


def SF_photon_ID_LM(photons, weights, year="2022postEE", is_correction=True, **kwargs):
    """
    Run3 Lowmass photon ID SFs. Only have 2022 SFs available for now,
    using them as placeholders for other years until the official SFs are available
    """
    # era/year defined as parameter of the function
    avail_years = ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024"]
    if year not in avail_years:
        logger.warning(
            f"\n WARNING: only photon ID SFs for the year strings {avail_years} are already implemented! \n Exiting. \n"
        )
        logger.warning(
            "If you need the SFs for the central Egamma MVA ID for Run 2 UL, take action yourself or contact us!"
        )
        exit()

    if year == "2022preEE":
        json_file = os.path.join(
            os.path.dirname(__file__),
            "JSONs/SF_photon_ID_LM/2022/IDMVA0p9_2022PreEE_LowHgg.json",
        )
    elif year == "2022postEE":
        json_file = os.path.join(
            os.path.dirname(__file__),
            "JSONs/SF_photon_ID_LM/2022/IDMVA0p9_2022PostEE_LowHgg.json",
        )
    # preliminary 2023, 2024 results, has to be changed once the official SFs are available
    elif year == "2023preBPix":
        json_file = os.path.join(
            os.path.dirname(__file__),
            "JSONs/SF_photon_ID_LM/2022/IDMVA0p9_2022PostEE_LowHgg.json",
        )
        logger.warning(
            f"Using 2022postEE Lowmass PhotonID SFs for {year} as a placeholder until {year} SFs are available!"
        )
    elif year == "2023postBPix":
        json_file = os.path.join(
            os.path.dirname(__file__),
            "JSONs/SF_photon_ID_LM/2022/IDMVA0p9_2022PostEE_LowHgg.json",
        )
        logger.warning(
            f"Using 2022postEE Lowmass PhotonID SFs for {year} as a placeholder until {year} SFs are available!"
        )
    elif year == "2024":
        json_file = os.path.join(
            os.path.dirname(__file__),
            "JSONs/SF_photon_ID_LM/2022/IDMVA0p9_2022PostEE_LowHgg.json",
        )
        logger.warning(
            f"Using 2022postEE Lowmass PhotonID SFs for {year} as a placeholder until {year} SFs are available!"
        )

    evaluator = correctionlib.CorrectionSet.from_file(json_file)["IDMVA_SF"]

    # In principle, we should use the fully correct formula https://indico.cern.ch/event/1360948/contributions/5783762/attachments/2788516/4870824/24_02_02_HIG-23-014_PreAppPres.pdf#page=7
    # However, if the SF is pt-binned, the approximation of the multiplication of the two SFs is fully exact
    # N.B. These phoID SFs are computed for the workin point optimised for the fiducial XS analysis (0.25 for 22, and 0.19 for 23)
    if "2022" in year or "2023" in year or "2024" in year:
        if is_correction:
            # only calculate correction to nominal weight
            sf_lead = evaluator.evaluate(
                abs(photons["pho_lead"].ScEta),
                photons["pho_lead"].r9,
                photons["pho_lead"].pt,
                "nominal",
            )
            sf_sublead = evaluator.evaluate(
                abs(photons["pho_sublead"].ScEta),
                photons["pho_sublead"].r9,
                photons["pho_sublead"].pt,
                "nominal",
            )
            sf = sf_lead * sf_sublead

            sfup, sfdown = None, None

        else:
            # only calculate systs

            sf = np.ones(len(weights._weight))
            sf_lead = evaluator.evaluate(
                abs(photons["pho_lead"].ScEta),
                photons["pho_lead"].r9,
                photons["pho_lead"].pt,
                "nominal",
            )
            sf_sublead = evaluator.evaluate(
                abs(photons["pho_sublead"].ScEta),
                photons["pho_sublead"].r9,
                photons["pho_sublead"].pt,
                "nominal",
            )
            _sf = sf_lead * sf_sublead

            sf_unc_lead = evaluator.evaluate(
                abs(photons["pho_lead"].ScEta),
                photons["pho_lead"].r9,
                photons["pho_lead"].pt,
                "uncertainty",
            )
            sf_unc_sublead = evaluator.evaluate(
                abs(photons["pho_sublead"].ScEta),
                photons["pho_sublead"].r9,
                photons["pho_sublead"].pt,
                "uncertainty",
            )

            sfup = (sf_lead + sf_unc_lead) * (sf_sublead + sf_unc_sublead) / _sf

            sfdown = (sf_lead - sf_unc_lead) * (sf_sublead - sf_unc_sublead) / _sf

    name = "SF_photon_ID_corr" if is_correction else "SF_photon_ID"
    weights.add(name=name, weight=sf, weightUp=sfup, weightDown=sfdown)

    return weights


def PreselSF_LM(photons, weights, year="2022postEE", is_correction=True, **kwargs):
    """
    Run3 Lowmass preselection SFs. Only have 2022 SFs available for now
    Using them as placeholders for other years until the official SFs are available
    """
    # era/year defined as parameter of the function
    avail_years = ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024"]
    if year not in avail_years:
        logger.warning(
            f"\n WARNING: only preselection SFs for the year strings {avail_years} are already implemented! \n Exiting. \n"
        )
        exit()

    if year == "2022preEE":
        json_file = os.path.join(
            os.path.dirname(__file__),
            "JSONs/Preselection_LM/2022/Preselection_2022PreEE_LowHgg.json",
        )
    elif year == "2022postEE":
        json_file = os.path.join(
            os.path.dirname(__file__),
            "JSONs/Preselection_LM/2022/Preselection_2022PostEE_LowHgg.json",
        )
    # preliminary 2023, 2024 results, has to be changed once the official SFs are available
    elif year == "2023preBPix":
        json_file = os.path.join(
            os.path.dirname(__file__),
            "JSONs/Preselection_LM/2022/Preselection_2022PostEE_LowHgg.json",
        )
        logger.warning(
            f"Using 2022postEE Lowmass Preselection SFs for {year} as a placeholder until {year} SFs are available!"
        )
    elif year == "2023postBPix":
        json_file = os.path.join(
            os.path.dirname(__file__),
            "JSONs/Preselection_LM/2022/Preselection_2022PostEE_LowHgg.json",
        )
        logger.warning(
            f"Using 2022postEE Lowmass Preselection SFs for {year} as a placeholder until {year} SFs are available!"
        )
    elif year == "2024":
        json_file = os.path.join(
            os.path.dirname(__file__),
            "JSONs/Preselection_LM/2022/Preselection_2022PostEE_LowHgg.json",
        )
        logger.warning(
            f"Using 2022postEE Lowmass Preselection SFs for {year} as a placeholder until {year} SFs are available!"
        )

    evaluator = correctionlib.CorrectionSet.from_file(json_file)["Preselection_SF"]

    if is_correction:
        # only calculate correction to nominal weight
        sf_lead = evaluator.evaluate(
            abs(photons["pho_lead"].ScEta),
            photons["pho_lead"].r9,
            photons["pho_lead"].pt,
            "nominal",
        )
        sf_sublead = evaluator.evaluate(
            abs(photons["pho_sublead"].ScEta),
            photons["pho_sublead"].r9,
            photons["pho_sublead"].pt,
            "nominal",
        )
        sf = sf_lead * sf_sublead

        sfup, sfdown = None, None
    else:
        # only calculate systs

        sf = np.ones(len(weights._weight))
        sf_lead = evaluator.evaluate(
            abs(photons["pho_lead"].ScEta),
            photons["pho_lead"].r9,
            photons["pho_lead"].pt,
            "nominal",
        )
        sf_sublead = evaluator.evaluate(
            abs(photons["pho_sublead"].ScEta),
            photons["pho_sublead"].r9,
            photons["pho_sublead"].pt,
            "nominal",
        )
        _sf = sf_lead * sf_sublead

        sf_unc_lead = evaluator.evaluate(
            abs(photons["pho_lead"].ScEta),
            photons["pho_lead"].r9,
            photons["pho_lead"].pt,
            "uncertainty",
        )
        sf_unc_sublead = evaluator.evaluate(
            abs(photons["pho_sublead"].ScEta),
            photons["pho_sublead"].r9,
            photons["pho_sublead"].pt,
            "uncertainty",
        )

        sfup = (sf_lead + sf_unc_lead) * (sf_sublead + sf_unc_sublead) / _sf

        sfdown = (sf_lead - sf_unc_lead) * (sf_sublead - sf_unc_sublead) / _sf

    name = "PreselSF_corr" if is_correction else "PreselSF"
    weights.add(name=name, weight=sf, weightUp=sfup, weightDown=sfdown)

    return weights


def ElectronIndexSF_LM(
    photons, weights, year="2022postEE", is_correction=True, **kwargs
):
    """
    Run3 Lowmass Electron Index Selection SFs. Only have 2022 SFs available for now
    Using them as placeholders for other years until the official SFs are available
    """

    # era/year defined as parameter of the function
    avail_years = ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix"]
    if year not in avail_years:
        logger.warning(
            f"\n WARNING: only eVetoSF corrections for the year strings {avail_years} are already implemented! \n Exiting. \n"
        )
        exit()

    if year == "2022preEE":
        json_file = os.path.join(
            os.path.dirname(__file__),
            "JSONs/ElectronIdxSF_LM/2022/preEE_EIdx_SFcorrections.json",
        )
    elif year == "2022postEE":
        json_file = os.path.join(
            os.path.dirname(__file__),
            "JSONs/ElectronIdxSF_LM/2022/postEE_EIdx_SFcorrections.json",
        )
    elif "2023preBPix" in year:
        json_file = os.path.join(
            os.path.dirname(__file__),
            "JSONs/ElectronIdxSF_LM/2022/postEE_EIdx_SFcorrections.json",
        )
        logger.warning(
            f"Using 2022postEE Lowmass Electron Index SFs for {year} as a placeholder until {year} SFs are available!"
        )
    elif "2023postBPix" in year:
        json_file = os.path.join(
            os.path.dirname(__file__),
            "JSONs/ElectronIdxSF_LM/2022/postEE_EIdx_SFcorrections.json",
        )
        logger.warning(
            f"Using 2022postEE Lowmass Electron Index SFs for {year} as a placeholder until {year} SFs are available!"
        )

    evaluator = correctionlib.CorrectionSet.from_file(json_file)["EIdx_SFs"]

    if is_correction:
        # only calculate correction to nominal weight
        sf_lead = evaluator.evaluate(
            abs(photons["pho_lead"].ScEta), photons["pho_lead"].r9, "nominal"
        )
        sf_sublead = evaluator.evaluate(
            abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].r9, "nominal"
        )
        sf = sf_lead * sf_sublead

        sfup, sfdown = None, None

    else:
        # only calculate systs
        sf = np.ones(len(weights._weight))
        sf_lead = evaluator.evaluate(
            abs(photons["pho_lead"].ScEta), photons["pho_lead"].r9, "nominal"
        )
        sf_sublead = evaluator.evaluate(
            abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].r9, "nominal"
        )
        _sf = sf_lead * sf_sublead

        unc_lead = evaluator.evaluate(
            abs(photons["pho_lead"].ScEta), photons["pho_lead"].r9, "uncertainty"
        )
        unc_sublead = evaluator.evaluate(
            abs(photons["pho_sublead"].ScEta),
            photons["pho_sublead"].r9,
            "uncertainty",
        )

        sfup = (sf_lead + unc_lead) * (sf_sublead + unc_sublead) / _sf
        sfdown = (sf_lead - unc_lead) * (sf_sublead - unc_sublead) / _sf

    name = "ElectronIndexSF_corr" if is_correction else "ElectronIndexSF"
    weights.add(name=name, weight=sf, weightUp=sfup, weightDown=sfdown)

    return weights


def ElectronVetoSF_LM(
    photons, weights, year="2022postEE", is_correction=True, **kwargs
):
    """
    Run3 Lowmass Electron Veto Selection SFs. Only have 2022 SFs available for now
    Using them as placeholders for other years until the official SFs are available
    """

    # era/year defined as parameter of the function
    avail_years = [
        "2022preEE",
        "2022postEE",
        "2023preBPix",
        "2023postBPix",
    ]
    if year not in avail_years:
        logger.warning(
            f"\n WARNING: only eVetoSF corrections for the year strings {avail_years} are already implemented! \n Exiting. \n"
        )
        exit()

    # presentation of SF: https://indico.cern.ch/event/1360961/#173-run-3-electron-veto-sfs
    if year == "2022preEE":
        json_file = os.path.join(
            os.path.dirname(__file__),
            "JSONs/ElectronVetoSF_LM/2022/preEE_PSV_SFcorrections.json",
        )
    elif year == "2022postEE":
        json_file = os.path.join(
            os.path.dirname(__file__),
            "JSONs/ElectronVetoSF_LM/2022/postEE_PSV_SFcorrections.json",
        )
    else:
        json_file = os.path.join(
            os.path.dirname(__file__),
            "JSONs/ElectronVetoSF_LM/2022/postEE_PSV_SFcorrections.json",
        )
        logger.warning(
            f"Using 2022postEE Lowmass Electron Veto SFs for {year} as a placeholder until {year} SFs are available!"
        )

    evaluator = correctionlib.CorrectionSet.from_file(json_file)["PSV_SFs"]

    if is_correction:
        # only calculate correction to nominal weight
        sf_lead = evaluator.evaluate(
            abs(photons["pho_lead"].ScEta), photons["pho_lead"].r9, "nominal"
        )
        sf_sublead = evaluator.evaluate(
            abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].r9, "nominal"
        )
        sf = sf_lead * sf_sublead

        sfup, sfdown = None, None

    else:
        # only calculate systs
        sf = np.ones(len(weights._weight))
        sf_lead = evaluator.evaluate(
            abs(photons["pho_lead"].ScEta), photons["pho_lead"].r9, "nominal"
        )
        sf_sublead = evaluator.evaluate(
            abs(photons["pho_sublead"].ScEta), photons["pho_sublead"].r9, "nominal"
        )
        _sf = sf_lead * sf_sublead

        unc_lead = evaluator.evaluate(
            abs(photons["pho_lead"].ScEta), photons["pho_lead"].r9, "uncertainty"
        )
        unc_sublead = evaluator.evaluate(
            abs(photons["pho_sublead"].ScEta),
            photons["pho_sublead"].r9,
            "uncertainty",
        )

        sfup = (sf_lead + unc_lead) * (sf_sublead + unc_sublead) / _sf
        sfdown = (sf_lead - unc_lead) * (sf_sublead - unc_sublead) / _sf

    name = "ElectronVetoSF_corr" if is_correction else "ElectronVetoSF"
    weights.add(name=name, weight=sf, weightUp=sfup, weightDown=sfdown)

    return weights


def Tau_ID(taus, weights, year, is_correction=True, wp_VSjet="Medium", wp_VSe="VVLoose", **kwargs):
    year_mapping = {
        "2022preEE": "2022_Summer22",
        "2022postEE": "2022_Summer22EE",
        "2023preBPix": "2023_Summer23",
        "2023postBPix": "2023_Summer23BPix",
        "2024": "2024_Summer24",
    }
    folder_name = year_mapping.get(year)
    json_name = "DeepTau2018v2p5VSjet"
    if folder_name and json_name:
        path_to_json = os.path.join(os.path.dirname(__file__), f"JSONs/POG/TAU/{folder_name}/tau.json.gz")
        evaluator = correctionlib.CorrectionSet.from_file(path_to_json)[json_name]
    else:
        raise ValueError(f"Year {year} not supported for TAU ID SFs.")

    era_syst_map = {
        "2022preEE": ("syst_2022_preEE_up", "syst_2022_preEE_down"),
        "2022postEE": ("syst_2022_postEE_up", "syst_2022_postEE_down"),
        "2023preBPix": ("syst_2023_preBPix_up", "syst_2023_preBPix_down"),
        "2023postBPix": ("syst_2023_postBPix_up", "syst_2023_postBPix_down"),
        "2024": ("up", "down"),
    }
    syst_up_str, syst_down_str = era_syst_map[year]

    counts = ak.num(taus)
    taus_flat = ak.flatten(taus)

    pt = ak.to_numpy(taus_flat.pt)
    dm = ak.to_numpy(taus_flat.decayMode)
    genPartFlav = ak.to_numpy(taus_flat.genPartFlav)

    sf_nom_flat = evaluator.evaluate(pt, dm, genPartFlav, wp_VSjet, wp_VSe, "nom", "dm")

    if is_correction:
        sf = ak.to_numpy(ak.prod(ak.unflatten(sf_nom_flat, counts), axis=1))
        sfup, sfdown = None, None
    else:
        sf_up_flat = evaluator.evaluate(pt, dm, genPartFlav, wp_VSjet, wp_VSe, syst_up_str, "dm")
        sf_down_flat = evaluator.evaluate(pt, dm, genPartFlav, wp_VSjet, wp_VSe, syst_down_str, "dm")

        prod_nom = ak.to_numpy(ak.prod(ak.unflatten(sf_nom_flat, counts), axis=1))
        prod_up = ak.to_numpy(ak.prod(ak.unflatten(sf_up_flat, counts), axis=1))
        prod_down = ak.to_numpy(ak.prod(ak.unflatten(sf_down_flat, counts), axis=1))

        sf = np.ones(len(weights._weight))
        sfup = prod_up / prod_nom
        sfdown = prod_down / prod_nom

    name = "Tau_ID_corr" if is_correction else "Tau_ID"
    weights.add(name=name, weight=sf, weightUp=sfup, weightDown=sfdown)

    return weights


def leptonTriggerDataEff(
    electrons,
    muons,
    weights,
    year,
    is_correction=True,
    **kwargs,
):
    """
    Apply single-lepton trigger DATA efficiency to MC.

    The MC sample has not had the HLT applied, so this function
    applies the DATA trigger efficiency directly.

    Triggers
    --------
    Electron:
        Ele30_tightgsf

        Correction:
            Electron-HLT-DataEff

        Working point:
            HLT_SF_Ele30_MVAiso80ID

        Variations:
            nominal -> nom
            up      -> up
            down    -> down

    Muon:
        IsoMu24

        Correction:
            NUM_IsoMu24_DEN_CutBasedIdMedium_and_PFIsoMedium

        Variations:
            nominal -> nominal_DATAeff
            up      -> systup_DATAeff
            down    -> systdown_DATAeff

    Since exactly one electron OR one muon is present in every event,
    the corresponding trigger efficiency is used.

    Parameters
    ----------
    electrons : ak.Array
        Electron collection. Needs pt and eta.

    muons : ak.Array
        Muon collection. Needs pt and eta.

    weights : coffea.analysis_tools.Weights
        Weights object to modify.

    year : str
        Analysis year:
            "2022preEE"
            "2022postEE"
            "2023preBPix"
            "2023postBPix"
            "2024"

    is_correction : bool
        If True:
            Add nominal DATA efficiency directly.

        If False:
            Add nominal weight = 1 and systematic variations
            as ratios to the nominal efficiency.
    """

    # ============================================================
    # Check that exactly one lepton is present
    # ============================================================

    n_ele = ak.num(electrons, axis=1)
    n_mu = ak.num(muons, axis=1)

    # is_ele = n_ele == 1
    # is_mu = n_mu == 1

    # if not ak.all(is_ele | is_mu):
    #     raise ValueError(
    #         "leptonTriggerDataEff requires exactly one electron "
    #         "OR exactly one muon in every event."
    #     )

    # if ak.any(is_ele & is_mu):
    #     raise ValueError(
    #         "Found events containing both one electron and one muon."
    #     )

    # ============================================================
    # Check lepton multiplicity
    # ============================================================

    is_ele = (n_ele == 1) & (n_mu == 0)
    is_mu = (n_mu == 1) & (n_ele == 0)

    valid_event = is_ele | is_mu
    invalid_event = ~valid_event

    n_invalid = int(ak.sum(invalid_event))

    if n_invalid > 0:
        logger.warning(
            f"leptonTriggerDataEff: Found {n_invalid} events "
            "that do not contain exactly one electron and zero muons, "
            "or exactly one muon and zero electrons. "
            "Their trigger efficiency will be set to 1.0."
        )

    # ============================================================
    # Load correction files
    # ============================================================

    avail_years = ["2024"]

    if year not in avail_years:
        logger.warning(
            f"\n WARNING: only single-lepton TriggerSF corrections "
            f"for the year strings {avail_years} are currently implemented! \n"
            f" Exiting. \n"
        )
        exit()

    json_file_electron = os.path.join(
        os.path.dirname(__file__),
        f"JSONs/TriggerSF/{year}/electronHlt.json"
    )

    json_file_muon = os.path.join(
        os.path.dirname(__file__),
        f"JSONs/TriggerSF/{year}/muon_Z.json"
    )

    electron_cset = correctionlib.CorrectionSet.from_file(
        json_file_electron
    )

    muon_cset = correctionlib.CorrectionSet.from_file(
        json_file_muon
    )

    electron_evaluator = electron_cset[
        "Electron-HLT-DataEff"
    ]

    muon_evaluator = muon_cset[
        "NUM_IsoMu24_DEN_CutBasedIdMedium_and_PFIsoMedium"
    ]

    # ============================================================
    # Electron era
    # ============================================================

    electron_era = {
        "2022preEE": "2022Re-recoBCD",
        "2022postEE": "2022Re-recoE+PromptFG",
        "2023preBPix": "2023PromptC",
        "2023postBPix": "2023PromptD",
        "2024": "2024Prompt",
    }[year]

    # ============================================================
    # Flatten electron inputs
    # ============================================================

    ele_eta = ak.flatten(electrons.eta)
    ele_pt = ak.flatten(electrons.pt)

    # ============================================================
    # Flatten muon inputs
    # ============================================================

    mu_eta = ak.flatten(muons.eta)
    mu_pt = ak.flatten(muons.pt)

    # ============================================================
    # ------------------------------------------------------------
    # Nominal electron DATA efficiency
    # ------------------------------------------------------------
    # ============================================================

    ele_nom_flat = electron_evaluator.evaluate(
        electron_era,
        "nom",
        "HLT_SF_Ele30_MVAiso80ID",
        ele_eta,
        ele_pt,
    )

    ele_nom = ak.unflatten(
        ele_nom_flat,
        n_ele,
    )

    # ============================================================
    # ------------------------------------------------------------
    # Nominal muon DATA efficiency
    # ------------------------------------------------------------
    # ============================================================

    mu_nom_flat = muon_evaluator.evaluate(
        mu_eta,
        mu_pt,
        "nominal_DATAeff",
    )

    mu_nom = ak.unflatten(
        mu_nom_flat,
        n_mu,
    )

    # ============================================================
    # Combine electron and muon nominal efficiencies
    # ============================================================

    # trigger_nom = ak.where(
    #     is_ele,
    #     ak.firsts(ele_nom),
    #     ak.firsts(mu_nom),
    # )

    trigger_nom = ak.where(valid_event,ak.where(is_ele,ak.firsts(ele_nom),ak.firsts(mu_nom)), 1.0)

    # ============================================================
    # Central correction
    # ============================================================

    if is_correction:

        weights.add(
            name="LeptonTriggerDataEff_corr",
            weight=trigger_nom,
            weightUp=None,
            weightDown=None,
        )

        return weights

    # ============================================================
    # ------------------------------------------------------------
    # Electron systematic variations
    # ------------------------------------------------------------
    # ============================================================

    ele_up_flat = electron_evaluator.evaluate(
        electron_era,
        "up",
        "HLT_SF_Ele30_MVAiso80ID",
        ele_eta,
        ele_pt,
    )

    ele_down_flat = electron_evaluator.evaluate(
        electron_era,
        "down",
        "HLT_SF_Ele30_MVAiso80ID",
        ele_eta,
        ele_pt,
    )

    ele_up = ak.unflatten(
        ele_up_flat,
        n_ele,
    )

    ele_down = ak.unflatten(
        ele_down_flat,
        n_ele,
    )

    # ============================================================
    # ------------------------------------------------------------
    # Muon systematic variations
    # ------------------------------------------------------------
    # ============================================================

    mu_up_flat = muon_evaluator.evaluate(
        mu_eta,
        mu_pt,
        "systup_DATAeff",
    )

    mu_down_flat = muon_evaluator.evaluate(
        mu_eta,
        mu_pt,
        "systdown_DATAeff",
    )

    mu_up = ak.unflatten(
        mu_up_flat,
        n_mu,
    )

    mu_down = ak.unflatten(
        mu_down_flat,
        n_mu,
    )

    # ============================================================
    # Combine electron/muon variations
    # ============================================================

    # trigger_up = ak.where(
    #     is_ele,
    #     ak.firsts(ele_up),
    #     ak.firsts(mu_up),
    # )

    # trigger_down = ak.where(
    #     is_ele,
    #     ak.firsts(ele_down),
    #     ak.firsts(mu_down),
    # )

    trigger_up = ak.where(valid_event,ak.where(is_ele,ak.firsts(ele_up),ak.firsts(mu_up)),1.0)
    trigger_down = ak.where(valid_event,ak.where(is_ele,ak.firsts(ele_down),ak.firsts(mu_down)),1.0)
    # ============================================================
    # Convert variations to ratios
    #
    # Same convention as your Pileup() function:
    #
    #     nominal = 1
    #     up       = efficiency_up / efficiency_nominal
    #     down     = efficiency_down / efficiency_nominal
    # ============================================================

    trigger_up_ratio = trigger_up / trigger_nom
    trigger_down_ratio = trigger_down / trigger_nom

    # ============================================================
    # Add systematic weight
    # ============================================================

    weights.add(
        name="LeptonTriggerDataEff",
        weight=np.ones(len(trigger_nom)),
        weightUp=trigger_up_ratio,
        weightDown=trigger_down_ratio,
    )

    return weights
