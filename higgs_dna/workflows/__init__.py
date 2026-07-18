from higgs_dna.workflows.base import HggBaseProcessor
from higgs_dna.workflows.fiducial import HggFiducialProcessor
from higgs_dna.workflows.dystudies import TagAndProbeProcessor
from higgs_dna.workflows.HHbbgg import HHbbggProcessor
from higgs_dna.workflows.particleLevel import ParticleLevelProcessor
from higgs_dna.workflows.top import TopProcessor
from higgs_dna.workflows.Zmmy import ZmmyProcessor, ZmmyHist, ZmmyZptHist
from higgs_dna.workflows.hpc_processor import HplusCharmProcessor
from higgs_dna.workflows.zee_processor import ZeeProcessor
from higgs_dna.workflows.lowmass import LowMassProcessor
from higgs_dna.workflows.btagging import BTaggingEfficienciesProcessor, BTaggingEfficienciesHHbbggProcessor
from higgs_dna.workflows.stxs import STXSProcessor
from higgs_dna.workflows.diphoton_training import DiphoTrainingProcessor
from higgs_dna.workflows.My_first_processor_WH_Cat1_HDNA_presel import My_first_processor_WH_Cat1_HDNA_presel
from higgs_dna.workflows.My_first_processor_WH_Cat1_HDNA_presel_test import My_first_processor_WH_Cat1_HDNA_presel_test

from higgs_dna.workflows.taggers import taggers

workflows = {}

workflows["base"] = HggBaseProcessor
workflows["fiducial"] = HggFiducialProcessor
workflows["tagandprobe"] = TagAndProbeProcessor
workflows["HHbbgg"] = HHbbggProcessor
workflows["particleLevel"] = ParticleLevelProcessor
workflows["top"] = TopProcessor
workflows["zmmy"] = ZmmyProcessor
workflows["zmmyHist"] = ZmmyHist
workflows["zmmyZptHist"] = ZmmyZptHist
workflows["hpc"] = HplusCharmProcessor
workflows["zee"] = ZeeProcessor
workflows["lowmass"] = LowMassProcessor
workflows["BTagging"] = BTaggingEfficienciesProcessor
workflows["BTaggingHHbbgg"] = BTaggingEfficienciesHHbbggProcessor
workflows["stxs"] = STXSProcessor
workflows["diphotonID"] = DiphoTrainingProcessor
workflows["My_first_processor_WH_Cat1_HDNA_presel"] = My_first_processor_WH_Cat1_HDNA_presel
workflows["My_first_processor_WH_Cat1_HDNA_presel_test"] = My_first_processor_WH_Cat1_HDNA_presel_test

__all__ = ["workflows", "taggers"]
