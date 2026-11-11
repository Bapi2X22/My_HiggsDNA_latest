#!/usr/bin/env python
# Author Tiziano Bevilacqua (03/03/2023) and Nico Haerringer (16/07/2024)
import os
import sys
import subprocess
from optparse import OptionParser
import glob
import json
from importlib import resources
from higgs_dna.utils.logger_utils import setup_logger
from higgs_dna.scripts.postprocessing.remote.slurm import slurm_postprocessing
from higgs_dna.scripts.postprocessing.remote.htcondor import htcondor_postprocessing
from concurrent.futures import ThreadPoolExecutor
import yaml

from pathlib import Path

# ---------------------- A few helping functions  ----------------------


def MKDIRP(dirpath, verbose=False, dry_run=False):
    if verbose:
        print("\033[1m" + ">" + "\033[0m" + ' os.mkdirs("' + dirpath + '")')
    if dry_run:
        return
    try:
        os.makedirs(dirpath)
    except OSError:
        if not os.path.isdir(dirpath):
            raise
    return


# function to activate the (pre-existing) FlashggFinalFit environment and use Tree2WS scripts
# full Power Ranger style :D
def activate_final_fit(path, command):
    current_path = os.getcwd()
    os.chdir(path)
    os.system(
        f"eval `scram runtime -sh` && source {path}/flashggFinalFit/setup.sh && cd {path}/flashggFinalFit/Trees2WS && {command} "
    )
    os.chdir(current_path)


def decompose_string(input_string, process_map, era_flag=False):
    """
    Decomposes the input string into process, mass, and era components based on underscores.

    Args:
        input_string (str): The string to be decomposed.
        era_flag (bool): If True, include the era in the output. If False, exclude the era.

    Returns:
        str: The formatted string in the style process + _ + mass (+ _ + era if era_flag is True).
    """

    parts = input_string.split("_")

    # Extract the process by matching known keywords
    process = "unknown"
    for key, value in process_map.items():
        if key in parts[0]:
            process = value
            break

    # Extract the mass component
    mass = next((part[2:] for part in parts if part.startswith("M-") and part[2:].isdigit()), "")

    # Find the era (if it exists) and strip the year if included.
    era = ""
    for part in parts:
        if "pre" in part or "post" in part:
            era = part
            if part[:4].isdigit():
                era = part[4:]
            break

    # Assemble the output.
    if mass:
        if era_flag and era:
            return f"{process}_{mass}_{era}"
        elif era_flag:
            return f"{process}_{mass}"
        else:
            return f"{process}_{mass}"
    else:
        if era_flag and era:
            return f"{process}_{era}"
        else:
            return process


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# - EXAMPLE USAGE: ----------------------------------------------------------------------------------------------------------------------------------------------------------------#
# - prepare_output_file --input <dir_to_HiggsDNA_dump> --merge --varDict <path_to_varDict> --root --syst --cats --catDict <path_to_catDict> --output <path_to_output_dir>
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def main():
    # Read options from command line
    usage = "Usage: python %prog filelists [options]"
    parser = OptionParser(usage=usage)
    parser.add_option("--input", dest="input", type="string", default="", help="input dir")
    parser.add_option(
        "--merge",
        dest="merge",
        action="store_true",
        default=False,
        help="Do merging of the .parquet files",
    )
    parser.add_option(
        "--varDict",
        dest="varDict",
        default=None,
        help="Path to JSON that holds dictionary that encodes the mapping of the systematic variation branches (includes nominal and object-based systematics, up/down). If not provided, use only nominal.",
    )
    parser.add_option(
        "--root",
        dest="root",
        action="store_true",
        default=False,
        help="Do root conversion step",
    )
    parser.add_option(
        "--ws",
        dest="ws",
        action="store_true",
        default=False,
        help="Do root to workspace conversion step",
    )
    parser.add_option(
        "--ws-config",
        dest="config",
        type="string",
        default="config_simple.py",
        help="configuration file for Tree2WS, as it is now it must be stored in Tree2WS directory in FinalFit",
    )
    parser.add_option(
        "--final-fit",
        dest="final_fit",
        type="string",
        default="/afs/cern.ch/user/n/niharrin/cernbox/PhD/Higgs/CMSSW_10_2_13/src/",
        help="FlashggFinalFit path",
    )  # the default is just for me, it should be changed but I don't see a way to make this generally valid
    parser.add_option(
        "--syst",
        dest="syst",
        action="store_true",
        default=False,
        help="Do systematics variation treatment",
    )
    parser.add_option(
        "--cats",
        dest="cats",
        action="store_true",
        default=False,
        help="Split into categories.",
    )
    parser.add_option(
        "--catDict",
        dest="catDict",
        default=None,
        help="Path to JSON that defines the conditions for splitting into multiple categories. For well-defined statistical analyses in final fits, the categories should be mutually exclusive (it is your job to ensure this!). If not provided, use only one inclusive untagged category with no conditions.",
    )
    parser.add_option(
        "--genBinning",
        dest="genBinning",
        default="",
        help="Optional: Path to the JSON containing the binning at gen-level.",
    )
    parser.add_option(
        "--diff-variable",
        dest="diff_variable",
        default="",
        help="Optional: Differential variable key to select from genBinning (e.g. 'PTH').",
    )
    parser.add_option(
        "--skip-normalisation",
        dest="skip_normalisation",
        action="store_true",
        default=False,
        help="Independent of file type, skip normalisation step",
    )
    parser.add_option(
        "--args",
        dest="args",
        type="string",
        default="",
        help="additional options for root converter: --notag",
    )
    parser.add_option(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Debugging verbosity for logger.",
    )
    parser.add_option(
        "--output",
        dest="output",
        type="string",
        default="",
        help="Output path for the merged and ROOT files.",
    )
    parser.add_option(
        "--folder-structure",
        dest="folder_structure",
        type="string",
        default="",
        help="Uses the given folder structure for dirlist.",
    )
    parser.add_option(
        "--merge-data-only",
        dest="merge_data",
        action="store_true",
        default=False,
        help="Flag for merging data to an allData file.",
    )
    parser.add_option(
        "--batch",
        dest="batch",
        choices=["condor", "slurm", "slurm/psi", "local", "local/futures"],
        default="local",
        help="Run HTCondor with or without Docker image of HiggsDNA's current master branch or run via SLURM. The slurm/psi option is for use on the PSI Tier 3 only. If local, run with futures. Default: futures.",
    )
    parser.add_option(
        "--logs",
        dest="logs",
        default="",
        help="Output path of the Log files of either HTCondor or SLURM.",
    )
    parser.add_option(
        "--eraFlag",
        dest="eraFlag",
        action="store_true",
        default=False,
        help="Returns era flag in the process dictionary to allow distinction.",
    )
    parser.add_option(
        "--root-only",
        dest="root_only",
        action="store_true",
        default=False,
        help="Produces only ROOT output (for example for FinalFits).",
    )
    parser.add_option(
        "--type",
        type="choice",
        dest="type",
        default="",
        choices=["mc", "MC", "Mc", "mC", "data", "Data", "DATA", ""],
        help="Type of dataset (data or mc).",
    )
    parser.add_option(
        "--process-map",
        dest="process_map",
        type="string",
        default=None,
        help="Path to a YAML file defining the process-to-keyword map."
    )
    parser.add_option(
        "--outfiles-map",
        dest="outfiles_map",
        type="string",
        default=None,
        help="Path to YAML/JSON defining ROOT-output filename templates."
    )
    parser.add_option(
        "--time",
        type=str,
        dest="time",
        default=None,
        help="Specifies the maximum duration for the job. Currently only applied when running on SLURM. "
            "Format 'HH:MM:SS'.",
    )
    parser.add_option(
        "--memory",
        type=str,
        dest="memory",
        default=None,
        help="Defines the memory allocation for the job. Only applied on batch systems like SLURM or HTCondor. "
            "Value should include a unit (e.g., '2GB', '4096MB').",
    )
    parser.add_option(
        "--do-b-weight-normalisation",
        default=False,
        action="store_true",
        help="Perform the bweight normalization to make sure the number of event remain the same before and after applying the b tagging weights",
    )
    parser.add_option(
        "--BTagRescaleVariableInfo",
        dest="BTagRescaleVariableInfo",
        default=None,
        help="Provide explicit variable info, e.g. 'JetHT,50,0,1000'.",
    )
    parser.add_option(
        "--use-default-BTagRescaleVariableInfo",
        action="store_true",
        dest="use_default_BTagRescaleVariableInfo",
        default=False,
        help="Rescaling variable info. If passed with no value, defaults to 'n_jets,10,0,10' if you want use other bariable used --BTagRescaleVariableInfo option"
        )
    parser.add_option(
        "--do-theory-weight-normalisation",
        dest="do_theory_weight_normalisation",
        default=False,
        action="store_true",
        help="Perform theory-weight normalization (LHEScale/LHEPdf/AlphaS/PS) to focus on acceptance effects.",
    )
    parser.add_option(
        "--do-lhe-weight-normalisation",
        dest="do_theory_weight_normalisation",
        default=False,
        action="store_true",
        help="DEPRECATED: use --do-theory-weight-normalisation.",
    )
    parser.add_option(
        "--n-workers",
        type=int,
        dest="n_workers",
        default=4,
        help="Number of workers to use to parallelize the merging over datasets. Only used in local processing. ",
    )
    parser.add_option(
        "--n-workers-syst",
        type=int,
        dest="n_workers_syst",
        default=2,
        help="Number of workers to use to parallelize the merging over systematic variations. Only used in local processing. ",
    )
    parser.add_option(
        "--job-flavor",
        type=str,
        dest="job_flavor",
        default=None,
        help="Specifies the job priority or resource class, primarily for HTCondor. "
            "Determines resource allocation and expected queue time. Common flavors include 'espresso', 'microcentury', 'longlunch', etc.",
    )
    parser.add_option(
        "--apptainer-image",
        type=str,
        dest="apptainer_image",
        default="/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/higgsdna:lxplus-el9-latest",
        help="Specifies the Apptainer image used for the HTCondor submission. ",
    )
    parser.add_option(
        "--root-tbasket-length",
        type=int,
        dest="root_tbasket_length",
        default=5000,
        help="Length of the TBaskets while converting the dictionaries carrying the data in the ROOT step.",
    )
    parser.add_option(
        "--custom-accumulator",
        default=False,
        action="store_true",
        dest="custom_accumulator",
        help="If set, the script will process the custom accumulator from the parquet files.",
    )

    parser.add_option(
        "--dataset",
        dest="dataset",
        default=None,
        help="Process only the specified dataset."
    )

    (opt, args) = parser.parse_args()

    opt.output = os.path.realpath(opt.output) if opt.output else ""
    opt.genBinning = os.path.realpath(opt.genBinning) if (opt.genBinning != "") else ""

    # Create output folder if it does not exist
    if opt.output and not os.path.exists(opt.output):
        MKDIRP(opt.output)

    BASEDIR = resources.files("higgs_dna").joinpath("")

    logger_verbosity = "DEBUG" if opt.verbose else "INFO"

    logger = setup_logger(level=logger_verbosity)

    if (opt.batch == "condor") or ("slurm" in opt.batch):
        if not opt.logs:
            logger.error("You did not specify the log directory for the batch submission. Specify one with --logs /path/to/logs.")
            sys.exit()
        else:
            BATCH_PATH = os.path.realpath(os.path.abspath(opt.logs)) # need real absolute path. otherwise problems can arise on lxplus (between afs and eos) for the condor submission

    # load external process_map
    if opt.process_map:
        pm_file = opt.process_map
    else:
        pm_file = os.path.join(BASEDIR, "scripts/postprocessing/config_jsons/process_map.yaml")

    with open(pm_file, "r") as f:
        if pm_file.endswith((".yml", ".yaml")):
            process_map = yaml.safe_load(f)
        else:
            process_map = json.load(f)

    if opt.outfiles_map:
        outfiles_map_file = os.path.realpath(opt.outfiles_map)
    else:
        outfiles_map_file = os.path.join(BASEDIR, "scripts/postprocessing/config_jsons/outfiles.yaml")

    folder_for_dirlist = opt.input
    if opt.batch == "condor":
        if opt.root and not opt.merge:
            folder_for_dirlist = opt.input + "/merged"
        elif opt.folder_structure != "":
            folder_for_dirlist = opt.folder_structure
    else:
        if opt.root and not opt.merge:
            folder_for_dirlist = opt.input + "/merged"
        if opt.folder_structure != "":
            folder_for_dirlist = opt.folder_structure
    
# the key of the var_dict entries is also used as a key for the related root tree branch
# to be consistent with FinalFit naming scheme you shoud use SystNameUp and SystNameDown,
# e.g. "FNUFUp": "FNUF_up", "FNUFDown": "FNUF_down"
    if opt.varDict is None: # If not given in the command line
        logger.info("You did not specify the path to a variation dictionary JSON, so we will only use the nominal input trees.")
        var_dict = {
            "NOMINAL": "nominal",
        }
        # Creating nominal var_dict temporarily in higgs_dna's base folder or the batch log directory
        if ("slurm" in opt.batch) or (opt.batch == "condor"):
            var_dict_loc = os.path.join(BATCH_PATH, "variation.json")
        else:
            var_dict_loc = os.path.join(BASEDIR, "variation.json")
        with open(var_dict_loc, "w") as file:
            file.write(json.dumps(var_dict))
    else:
        var_dict_loc = os.path.realpath(opt.varDict)
        with open(var_dict_loc, "r") as jf:
            var_dict = json.load(jf)

    # input_dir = Path(folder_for_dirlist)

    # # List first-level subdirectories
    # subdirs = [d for d in input_dir.iterdir() if d.is_dir()]

    # # Does this directory itself look like a HiggsDNA dataset?
    # is_single_dataset = any((input_dir / v).is_dir() for v in var_dict.values())

# # Creating dirlist
#     os.system(
#         f"find {folder_for_dirlist} -mindepth 1 -maxdepth 1 -type d | grep -v '^.$' | grep -v .coffea | grep -vE '/[^/]*(merged|root)[^/]*$' |"
#         + "awk -F'/' '{print $NF}' > dirlist.txt"
#         )

    # if is_single_dataset:
    #     logger.info(f"Detected single dataset input: {input_dir.name}")

    #     with open("dirlist.txt", "w") as f:
    #         f.write(input_dir.name + "\n")

    #     # Make the rest of the script treat the parent as the input directory
    #     opt.input = str(input_dir.parent)
    #     folder_for_dirlist = opt.input
    #     input_dir = Path(folder_for_dirlist)



    # else:
    #     os.system(
    #         f"find {folder_for_dirlist} -mindepth 1 -maxdepth 1 -type d "
    #         "| grep -v '^.$' "
    #         "| grep -v .coffea "
    #         "| grep -vE '/[^/]*(merged|root)[^/]*$' "
    #         "| awk -F'/' '{print $NF}' > dirlist.txt"
    #     )

    if opt.dataset is not None:
        logger.info(f"Processing only dataset: {opt.dataset}")

        with open("dirlist.txt", "w") as f:
            f.write(opt.dataset + "\n")

    else:
        # os.system(
        #     f"find {folder_for_dirlist} -mindepth 1 -maxdepth 1 -type d "
        #     "| grep -v '^.$' "
        #     "| grep -v .coffea "
        #     "| grep -vE '/[^/]*(merged|root)[^/]*$' "
        #     "| awk -F'/' '{print $NF}' > dirlist.txt"
        # )

        os.system(
            f"find {folder_for_dirlist} -mindepth 1 -maxdepth 1 -type d "
            "| grep -v '^.$' "
            "| grep -v .coffea "
            "| grep -vE '/[^/]*(merged|root|CATS)[^/]*$' "
            "| awk -F'/' '{print $NF}' > dirlist.txt"
        )
# Here we prepare to split the output into categories, in the dictionary are defined the cuts to be applyed by pyarrow.ParquetDataset
# when reading the data, the variable obviously has to be in the dumped .parquet
# This can be improved by passing the configuration via json loading
    if opt.cats and opt.catDict is not None:
        cat_dict_loc = os.path.realpath(opt.catDict)
        with open(cat_dict_loc, "r") as jf:
            cat_dict = json.load(jf)
    else:
        logger.info("You chose to run without cats or you did not specify the path to a categorisation dictionary JSON, so we will only use one inclusive NOTAG category.")
        cat_dict = {"NOTAG": {"cat_filter": [("pt", ">", -1.0)]}}
        # Creating NOTAG cat_dict temporarily in higgs_dna's base folder or the batch log directory
        if ("slurm" in opt.batch) or (opt.batch == "condor"):
            cat_dict_loc = os.path.join(BATCH_PATH, "category.json")
        else:
            cat_dict_loc = os.path.join(BASEDIR, "category.json")
        with open(cat_dict_loc, "w") as file:
            file.write(json.dumps(cat_dict))

    EXEC_PATH = os.path.realpath(os.getcwd())
    os.chdir(opt.input)
    IN_PATH = os.path.realpath(os.getcwd())
    SCRIPT_DIR = os.path.dirname(
        os.path.abspath(__file__)
    )  # script directory

# I create a dictionary and save it to a temporary json so that this can be shared between the two scripts
# and then gets deleted to not leave trash around. We have to care for the environment :P.
# Not super elegant, open for suggestions
# with open("category.json", "w") as file:
#     file.write(json.dumps(cat_dict))

# Same for variation dictionary, which is shared between merging and ROOTing steps.
# with open("variation.json", "w") as file:
#     file.write(json.dumps(var_dict))

# Using OUT_PATH for the location of the output if different from the input path
    if (not opt.batch == "condor") and (not "slurm" in opt.batch):
        if opt.output == EXEC_PATH:
            OUT_PATH = IN_PATH

            dirlist_path = f"{EXEC_PATH}/dirlist.txt"
        else:
            OUT_PATH = os.path.realpath(opt.output)

            dirlist_path = f"{OUT_PATH}/dirlist.txt"
            os.system(f"mv {EXEC_PATH}/dirlist.txt {OUT_PATH}/dirlist.txt")
    else:
        if (opt.output == "") or (opt.batch == "slurm/psi"):
            OUT_PATH = IN_PATH
            if (opt.batch == "slurm/psi"):
                OUT_PATH = os.path.realpath(opt.output)

            dirlist_path = f"{EXEC_PATH}/dirlist.txt"
        else:
            OUT_PATH = os.path.realpath(opt.output)

            dirlist_path = f"{OUT_PATH}/dirlist.txt"
            os.system(f"mv {EXEC_PATH}/dirlist.txt {OUT_PATH}/dirlist.txt")

    genBinning_str = f"--genBinning {opt.genBinning}" if (opt.genBinning != "")  else ""
    diff_variable_str = f"--diff-variable {opt.diff_variable}" if opt.diff_variable else ""
    tbasket_str = f"--tbasket-length {opt.root_tbasket_length}" if (opt.root_tbasket_length != "") else ""
    outfiles_map_str = f"--outfiles-map {outfiles_map_file}" if (opt.outfiles_map != "")  else ""
    custom_accumulator_str = "--custom-accumulator" if opt.custom_accumulator else ""
    verbose_str = "--verbose" if opt.verbose else ""

# Define string if normalisation to be skipped
    skip_normalisation_str = "--skip-normalisation" if opt.skip_normalisation else ""
    merge_data_str = "--merge-all-data" if opt.merge_data else ""
    do_syst_str = "--do-syst" if opt.syst else ""
    do_b_weight_normalisation_str = "--do-b-weight-normalisation" if opt.do_b_weight_normalisation else ""
    if opt.BTagRescaleVariableInfo is not None:
        do_BTagRescaleVariableInfo_str = f"--BTagRescaleVariableInfo {opt.BTagRescaleVariableInfo}"
    elif opt.use_default_BTagRescaleVariableInfo:
        do_BTagRescaleVariableInfo_str = "--BTagRescaleVariableInfo"
    else:
        do_BTagRescaleVariableInfo_str = ""
    do_theory_weight_normalisation_str = "--do-theory-weight-normalisation" if opt.do_theory_weight_normalisation else ""

# The process var below is the function that will be executed in parallel for each systematic variation. It substitutes the old loop of the systematics to speed up the process.
# Paths now must be ABSOLUTE!! - CD while multi thread is not a good idea!
    def process_var(var, var_dict, IN_PATH, OUT_PATH, SCRIPT_DIR, file, cat_dict, verbose_str, skip_normalisation_str):
        target_dir = f"{OUT_PATH}/merged/{file}/{var_dict[var]}"
        MKDIRP(target_dir)

        command = f"merge_parquet.py --source {IN_PATH}/{file}/{var_dict[var]} --target {target_dir}/ --cats {cat_dict} {verbose_str} {skip_normalisation_str} {genBinning_str} {diff_variable_str} {do_b_weight_normalisation_str} {do_BTagRescaleVariableInfo_str} {do_theory_weight_normalisation_str} --abs {custom_accumulator_str}"
        logger.info(command)

        # Execute the command using subprocess.run
        subprocess.run(command, shell=True, cwd=SCRIPT_DIR, check=True)

# Loop to paralelize the loop over the "files", which are the ttH_125_preEE, etc. datasets
    def process_file(file, IN_PATH, OUT_PATH, SCRIPT_DIR, var_dict, cat_dict, verbose_str, skip_normalisation_str, opt):
        file = file.strip()  # Removes newline characters and leading/trailing whitespace
        if "data" not in file.lower():
            if opt.type and opt.type.lower() == "data":
                logger.warning(f"Type data selected, but {file} is MC. Ignoring it...")
                return
            target_path = f"{OUT_PATH}/merged/{file}"
            if os.path.exists(target_path):
                raise Exception(f"The selected target path: {target_path} already exists")
            MKDIRP(target_path)

            if opt.syst:
                # Systematic variations processing
                with ThreadPoolExecutor(max_workers=opt.n_workers_syst) as executor:
                    futures = [executor.submit(process_var, var, var_dict, IN_PATH, OUT_PATH, SCRIPT_DIR, file, cat_dict_loc, verbose_str, skip_normalisation_str) for var in var_dict]

                for future in futures:
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Error processing variable: {e}")
            else:
                # Single nominal processing for MC
                command = f"merge_parquet.py --source {IN_PATH}/{file}/nominal --target {target_path}/ --cats {cat_dict_loc} {verbose_str} {skip_normalisation_str} {genBinning_str} {diff_variable_str} {do_b_weight_normalisation_str} {do_BTagRescaleVariableInfo_str} {do_theory_weight_normalisation_str} --abs {custom_accumulator_str}"
                subprocess.run(command, shell=True, cwd=SCRIPT_DIR, check=True)
        else:
            if opt.type and opt.type.lower() == "mc":
                logger.warning(f"Type MC selected, but {file} is data. Ignoring it...")
                return
            # Data processing
            data_dir_path = f'{OUT_PATH}/merged/Data_{file.split("_")[-1]}'
            target_file = f'{data_dir_path}/{file}_'
            if len(glob.glob(target_file+'*')) > 0:
                raise Exception(f"The selected target files: {target_file}* already exist")
            if not os.path.exists(data_dir_path):
                MKDIRP(data_dir_path)
            command = f'merge_parquet.py --source {IN_PATH}/{file}/nominal --target {target_file} --cats {cat_dict_loc} {verbose_str} --is-data {genBinning_str} {diff_variable_str} --abs {custom_accumulator_str}'
            subprocess.run(command, shell=True, cwd=SCRIPT_DIR, check=True)

    def root_process_var(cat_dict_loc, var_dict_loc, IN_PATH, OUT_PATH, SCRIPT_DIR, file, verbose_str, skip_normalisation_str):
        file = file.split("\n")[0]
        if opt.merge_data and (opt.type.lower() == "data"):
            source_folder_path = f"{IN_PATH}"
            target_file_path = f"{OUT_PATH}/root/Data/merged.root"
            target_folder_path = f"{OUT_PATH}/root/Data"
            if os.path.exists(target_folder_path):
                raise Exception(
                    f"The selected target path: {target_folder_path} already exists"
                )
            MKDIRP(target_folder_path)
        if (not opt.merge_data) or (opt.type.lower() == "mc"):
            source_folder_path = f"{IN_PATH}/{file}"
            target_file_path = f"{OUT_PATH}/root/{file}/merged.root"
            target_folder_path = f"{OUT_PATH}/root/{file}"
            if os.path.exists(target_folder_path):
                raise Exception(
                    f"The selected target path: {target_folder_path} already exists"
                )

            MKDIRP(target_folder_path)

        command = f"merge_root.py --source {source_folder_path} --target {target_file_path} --cats {cat_dict_loc} --abs {genBinning_str} {diff_variable_str} --vars {var_dict_loc} --type {opt.type} --process {decompose_string(file, process_map)} {verbose_str} {outfiles_map_str} {do_b_weight_normalisation_str} {do_BTagRescaleVariableInfo_str} {do_theory_weight_normalisation_str} {skip_normalisation_str} {merge_data_str} {do_syst_str}"
        logger.info(command)
        
        # Execute the command using subprocess.run
        subprocess.run(command, shell=True, cwd=SCRIPT_DIR, check=True)

    if ("local" in opt.batch):
        if opt.root_only:
            if opt.batch == "local/futures":
                logger.info("Using futures")
                with open(dirlist_path) as fl:
                    files = fl.readlines()
                    if opt.merge_data and (opt.type.lower() == "data"):
                        for j, file in enumerate(files):
                            file = file.split("\n")[0]  # otherwise it contains an end of line and messes up the os.walk() call
                            if j > 0: continue
                            root_process_var(cat_dict_loc, var_dict_loc, IN_PATH, OUT_PATH, SCRIPT_DIR, file, verbose_str, skip_normalisation_str)
                    else:
                        print(files)
                        # No more loop over the files, we will use the ThreadPoolExecutor to parallelize the process!
                        with ThreadPoolExecutor(max_workers=opt.n_workers) as executor:
                            futures = [executor.submit(root_process_var, cat_dict_loc, var_dict_loc, IN_PATH, OUT_PATH, SCRIPT_DIR, file, verbose_str, skip_normalisation_str) for file in files]

                        # Optionally, wait for all futures to complete and check for exceptions
                        for future in futures:
                            try:
                                future.result()
                            except Exception as e:
                                # Log file-level exceptions
                                logger.error(f"Error processing file: {e}")

            else:
                with open(dirlist_path) as fl:
                    files = fl.readlines()
                    for j, file in enumerate(files):
                        file = file.split("\n")[0]
                        if opt.merge_data and (opt.type.lower() == "data") and j > 0: continue
                        root_process_var(cat_dict_loc, var_dict_loc, IN_PATH, OUT_PATH, SCRIPT_DIR, file, verbose_str, skip_normalisation_str)
        if opt.merge:
            with open(dirlist_path) as fl:
                files = fl.readlines()

                # No more loop over the files, we will use the ThreadPoolExecutor to parallelize the process!
                with ThreadPoolExecutor(max_workers=opt.n_workers) as executor:
                    futures = [executor.submit(process_file, file, IN_PATH, OUT_PATH, SCRIPT_DIR, var_dict, cat_dict, verbose_str, skip_normalisation_str, opt) for file in files]

                # Optionally, wait for all futures to complete and check for exceptions
                for future in futures:
                    try:
                        future.result()
                    except Exception as e:
                        # Log file-level exceptions
                        logger.error(f"Error processing file: {e}")


                # at this point Data will be split in eras if any Data dataset is present, here we merge them again in one allData file to rule them all
                # we also skip this step if there is no Data
                for file in files:
                    file = file.split("\n")[0]  # otherwise it contains an end of line and messes up the os.walk() call
                    if "data" in file.lower() or "DoubleEG" in file:
                        if opt.type and opt.type.lower() == "mc":
                            logger.warning(f"Type MC selected, but {file} is data. Ignoring it...")
                            continue
                        dirpath, dirnames, filenames = next(os.walk(f'{OUT_PATH}/merged/Data_{file.split("_")[-1]}'))
                        if len(filenames) > 0:
                            command = f'merge_parquet.py --source {OUT_PATH}/merged/Data_{file.split("_")[-1]} --target {OUT_PATH}/merged/Data_{file.split("_")[-1]}/allData_ --cats {cat_dict_loc} --is-data --merge-all-data {genBinning_str} {diff_variable_str} {verbose_str} --abs {custom_accumulator_str}'
                            subprocess.run(command, shell=True, cwd=SCRIPT_DIR, check=True)
                            break
                        else:
                            logger.info(f'No merged parquet found for {file} in the directory: {OUT_PATH}/merged/Data_{file.split("_")[-1]}')

        if opt.root:
            logger.info("Starting root step")
            if opt.syst:
                logger.info("you've selected the run with systematics")
                args = "--do-syst"
            else:
                logger.info("you've selected the run without systematics")
                args = ""

            if opt.merge:
                IN_PATH = OUT_PATH
            # Note, in my version of HiggsDNA I run the analysis splitting data per Era in different datasets
            # the treatment of data here is tested just with that structure
            with open(dirlist_path) as fl:
                files = fl.readlines()
                for file in files:
                    file = file.split("\n")[0]
                    if "data" not in file.lower() and (not "unknown" in decompose_string(file, process_map, era_flag=opt.eraFlag)):
                        if opt.type and opt.type.lower() == "data":
                            logger.warning(f"Type data selected, but {file} is MC. Ignoring it...")
                            continue
                        if os.path.exists(f"{OUT_PATH}/root/{file}"):
                            raise Exception(
                                f"The selected target path: {OUT_PATH}/root/{file} already exists"
                            )
                        print(file)
                        if os.listdir(f"{IN_PATH}/merged/{file}/"):
                            logger.info(f"Found merged files {IN_PATH}/merged/{file}/")
                        else:
                            raise Exception(f"Merged parquet not found at {IN_PATH}/merged/")
                        MKDIRP(f"{OUT_PATH}/root/{file}")
                        os.chdir(SCRIPT_DIR)
                        os.system(
                            f"convert_parquet_to_root.py {IN_PATH}/merged/{file}/merged.parquet {OUT_PATH}/root/{file}/merged.root mc --process {decompose_string(file, process_map)} {args} --cats {cat_dict_loc} --vars {var_dict_loc} {verbose_str} {genBinning_str} {tbasket_str} {outfiles_map_str} --abs"
                        )
                    elif "data" in file.lower():
                        if opt.type and opt.type.lower() == "mc":
                            logger.warning(f"Type MC selected, but {file} is data. Ignoring it...")
                            continue
                        if os.listdir(f'{IN_PATH}/merged/Data_{file.split("_")[-1]}/'):
                            logger.info(
                                f'Found merged data files in: {IN_PATH}/merged/Data_{file.split("_")[-1]}/'
                            )
                        else:
                            raise Exception(
                                f'Merged parquet not found at: {IN_PATH}/merged/Data_{file.split("_")[-1]}/'
                            )

                        if os.path.exists(
                            f'{OUT_PATH}/root/Data/allData_{file.split("_")[-1]}.root'
                        ):
                            logger.info(
                                f'Data already converted: {OUT_PATH}/root/Data/allData_{file.split("_")[-1]}.root'
                            )
                            continue
                        elif not os.path.exists(f"{OUT_PATH}/root/Data/"):
                            MKDIRP(f"{OUT_PATH}/root/Data")
                            os.chdir(SCRIPT_DIR)
                            os.system(
                                f'convert_parquet_to_root.py {IN_PATH}/merged/Data_{file.split("_")[-1]}/allData_merged.parquet {OUT_PATH}/root/Data/allData_{file.split("_")[-1]}.root data --cats {cat_dict_loc} --vars {var_dict_loc} {verbose_str} {genBinning_str} {tbasket_str} {outfiles_map_str} --abs'
                            )
                        else:
                            os.chdir(SCRIPT_DIR)
                            os.system(
                                f'convert_parquet_to_root.py {IN_PATH}/merged/Data_{file.split("_")[-1]}/allData_merged.parquet {OUT_PATH}/root/Data/allData_{file.split("_")[-1]}.root data --cats {cat_dict_loc} --vars {var_dict_loc} {verbose_str} {genBinning_str} {tbasket_str} {outfiles_map_str} --abs'
                            )

        if opt.ws:
            if not os.listdir(opt.final_fit):
                raise Exception(
                    f"The selected FlashggFinalFit path: {opt.final_fit} is invalid"
                )

            if os.path.exists(f"{IN_PATH}/root/Data"):
                os.system(f"echo Data >> {dirlist_path}")

            data_done = False

            with open(dirlist_path) as fl:
                files = fl.readlines()
                if opt.syst:
                    doSystematics = "--doSystematics"
                else:
                    doSystematics = ""
                for dir in files:
                    dir = dir.split("\n")[0]
                    # if MC
                    if "data" not in dir.lower() and (not "unknown" in decompose_string(dir, process_map, era_flag=opt.eraFlag)):
                        if os.listdir(f"{IN_PATH}/root/{dir}/"):
                            filename = subprocess.check_output(
                                f"find {IN_PATH}/root/{dir} -name *.root -type f",
                                shell=True,
                                universal_newlines=True,
                            )
                        else:
                            raise Exception(
                                f"The selected target path: {IN_PATH}/root/{dir} it's empty"
                            )
                        doNOTAG = ""
                        if ("NOTAG" in cat_dict.keys()):
                            doNOTAG = "--doNOTAG"
                        command = f"python trees2ws.py {doNOTAG} --inputConfig {opt.config} --productionMode {decompose_string(dir, process_map)} --year 2017 {doSystematics} --inputTreeFile {filename}"
                        activate_final_fit(opt.final_fit, command)
                    elif "data" in dir.lower() and not data_done:
                        if os.listdir(f"{IN_PATH}/root/Data/"):
                            filename = subprocess.check_output(
                                f"find {IN_PATH}/root/Data -name *.root -type f",
                                shell=True,
                                universal_newlines=True,
                            )
                        else:
                            raise Exception(
                                f"The selected target path: {IN_PATH}/root/{dir} it's empty"
                            )
                        doNOTAG = ""
                        if ("NOTAG" in cat_dict.keys()):
                            doNOTAG = "--doNOTAG"
                        command = f"python trees2ws_data.py {doNOTAG} --inputConfig {opt.config} --inputTreeFile {filename}"
                        activate_final_fit(opt.final_fit, command)
                        data_done = True
            os.chdir(EXEC_PATH)

    elif ("slurm" in opt.batch):
        slurm_postprocessing(
            _opt=opt, OUT_PATH=OUT_PATH, IN_PATH=IN_PATH, SLURM_PATH=BATCH_PATH, dirlist_path=dirlist_path, var_dict=var_dict,
            cat_dict_loc=cat_dict_loc, var_dict_loc=var_dict_loc, genBinning_str=genBinning_str,
            diff_variable_str=diff_variable_str,
            skip_normalisation_str=skip_normalisation_str, merge_data_str=merge_data_str, do_syst_str=do_syst_str, tbasket_str=tbasket_str, time=opt.time, partition=opt.job_flavor,memory=opt.memory, decompose_string=decompose_string, logger=logger,
            process_map=process_map, outfiles_map_str=outfiles_map_str, verbose_str=verbose_str, custom_accumulator_str=custom_accumulator_str, do_b_weight_normalisation_str=do_b_weight_normalisation_str, do_BTagRescaleVariableInfo_str=do_BTagRescaleVariableInfo_str, do_theory_weight_normalisation_str=do_theory_weight_normalisation_str
            )

    elif opt.batch == "condor":
        htcondor_postprocessing(
            _opt=opt, OUT_PATH=OUT_PATH, IN_PATH=IN_PATH, CONDOR_PATH=BATCH_PATH, SCRIPT_DIR=SCRIPT_DIR, dirlist_path=dirlist_path,
            var_dict=var_dict, cat_dict_loc=cat_dict_loc, var_dict_loc=var_dict_loc, genBinning_str=genBinning_str,
            diff_variable_str=diff_variable_str,
            skip_normalisation_str=skip_normalisation_str, merge_data_str=merge_data_str, do_syst_str=do_syst_str, tbasket_str=tbasket_str, job_flavor=opt.job_flavor, memory=opt.memory,decompose_string=decompose_string, logger=logger,
            process_map=process_map, outfiles_map_str=outfiles_map_str, verbose_str=verbose_str, custom_accumulator_str=custom_accumulator_str, do_b_weight_normalisation_str=do_b_weight_normalisation_str,do_BTagRescaleVariableInfo_str=do_BTagRescaleVariableInfo_str, do_theory_weight_normalisation_str=do_theory_weight_normalisation_str
        )

    # We don't want to leave trash around
    if 'local' in opt.batch:
        if os.path.exists(dirlist_path):
            os.system(f"rm {dirlist_path}")
        if ((opt.catDict is None) and (opt.varDict is None)) and (opt.output == ""):
            if os.path.exists(cat_dict_loc):
                os.system(f"rm {cat_dict_loc}")
            if os.path.exists(var_dict_loc):
                os.system(f"rm {var_dict_loc}")

if __name__ == "__main__":
    main()
