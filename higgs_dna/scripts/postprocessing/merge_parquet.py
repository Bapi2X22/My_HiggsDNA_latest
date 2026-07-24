#!/usr/bin/env python
import argparse
import json
import ast
import os
import glob
import awkward as ak
from higgs_dna.utils.logger_utils import setup_logger
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
from pathlib import Path
from importlib import resources
from higgs_dna.scripts.postprocessing.tools.Btag_WeightSum_Calculation import Get_WeightSum_Btag, Renormalize_BTag_Weights, Get_bin_edges_and_ration, apply_rescaling, Get_ratio_with_bWeight
from higgs_dna.scripts.postprocessing.tools.LHE_WeightSum_Calculation import Get_WeightSum_LHE, Renormalize_LHE_Weights
from higgs_dna.scripts.postprocessing.tools.postprocessing_tools import filter_and_set_diff_variable
from coffea.processor.accumulator import iadd

def get_active_gen_binning(gen_binning, diff_variable, logger):
    if gen_binning is None:
        return None

    if not diff_variable:
        return gen_binning

    if diff_variable in gen_binning:
        return {diff_variable: gen_binning[diff_variable]}

    logger.warning(
        f"Differential variable '{diff_variable}' was requested but not found in genBinning keys: {list(gen_binning.keys())}. "
        "Falling back to all genBinning entries."
    )
    return gen_binning

def process_custom_accumulator(source_path, logger):
    # Get the custom accumulator from all files in the source path
    accumulator = None
    source_files = glob.glob("%s/*.parquet" % source_path)
    for f in source_files:
        try:
            file_accumulator = pq.read_schema(f).metadata[b'custom_accumulator']
        except KeyError:
            logger.warning(f"Custom accumulator requested but not found in file {f}")
            file_accumulator = b"{}"
        file_accumulator = json.loads(file_accumulator)
        if accumulator is None:
            accumulator = file_accumulator
        else:
            accumulator = iadd(accumulator, file_accumulator)
    return accumulator


def main():
    parser = argparse.ArgumentParser(
        description="Simple utility script to merge all parquet files in one folder."
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Debugging verbosity for logger.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="",
        help="Comma separated paths (with trailing slash) to folder where multiple parquet files are located. Careful: Folder should ONLY contain parquet files!",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="",
        help="Comma separated paths (with trailing slash) to desired folder. Resulting merged file is placed there.",
    )
    parser.add_argument(
        "--cats",
        type=str,
        dest="cats_dict",
        default="",
        help="Dictionary containing category selections.",
    )
    parser.add_argument(
        "--is-data",
        default=False,
        action="store_true",
        help="Files to be merged are data and therefore do not require normalisation.",
    )
    parser.add_argument(
        "--merge-all-data",
        default=False,
        action="store_true",
        help="To be used for merging 'Data*' parquets to a single 'allData' parquet",
    )
    parser.add_argument(
        "--skip-normalisation",
        default=False,
        action="store_true",
        help="Independent of file type, skip normalisation step",
    )
    parser.add_argument(
        "--abs",
        dest="abs",
        action="store_true",
        default=False,
        help="Uses absolute path for the dictionary files.",
    )
    parser.add_argument(
        "--genBinning",
        type=str,
        dest="genBinning",
        default="",
        help="Optional: Path to the JSON containing the binning at gen-level.",
    )
    parser.add_argument(
        "--diff-variable",
        type=str,
        dest="diff_variable",
        default="",
        help="Optional: Differential variable key to select from genBinning (e.g. 'PTH').",
    )
    parser.add_argument(
        "--do-b-weight-normalisation",
        default=False,
        action="store_true",
        help="Perform the bweight normalization to make sure the number of event remain the same before and after applying the b tagging weights",
     )
    parser.add_argument(
        "--BTagRescaleVariableInfo",
        nargs="?",
        const="n_jets,10,0,10",
        default=None,
        type=str,
        help="Rescaling variable info. If passed with no value, defaults to 'n_jets,10,0,10', other variable and bin info can be provided 'JetHT,50,0,1000' ",
    )
    parser.add_argument(
        "--do-theory-weight-normalisation",
        dest="do_theory_weight_normalisation",
        default=False,
        action="store_true",
        help="Perform theory-weight normalization (LHEScale/LHEPdf/AlphaS/PS) to focus on acceptance effects.",
    )
    parser.add_argument(
        "--do-lhe-weight-normalisation",
        dest="do_theory_weight_normalisation",
        default=False,
        action="store_true",
        help="DEPRECATED: use --do-theory-weight-normalisation.",
    )
    parser.add_argument(
        "--custom-accumulator",
        default=False,
        action="store_true",
        dest="custom_accumulator",
        help="If set, the script will process the custom accumulator from the parquet files.",
    )
    parser.add_argument(
        "--events-per-row-group",
        default=131_072,  # 128k events per row group as default (can be adjusted based on memory constraints)
        type=int,
        help="Number of events per row group in the output parquet file. Default is 131_072.",
    )

    args = parser.parse_args()
    source_paths = args.source.split(",")
    target_paths = args.target.split(",")

    BASEDIR = resources.files("higgs_dna").joinpath("")

    logger_verbosity = "DEBUG" if args.verbose else "INFO"

    logger = setup_logger(level=logger_verbosity)

    if args.genBinning != "":
        if args.abs:
            genBinning_path = os.path.realpath(args.genBinning)
        else:
            genBinning_path = os.path.join(BASEDIR, "scripts/postprocessing/sample_gen_binning.json")
        with open(genBinning_path, 'r') as json_file:
            gen_binning = json.load(json_file)
        gen_binning = get_active_gen_binning(gen_binning, args.diff_variable, logger)
    else:
        gen_binning = None

    if(args.BTagRescaleVariableInfo):
        BTagRescaleVariable_Info = args.BTagRescaleVariableInfo.split(',')
        BTagRescaleVariable_Info[1:] = [int(i) for i in BTagRescaleVariable_Info[1:]]
        if len(BTagRescaleVariable_Info) !=4:
            raise Exception("Wrong format for BTagRescaleVariableInfo, please provide info in the following format: 'VariableName,nbins,min,max'")

    if (
        (len(source_paths) != len(target_paths))
        or (args.source == "")
        or (args.target == "")
    ):
        logger.info("You gave a different number of sources and targets")
        exit


    if args.cats_dict != "":
        if args.abs:
            cats_path = os.path.realpath(args.cats_dict)
        else:
            cats_path = os.path.join(BASEDIR, "category.json")
        with open(cats_path) as pf:
            cat_dict = json.load(pf)
        for cat in cat_dict:
            logger.info(f"Found category: {cat}")
    else:
        logger.info(
            "You provided an invalid dictionary containing categories information, have a look at your version of prepare_output_file.py"
        )
        logger.info(
            "An inclusive NOTAG category is used as default"
        )
        cat_dict = {"NOTAG": {"cat_filter": [("pt", ">", -1.0)]}}

    if (not args.is_data) & (not args.skip_normalisation):
        logger.info(
            "Extracting sum of gen weights (before selection) from metadata of files to be merged."
        )
        if args.do_b_weight_normalisation:
            IsBtagNorm_sys_arr, WeightSum_preBTag_arr, WeightSum_postBTag_arr, WeightSum_postBTag_sys_arr = Get_WeightSum_Btag(source_paths, logger)
        if args.do_theory_weight_normalisation:
            sum_LHEPdf_beforesel_arr, sum_LHEScale_beforesel_arr, do_lhe_norm = Get_WeightSum_LHE(source_paths, logger)

        sum_genw_beforesel_arr = []
        for i, source_path in enumerate(source_paths):
            source_files = glob.glob("%s/*.parquet" % source_path)
            sum_genw_beforesel = 0
            for f in source_files:
                sum_genw_beforesel += float(pq.read_schema(f).metadata[b'sum_genw_presel'])
            sum_genw_beforesel_arr.append(sum_genw_beforesel)
        logger.info(
            "Successfully extracted sum of gen weights (before selection)"
        )

    for i, source_path in enumerate(source_paths):
        # Process custom accumulator
        if args.custom_accumulator:
            logger.info(f"Processing custom accumulator for {source_path}")
            custom_accumulator = process_custom_accumulator(source_path, logger)

        for cat in cat_dict:
            logger.info("-" * 125)
            logger.info(
                f"INFO: Starting parquet file merging. Attempting to read parquet dataset from {source_path}, for category: {cat}"
            )
            # dataset = ds.dataset(glob.glob(source_path+"/Data*.parquet") if args.merge_all_data else source_path)
            dataset = ds.dataset(glob.glob(source_path+"/*.parquet") if args.merge_all_data else source_path)
            logger.info("Parquet dataset read successfully.")
            logger.info(
                f"Attempting to merge parquet dataset and save to {target_paths[i]}."
            )
            if "Data" in target_paths[i]:
                os.makedirs("/".join(target_paths[i].split("/")[:-1]), exist_ok=True)
            else:
                os.makedirs(target_paths[i], exist_ok=True) # Create target directory if it does not exist

            # Process in batches
            output_file = target_paths[i] + cat + "_merged.parquet"

            # Process in batches
            writer = None

            tables_buffer = []
            current_buffered_rows = 0

            for batch in dataset.to_batches(filter=pq.filters_to_expression(cat_dict[cat]["cat_filter"]), batch_size=args.events_per_row_group, batch_readahead=2, fragment_readahead=2):
                batch_arr = ak.from_arrow(batch)

                if (not args.is_data) & (not args.skip_normalisation):
                    if gen_binning != None:
                        for keys in gen_binning:
                            var_dict = {ast.literal_eval(key): value for key, value in gen_binning[keys].items()}
                            if len(list(var_dict.keys())[0]) == 5:
                                selectionVariableName = list(var_dict.keys())[0][0]
                                var_dict = {k[1:]: v for k, v in var_dict.items()}
                            else:
                                selectionVariableName = keys
                            if selectionVariableName not in batch_arr.fields:
                                logger.warning(
                                    f"Skipping gen-binning variable '{keys}' because selection field '{selectionVariableName}' is not present in input parquet fields."
                                )
                                continue
                            batch_arr = filter_and_set_diff_variable(batch_arr, var_dict, selectionVariableName, "diffVariable_" + keys)

                    batch_arr['weight_nominal'] = batch_arr['weight']
                    syst_weight_fields = [field for field in batch_arr.fields if (("weight_" in field) and ("Up" in field or "Down" in field))]
                    for weight_field in ["weight"] + syst_weight_fields:
                        batch_arr[weight_field] = batch_arr[weight_field] / sum_genw_beforesel_arr[i]
                    logger.info("Successfully added normalised weight column")

                    if args.do_b_weight_normalisation:
                        if (WeightSum_preBTag_arr[i] / WeightSum_postBTag_arr[i]) != 1:
                            batch_arr = Renormalize_BTag_Weights(batch_arr, syst_weight_fields, target_paths[i], cat, WeightSum_preBTag_arr[i], WeightSum_postBTag_arr[i], WeightSum_postBTag_sys_arr[i], IsBtagNorm_sys_arr[i], logger)
                    if args.do_theory_weight_normalisation:
                        if do_lhe_norm[i]:
                            batch_arr = Renormalize_LHE_Weights(batch_arr, target_paths[i], cat, sum_LHEPdf_beforesel_arr[i], sum_LHEScale_beforesel_arr[i], logger)

                table = ak.to_arrow_table(batch_arr, extensionarray=False)
                if args.custom_accumulator:
                    logger.info("Adding custom accumulator")
                    table = table.replace_schema_metadata({b'custom_accumulator': json.dumps(custom_accumulator).encode("utf-8")})
                    logger.info("Custom accumulator added successfully")
                else:
                    table = table.replace_schema_metadata()

                tables_buffer.append(table)
                current_buffered_rows += table.num_rows

                if current_buffered_rows >= args.events_per_row_group:
                    merged_table = pa.concat_tables(tables_buffer).combine_chunks()
                    if writer is None:
                        schema = merged_table.schema
                        writer = pq.ParquetWriter(output_file, schema)
                        logger.info(f"saving output file {output_file = }")

                    writer.write_table(merged_table)

                    tables_buffer = []
                    current_buffered_rows = 0

            if len(tables_buffer) > 0:
                merged_table = pa.concat_tables(tables_buffer).combine_chunks()
                if writer is None:
                    schema = merged_table.schema
                    writer = pq.ParquetWriter(output_file, schema)
                    logger.info(f"saving output file {output_file = }")

                writer.write_table(merged_table)

            if writer:
                writer.close()

            logger.info(f"Success! Merged parquet file is located in {output_file}")

            if(args.BTagRescaleVariableInfo):
                    if not args.do_b_weight_normalisation:
                        logger.warning("B-Tag weight rescaling requested but B-Tag weight normalisation not performed. Skipping B-Tag weight rescaling. Please enable --do-b-weight-normalisation to perform B-Tag weight reNormalization first.")
                        exit(0)
                    if(cat == "NOTAG"):
                        logger.info("Starting B-Tag weight rescaling process")
                        temp_file   = Path(target_paths[i] + cat + "_merged_rescaled.parquet")

                        dataset = ak.from_parquet(output_file, columns=[ BTagRescaleVariable_Info[0], "bTagWeight"])
                        xaxis_edges,ratio_val = Get_bin_edges_and_ration(dataset,target_paths[i],logger,Variable_info=BTagRescaleVariable_Info,plot_name="bTagWeight")

                        dataset = ds.dataset(output_file)
                        logger.info("Successfully read the merged parquet file for B-Tag weight rescaling")
                        orig_metadata = dataset.schema.metadata
                        # Process in batches
                        writer = None
                        for batch in dataset.to_batches(batch_size=args.events_per_row_group, batch_readahead=2, fragment_readahead=2):
                            batch_arr = ak.from_arrow(batch)
                            rescaled_weights_dict = apply_rescaling(batch_arr,['weight',"bTagWeight"]+syst_weight_fields,xaxis_edges,ratio_val,logger,Variable_info=BTagRescaleVariable_Info)
                            for weight_field in ["weight","bTagWeight"]+syst_weight_fields:
                                #With rescaled weight we have to make sure the total sum of the weight conserved.
                                weight_factor = ak.sum(batch_arr[weight_field])/ak.sum(rescaled_weights_dict[weight_field])
                                batch_arr[weight_field] = rescaled_weights_dict[weight_field]*weight_factor

                            table = ak.to_arrow_table(batch_arr, extensionarray=False)
                            table = table.replace_schema_metadata(orig_metadata)

                            if writer is None:
                                schema = table.schema
                                writer = pq.ParquetWriter(temp_file, schema)
                            writer.write_table(table)

                        if writer:
                            writer.close()
                        output_file = Path(output_file)
                        output_file.unlink()
                        temp_file.replace(output_file)
                        logger.info(f"Successfully applied B-Tag weight rescaling in file: {output_file}")

                        dataset = ak.from_parquet(output_file, columns=[BTagRescaleVariable_Info[0], "bTagWeight"])
                        Get_ratio_with_bWeight(dataset,BTagRescaleVariable_Info,bweight_name="bTagWeight",plot_name=target_paths[i]+f"/{BTagRescaleVariable_Info[0]}_bTagWeight_rescaled",plot_ratio_min=0.9,plot_ratio_max=1.2)
                    else:
                        logger.warning(f"skiping the B-Weight rescaling. The scale can be derived in bins of {BTagRescaleVariable_Info[0]} only for NOTAG category sinace we need to derive the scale before we apply any cut")
            logger.info("-" * 125)

if __name__ == "__main__":
    main()
