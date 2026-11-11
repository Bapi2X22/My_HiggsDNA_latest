from typing import List, Optional

import awkward as ak
import pandas
import copy
import os
import pathlib
import pyarrow.parquet as pq
import pyarrow as pa
import uproot


def apply_naming_convention(self, events: ak.Array) -> str:
    """
    Apply the correct naming convention.
    Select which uuid (DAS or uproot (Legacy)) should be included in the parquet name.
    """
    DAS_name = events.metadata["filename"]
    DAS_uuid = DAS_name.split("/")[-1].replace(".root", "")
    name = events.attrs["@events_factory"]._partition_key.split("/")

    try:
        convention = self.name_convention
    except AttributeError as err:
        raise AttributeError("Naming convention was not specified.") from err

    # Change the parquet name UUID with DAS UUID
    if convention == "DAS":
        name[0] = DAS_uuid
    # Keep the name unchanged for Legacy convention
    elif convention == "Legacy":
        pass
    else:
        raise ValueError("Invalid naming convention specified.")

    fname = '_'.join(name) + f".{self.output_format}"
    fname = (fname.replace("%2F","")).replace("%3B1","")
    return fname


def diphoton_list_to_pandas(self, diphotons: ak.Array) -> pandas.DataFrame:
    """
    Convert diphoton array to pandas dataframe.
    By default the observables related to each item of the diphoton pair are
    stored preceded by its prefix (e.g. 'lead', 'sublead').
    The observables related to the diphoton pair are stored with no prefix.
    To change the behavior, you can redefine the `diphoton_list_to_pandas` method in the
    derived class.
    """
    output = {}
    for field in ak.fields(diphotons):
        prefix = self.prefixes.get(field, "")
        if len(prefix) > 0:
            for subfield in ak.fields(diphotons[field]):
                if subfield != "__systematics__":
                    thearray = ak.to_numpy(diphotons[field][subfield])
                    output[f"{prefix}_{subfield}"] = thearray
        else:
            thearray = ak.to_numpy(diphotons[field])
            if thearray.ndim == 2:
                for i in range(thearray.shape[1]):
                    output[f"{field}_{i}"] = thearray[:, i]
            else:
                output[field] = thearray
    return pandas.DataFrame(output)


def dump_pandas(
    self,
    pddf: pandas.DataFrame,
    fname: str,
    location: str,
    subdirs: Optional[List[str]] = None,
) -> None:
    """
    Dump a pandas dataframe to disk at location/'/'.join(subdirs)/fname.
    """
    subdirs = subdirs or []

    merged_subdirs = "/".join(subdirs)
    if merged_subdirs:
        destination = f"{location.rstrip('/')}/{merged_subdirs}/{fname}"
    else:
        destination = f"{location.rstrip('/')}/{fname}"

    if not destination.startswith("root://"):
        dirname = os.path.dirname(destination)
        if not os.path.exists(dirname):
            pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
    if self.output_format == "parquet":
        pddf.to_parquet(destination)
    else:
        uproot_file = uproot.recreate(destination)
        uproot_file["Event"] = pddf
        uproot_file.close()


def diphoton_ak_array_fields(
    self, diphotons: ak.Array, fields, logger
) -> ak.Array:
    """
    This function allows you to add the list of variables to be dumped
    Adjust the prefix.
    By default the observables related to each item of the diphoton pair are
    stored preceded by its prefix (e.g. 'lead', 'sublead').
    The observables related to the diphoton pair are stored with no prefix.
    """
    output = {}
    for field in fields:
        if (
            field != "photons"
        ):  # not needed in the output, the information is already stored in pho_lead and pho_sublead
            prefix = self.prefixes.get(field, "")
            if len(prefix) > 0:
                for subfield in ak.fields(diphotons[field]):
                    if subfield != "__systematics__":
                        output[f"{prefix}_{subfield}"] = diphotons[field][subfield]
            else:
                output[field] = diphotons[field]
    return ak.Array(output)


def diphoton_ak_array(self, diphotons: ak.Array) -> ak.Array:
    """
    Adjust the prefix.
    By default the observables related to each item of the diphoton pair are
    stored preceded by its prefix (e.g. 'lead', 'sublead').
    The observables related to the diphoton pair are stored with no prefix.
    """
    output = {}
    for field in ak.fields(diphotons):
        prefix = self.prefixes.get(field, "")
        if len(prefix) > 0:
            for subfield in ak.fields(diphotons[field]):
                if subfield != "__systematics__":
                    output[f"{prefix}_{subfield}"] = diphotons[field][subfield]
        else:
            output[field] = diphotons[field]
    return ak.Array(output)

def diphoton_ak_array_new(self,
    diphotons: ak.Array,
    extra_fields: dict | None = None,
) -> ak.Array:
    """
    Convert diphoton record to a flat awkward array.
    extra_fields contains event-level branches that should be written
    independently of whether a diphoton exists.
    """

    output = {}

    # ---------------------------------------------------------
    # Diphoton branches
    # ---------------------------------------------------------
    for field in ak.fields(diphotons):

        prefix = self.prefixes.get(field, "")

        if prefix:
            for subfield in ak.fields(diphotons[field]):
                if subfield == "__systematics__":
                    continue

                arr = diphotons[field][subfield]

                if "float" in str(arr.type):
                    arr = ak.fill_none(arr, -999.)
                elif "int" in str(arr.type):
                    arr = ak.fill_none(arr, -999)

                output[f"{prefix}_{subfield}"] = arr

        else:

            arr = diphotons[field]

            if "float" in str(arr.type):
                arr = ak.fill_none(arr, -999.)
            elif "int" in str(arr.type):
                arr = ak.fill_none(arr, -999)

            output[field] = arr

    # ---------------------------------------------------------
    # Event-level branches
    # ---------------------------------------------------------
    if extra_fields is not None:
        output.update(extra_fields)

    return ak.Array(output)


def dump_ak_array(
    self,
    akarr: ak.Array,
    fname: str,
    location: str,
    metadata: None,
    subdirs: Optional[List[str]] = None,
) -> None:
    """
    Dump an awkward array to disk at location/'/'.join(subdirs)/fname.
    """
    subdirs = subdirs or []

    merged_subdirs = "/".join(subdirs)
    if merged_subdirs:
        destination = f"{location.rstrip('/')}/{merged_subdirs}/{fname}"
    else:
        destination = f"{location.rstrip('/')}/{fname}"

    pa_table = ak.to_arrow_table(akarr, extensionarray=False)
    # Ensure deterministic column ordering in output parquet files to fix reading issue
    col_names = sorted(pa_table.schema.names)
    pa_table = pa.table([pa_table.column(n) for n in col_names], names=col_names)
    if metadata:
        merged_metadata = {**metadata, **(pa_table.schema.metadata or {})}
        pa_table = pa_table.replace_schema_metadata(merged_metadata)

    if destination.startswith("root://"):
        import fsspec
        with fsspec.open(destination, 'wb') as f:
            pq.write_table(pa_table, f)
    else:
        dirname = os.path.dirname(destination)
        if not os.path.exists(dirname):
            pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
        pq.write_table(pa_table, destination)


def dress_branches(
    main_arr: ak.Array, additional_arr: ak.Array, prefix: str
) -> ak.Array:
    """_summary_

    Args:
        main_arr (awkward.Array): main array
        additional_arr (awkward.Array): add additional branches to the main array
        prefix (str): prefix of the new branches

    Returns:
        awkward.Array: return new array
    """
    import numpy as np

    for field in ak.fields(additional_arr):
        if not field == "__systematics__":
            if "bool" in str(additional_arr[field].type):
                # * change `bool` to `int8` avoid error when using coffea to read the parquet
                main_arr[f"{prefix}_{field}"] = ak.values_astype(
                    additional_arr[field], np.int8
                )
            else:
                main_arr[f"{prefix}_{field}"] = additional_arr[field]
    return main_arr


def get_obj_syst_dict(obj_ak: ak.Array, var_new: Optional[List[str]] = ["pt"]):
    """_summary_

    Args:
        obj_ak (awkward.Array): objects includes the systematics, e.g., Jet collection with jerc up/down branches
        var_new (Optional[List[str]], optional): changed variable(s) due to the systematics. Allow multiple variables, e.g., jerc systematics change both pt and mass, please mention all the changed variables here. Defaults to ["pt"].

    Returns:
        [list, dict]: list of systematics; dictionary of the nominal and variations
    """

    # NOTE: this function only works if the variations are in such a format:
    # variable_systematic_up/down

    # find the systematics
    var_all = obj_ak.fields
    var_syst = [i for i in var_all if i.endswith("_up") or i.endswith("_down")]
    # nominal
    obj_nom = obj_ak[list(set(var_all) - set(var_syst))]
    # extract systematics
    syst_list = []
    for i in var_syst:
        for j in var_new:
            if j in i:
                tmp_syst_name = i.replace(f"{j}_", "")
                if "_up" in tmp_syst_name:
                    tmp_syst_name = tmp_syst_name[:-3]
                else:
                    tmp_syst_name = tmp_syst_name[:-5]
                syst_list.append(tmp_syst_name)
    # remove duplication
    syst_list = list(set(syst_list))
    replace_dict = {}
    for i in syst_list:
        replace_dict[i] = {
            "up": {j: f"{j}_{i}_up" for j in var_new},
            "down": {j: f"{j}_{i}_down" for j in var_new},
        }
    obj_syst_dict = {"nominal": obj_nom}
    for isyst in syst_list:
        for ivariation in replace_dict[isyst]:
            obj_tmp = copy.copy(obj_nom)
            for ivariable in replace_dict[isyst][ivariation]:
                obj_tmp[ivariable] = obj_ak[replace_dict[isyst][ivariation][ivariable]]
            obj_syst_dict.update({f"{isyst}_{ivariation}": obj_tmp})
    return syst_list, obj_syst_dict
