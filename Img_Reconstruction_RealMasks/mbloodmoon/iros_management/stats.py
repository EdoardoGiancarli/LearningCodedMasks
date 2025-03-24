"""
IROS sources reconstruction analysis for fluence and positions.
"""

from copy import deepcopy
import numpy as np
from pandas import DataFrame

from .analyze import dict2df
from mbloodmoon.io import SimulationDataLoader


def iros_radec_accuracy(
    data: dict,
    catalogs: tuple[np.recarray],
    sigma: int | float,
) -> DataFrame: #tuple[list[tuple[str, bool]], tuple[float]]:
    """
    Analysis of the IROS reconstruction RA/DEC estimates and
    comparison with catalogs coordinates.
    """
    errboxes = ([], [])

    for cat, cameraID, errbox in zip(
        catalogs,
        data.keys(),
        errboxes,
    ):
        print(f"# Checking {cameraID.upper()}...")
        for name, ra, dec, dra, ddec in zip(
            data[cameraID]["catalog_name"]["data"],
            data[cameraID]["ra"]["data"],
            data[cameraID]["dec"]["data"],
            data[cameraID]["dra"]["data"],
            data[cameraID]["ddec"]["data"],
        ):
            cat_source_ra = cat.data[cat.data["NAME"] == name]["RA"][0]
            cat_source_dec = cat.data[cat.data["NAME"] == name]["DEC"][0]
            errbox.append(
                int((np.abs(cat_source_ra - ra) < sigma*dra) and (np.abs(cat_source_dec - dec) < sigma*ddec)),
            )

    #sources_in_box = tuple(
    #    [(name, bool(inbox)) for name, inbox in zip(data[cameraID]["catalog_name"]["data"], errboxes[idx])]
    #    for idx, cameraID in enumerate(list(data.keys()))
    #)
    
    sources_in_box = {
        cameraID: {
            "sources": data[cameraID]["catalog_name"]["data"],
            "inbox": errbox,
        }
        for cameraID, errbox in zip(data.keys(), errboxes)
    }
    accuracy = tuple(
        np.array(errbox).sum()*100/len(data[cameraID]["catalog_name"]["data"]) for errbox in errboxes
    )
    return dict2df(sources_in_box), accuracy


def iros_radec_accuracy_finecoord(
    data: dict,
    catalogs: tuple[np.recarray],
    sigma: int | float,
) -> DataFrame: #tuple[list[tuple[str, bool]], tuple[float]]:
    """
    Analysis of the IROS reconstruction RA/DEC estimates and
    comparison with catalogs coordinates taking into account
    the fine dimension for each camera of the WFM.
    """
    source_in_errbox_camA = []
    source_in_errbox_camB = []
    errboxes = (source_in_errbox_camA, source_in_errbox_camB)

    camA, camB = data.keys()
    sources_ra = data[camB]["ra"]["data"]
    sources_dra = data[camB]["dra"]["data"]
    sources_dec = data[camA]["dec"]["data"]
    sources_ddec = data[camA]["ddec"]["data"]

    for cat, cameraID, errbox in zip(
        catalogs,
        (camA, camB),
        errboxes,
    ):
        print(f"# Checking {cameraID.upper()}...")
        for name, ra, dec, dra, ddec in zip(
            data[cameraID]["catalog_name"]["data"],
            sources_ra, sources_dec,
            sources_dra, sources_ddec,
        ):
            cat_source_ra = cat.data[cat.data["NAME"] == name]["RA"][0]
            cat_source_dec = cat.data[cat.data["NAME"] == name]["DEC"][0]
            errbox.append(
                int((np.abs(cat_source_ra - ra) < sigma*dra) & (np.abs(cat_source_dec - dec) < sigma*ddec)),
            )
    
    #sources_in_box = tuple(
    #    [(name, bool(inbox)) for name, inbox in zip(data[cameraID]["catalog_name"]["data"], errboxes[idx])]
    #    for idx, cameraID in enumerate(list(data.keys()))
    #)
    
    sources_in_box = {
        cameraID: {
            "sources": data[cameraID]["catalog_name"]["data"],
            "inbox": errbox,
        }
        for cameraID, errbox in zip(data.keys(), errboxes)
    }
    accuracy = tuple(
        np.array(errbox).sum()*100/len(data[cameraID]["catalog_name"]["data"]) for errbox in errboxes
    )
    return dict2df(sources_in_box), accuracy


def iros_fluence(
    data: dict,
    sdls: tuple[SimulationDataLoader],
    catalogs: tuple[np.recarray],
) -> DataFrame:
    """
    Checks fluences.
    """
    db = {
        cameraID: deepcopy({
            "sources": [],
            "optm_wrt_fits": [],
            "obs_wrt_fits": [],
            "sub_wrt_fits": [],
            "retrv_wrt_fits": [],
        })
        for cameraID in data.keys()
    }

    for sdl, cat, cameraID in zip(
        sdls, catalogs, data.keys(),
    ):
        print(f"# Checking {cameraID.upper()}...")
        for name, f, of, sf, sp in zip(
            data[cameraID]["catalog_name"]["data"],
            data[cameraID]["fluence"]["data"],
            data[cameraID]["obs_fluence"]["data"],
            data[cameraID]["sub_fluence"]["data"],
            data[cameraID]["simulphotons"]["data"],
        ):
            cat_source_ra = cat.data[cat.data["NAME"] == name]["RA"][0]
            cat_source_dec = cat.data[cat.data["NAME"] == name]["DEC"][0]
            # conv float64 to float32
            total_phs_simulated = len(
                sdl.data[(np.abs(sdl.data["RA"] - cat_source_ra) < 1e-7) & (np.abs(sdl.data["DEC"] - cat_source_dec) < 1e-7)]
            )
            params = (
                name,
                f * 100 / total_phs_simulated,
                of * 100 / total_phs_simulated,
                sf * 100 / total_phs_simulated,
                sp * 100 / total_phs_simulated,
            )
            for key, param in zip(
                db[cameraID].keys(), params,
            ):
                db[cameraID][key].append(param)
    
    return dict2df(db)


def check_sources_res(
    data: dict,
    catalogs: tuple[np.recarray],
) -> DataFrame:
    """
    Check residues.
    """
    def source_res(
        catalog: np.recarray,
        source_name: str,
        ra: float,
        dec: float,
    ) -> float:
        """Computes the RA/DEC residues wrt catalog position."""
        cat_source_ra = catalog.data[catalog.data["NAME"] == source_name]["RA"][0]
        cat_source_dec = catalog.data[catalog.data["NAME"] == source_name]["DEC"][0]
        return ra - cat_source_ra, dec - cat_source_dec
    
    ra_res = ([], [])
    dec_res = ([], [])
        
    for idx, cameraID, cat in zip(
        (0, 1), data.keys(), catalogs,
    ):
        print(f"# Checking {cameraID.upper()}...")
        for name, ra, dec in zip(
            data[cameraID]["catalog_name"]["data"],
            data[cameraID]["ra"]["data"],
            data[cameraID]["dec"]["data"],
        ):
            rra, rdec = source_res(cat, name, ra, dec)
            ra_res[idx].append(rra); dec_res[idx].append(rdec)
    
    df = dict2df(
        {cameraID: {
                "sources": data[cameraID]["catalog_name"]["data"],
                "ra_res": np.array(ra_res[idx]),
                "dec_res": np.array(dec_res[idx]),
            }
        for idx, cameraID in enumerate(data.keys())
        }
    )
    return df


# end