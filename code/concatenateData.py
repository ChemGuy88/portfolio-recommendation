"""
Concatenates the data to be processed from the input folders.
"""

import logging
import re
from pathlib import Path
# Third-party packages
import pandas as pd
# Local packages
from drapi.drapi import getTimestamp, successiveParents, makeDirPath

# Arguments
INPUT_DIR_PATH = Path("data/input/scraperProfiles")
CENSUS_FILE_PATH = Path("data/input/Profile Links.CSV")

# Arguments: functions


def getLadyID(string):
    pattern = r"LadyID=(\d+)$"
    searchObj = re.search(pattern, string)
    if searchObj:
        ladyID = searchObj.groups()[0]
    else:
        ladyID = None
    return ladyID


# Arguments: Meta-variables
PROJECT_DIR_DEPTH = 2

ROOT_DIRECTORY = "PROJECT_OR_PORTION_DIRECTORY"  # TODO One of the following:
                                                 # ["IDR_DATA_REQUEST_DIRECTORY",    # noqa
                                                 #  "IRB_DIRECTORY",                 # noqa
                                                 #  "DATA_REQUEST_DIRECTORY",        # noqa
                                                 #  "PROJECT_OR_PORTION_DIRECTORY"]  # noqa

LOG_LEVEL = "INFO"

# Variables: Path construction: General
runTimestamp = getTimestamp()
thisFilePath = Path(__file__)
thisFileStem = thisFilePath.stem
projectDir, _ = successiveParents(thisFilePath.absolute(), PROJECT_DIR_DEPTH)
dataDir = projectDir.joinpath("data")
if dataDir:
    inputDataDir = dataDir.joinpath("input")
    outputDataDir = dataDir.joinpath("output")
    if outputDataDir:
        runOutputDir = outputDataDir.joinpath(thisFileStem, runTimestamp)
logsDir = projectDir.joinpath("logs")
if logsDir:
    runLogsDir = logsDir.joinpath(thisFileStem)
sqlDir = projectDir.joinpath("sql")

if ROOT_DIRECTORY == "PROJECT_OR_PORTION_DIRECTORY":
    rootDirectory = projectDir
else:
    raise Exception("An unexpected error occurred.")

# Variables: Path construction: Project-specific
pass

# Variables: Other
pass

# Directory creation: General
makeDirPath(runOutputDir)
makeDirPath(runLogsDir)

# Logging block
logpath = runLogsDir.joinpath(f"log {runTimestamp}.log")
logFormat = logging.Formatter("""[%(asctime)s][%(levelname)s](%(funcName)s): %(message)s""")

logger = logging.getLogger(__name__)

fileHandler = logging.FileHandler(logpath)
fileHandler.setLevel(9)
fileHandler.setFormatter(logFormat)

streamHandler = logging.StreamHandler()
streamHandler.setLevel(LOG_LEVEL)
streamHandler.setFormatter(logFormat)

logger.addHandler(fileHandler)
logger.addHandler(streamHandler)

logger.setLevel(9)

if __name__ == "__main__":
    logger.info(f"""Begin running "{thisFilePath}".""")
    logger.info(f"""All other paths will be reported in debugging relative to `{ROOT_DIRECTORY}`: "{rootDirectory}".""")
    logger.info(f"""Script arguments:


    # Arguments
    `INPUT_DIR_PATH`: "{INPUT_DIR_PATH}"
    `CENSUS_FILE_PATH`: "{CENSUS_FILE_PATH}"

    # Arguments: General
    `PROJECT_DIR_DEPTH`: "{PROJECT_DIR_DEPTH}"

    `LOG_LEVEL` = "{LOG_LEVEL}"
    """)

    # Concatenate input files
    profileContents = pd.DataFrame()
    for pathObj in INPUT_DIR_PATH.iterdir():
        if pathObj.is_dir():
            fpath = pathObj.joinpath("Profile Contents.CSV")
            df = pd.read_csv(fpath, index_col=0)
            profileContents = pd.concat([profileContents, df])
        else:
            pass

    # Load profile links
    census = pd.read_csv(CENSUS_FILE_PATH, index_col=0)
    census["Lady ID"] = census["href"].apply(getLadyID)

    # QA: Count duplicates in profile contents and census
    pclen0 = len(profileContents)
    profileContents = profileContents.drop_duplicates()
    pclen1 = len(profileContents)

    clen0 = len(census)
    census = census.drop_duplicates()
    clen1 = len(census)

    logger.info(f"""The number of scraped profiles before and after dropping duplicates, and the number dropped: {pclen0:,} - {pclen1:,} = {pclen0 - pclen1:,}.""")
    logger.info(f"""The number of profile links before and after dropping duplicates, and the number dropped: {clen0:,} - {clen1:,} = {clen0 - clen1:,}.""")

    # Set profile ID as integer type
    profileContents["Lady ID"] = profileContents["Lady ID"].astype(int)
    census["Lady ID"] = census["Lady ID"].astype(int)

    # Sort by profile ID and save tables
    profileContents = profileContents.set_index("Lady ID").sort_index()
    census = census.set_index("Lady ID").sort_index()

    fpath1 = runOutputDir.joinpath("Profile Contents.CSV")
    profileContents.to_csv(fpath1)

    fpath2 = runOutputDir.joinpath("Profile Links.CSV")
    census.to_csv(fpath2)

    # End script
    logging.info(f"""Finished running "{thisFilePath.relative_to(projectDir)}".""")
