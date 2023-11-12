"""
Embeds natural language into machine language and then finds similarities
"""

import logging
from pathlib import Path
# Third-party packages
import numpy as np
import pandas as pd
import tensorflow_text as tftext
import tensorflow_hub as hub
import warnings
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
# Local packages
from drapi.drapi import getTimestamp, successiveParents, makeDirPath

# Arguments
DATA_PATH = Path("data/output/concatenateData/2023-10-15 19-06-59/Profile Contents.CSV")

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
    `DATA_PATH`: "{DATA_PATH}"

    # Arguments: General
    `PROJECT_DIR_DEPTH`: "{PROJECT_DIR_DEPTH}"

    `LOG_LEVEL` = "{LOG_LEVEL}"
    """)

    # Script
    warnings.filterwarnings(action='ignore')

    data = pd.read_csv(DATA_PATH, index_col=0)
    _ = tftext
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
    embeddings = {}
    lendata = len(data)
    for it, (profileID, row) in enumerate(data.iterrows(), start=1):
        embeddedProfile = {}
        if it % 100 == 0:
            logger.info(f"""Working on profile {it:,} of {lendata:,}.""")
        for textTitle, text in row.items():
            if pd.isna(text):
                text = ""
            else:
                pass
            embeddedProfile[textTitle] = embed(text)
        embeddings[profileID] = embeddedProfile

    # NOTE The structure of `embeddings`
    # embeddings = {ladyID: {"Character": tf.Tensor,
    #                        "Interests": tf.Tensor,
    #                        "Her Type of Man": tf.Tensor}}

    # Find similarties
    embeddingsList = []
    for ladyID, profileDict in embeddings.items():
        embeddingsList.append((ladyID, profileDict))
    embeddingsList = sorted(embeddingsList, key=lambda tu: tu[0])

    arr1 = np.array([di["Character"][0] for (_, di) in embeddingsList])
    arr2 = np.array([di["Interests"][0] for (_, di) in embeddingsList])
    arr3 = np.array([di["Her Type of Man"][0] for (_, di) in embeddingsList])

    # Calculate similarity based on Euclidean distance
    sim1 = euclidean_distances(arr1)
    indices1 = np.vstack([np.argsort(-arr) for arr in sim1])

    # Calculate similarity based on cosine distance
    cos_sim1 = cosine_similarity(arr1)
    cos_indices1 = np.vstack([np.argsort(-arr) for arr in cos_sim1])

    # Find most similar profile for each case
    indices = {}
    for idx, (ladyID, embeddedProfile) in enumerate(embeddingsList):
        di = {}
        di['euclidean'] = indices1[idx][1:21]
        di['cosine'] = cos_indices1[idx][1:21]
        indices[ladyID] = di
    indices = pd.DataFrame.from_dict(indices, orient="index").sort_index()

    # End script
    logging.info(f"""Finished running "{thisFilePath.relative_to(projectDir)}".""")
