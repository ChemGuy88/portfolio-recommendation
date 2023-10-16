"""
Embeds natural language into machine language.
"""

import logging
from pathlib import Path
# Third-party packages
import pandas as pd
import warnings
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
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
    sentences = {"Character": [],
                 "Interests": [],
                 "Her Type of Man": []}
    lendata = len(data)
    for it, (profileID, row) in enumerate(data.iterrows(), start=1):
        if it % 100 == 0:
            logger.info(f"""Working on profile {it:,} of {lendata:,}.""")
        for textTitle, text in row.items():
            if pd.isna(text):
                text = ""
            else:
                pass
            for i in sent_tokenize(text):
                temp = []

                for j in word_tokenize(i):
                    temp.append(j.lower())

            sentences[textTitle].append(temp)

    # Create model
    model1a = Word2Vec(sentences=sentences["Character"],
                       min_count=1,
                       vector_size=100,
                       window=5)
    model2a = Word2Vec(sentences=sentences["Character"],
                       min_count=1,
                       vector_size=100,
                       window=5,
                       sg=1)
    
    # TODO
    # Find similarties
    if False:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

        vectors = [item['vector'] for item in full_data]
        X = np.array(vectors)

        # calculate similarity based on Euclidean distance
        sim = euclidean_distances(X)
        indices = np.vstack([np.argsort(-arr) for arr in sim])

        # calculate similarity based on cosine distance
        cos_sim = cosine_similarity(X)
        cos_indices = np.vstack([np.argsort(-arr) for arr in cos_sim])

        # find most similar books for each case
        for i, book in enumerate(full_data):
            book['euclidean'] = indices[i][1:21]
            book['cosine'] = cos_indices[i][1:21]

    # End script
    logging.info(f"""Finished running "{thisFilePath.relative_to(projectDir)}".""")
