import pickle
# Third-party packacges
import pandas as pd
from flask import Flask, jsonify, request, render_template

app = Flask(__name__, template_folder="flaskTemplates")

# load data and extract all the indices
with open('data/output/embedData/2023-11-12 18-28-07/Indices.pkl', 'rb') as f:
    indices0 = pickle.load(f)

indices = indices0["Character"]
list_ladies = sorted([ladyID for ladyID in indices.keys()])

# Load profile text
data = pd.read_csv("data/output/concatenateData/2023-10-15 19-06-59/Profile Contents.CSV")


@app.route("/", methods=['GET', 'POST'])
def template_test():
    if request.method == 'POST':
        selectedLadyID = int(request.form.get('selectedLadyID'))
        selectedMetric = request.form.get('selectedMetric')
        selectedLadyData = indices[selectedLadyID]
        similarLadies = [data["Lady ID"][idx] for idx in selectedLadyData[selectedMetric]]
        mask = data["Lady ID"].isin(similarLadies)
        similarProfiles = data["Character"][mask]
        similarData = zip(similarLadies, similarProfiles)
        return render_template('template.html',
                               list_ladies=list_ladies,
                               selectedLadyID=selectedLadyID,
                               selectedLadyData=selectedLadyData,
                               chosenProfile=data['Character'][data["Lady ID"] == selectedLadyID],
                               similarLadies=similarLadies[:],
                               similarProfiles=similarProfiles,
                               similarData=similarData)
    else:
        return render_template('template.html', list_ladies=list_ladies)


@app.route("/recommendations", methods=["GET"])
def get_recommendations():
    selectedLadyID = request.args.get("ladyID", default=38254, type=int)
    selectedMetric = request.args.get("metric", default="cosine", type=str)
    if not selectedLadyID:
        return jsonify("Missing lady ID"), 400
    elif selectedMetric not in ["cosine", "euclidean"]:
        return jsonify("Distance metric can only be cosine or Euclidean"), 400
    elif selectedLadyID not in list_ladies:
        return jsonify("This lady ID is not in our database"), 400
    else:
        try:
            selectedLadyData = indices[selectedLadyID]
            similarLadies = [int(data["Lady ID"][idx]) for idx in selectedLadyData[selectedMetric]]
            mask = data["Lady ID"].isin(similarLadies)
            similarProfiles = data["Character"][mask].to_list()
            similarData = {ladyID: text for ladyID, text in zip(similarLadies, similarProfiles)}
            return jsonify(similarData), 200
        except Exception as e:
            return jsonify(str(e)), 500


if __name__ == '__main__':
    app.run(debug=True)
