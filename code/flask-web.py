import pickle
# Third-party packacges
import pandas as pd
from flask import Flask, request, render_template

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
        selected_metric = request.form.get('selected_metric')
        # selectedLadyData = next(ladyID for ladyID in indices if ladyID == selectedLadyID)
        selectedLadyData = indices[selectedLadyID]
        similarLadies = [data["Lady ID"][idx] for idx in selectedLadyData[selected_metric]]
        mask = data["Lady ID"].isin(similarLadies)
        similarProfiles = data["Character"][mask]
        return render_template('template.html',
                               list_ladies=list_ladies,
                               selectedLadyID=selectedLadyID,
                               selectedLadyData=selectedLadyData,
                               chosenProfile=data['Character'][data["Lady ID"] == selectedLadyID],
                               similarLadies=similarLadies[:],
                               similarProfiles=similarProfiles)
    else:
        return render_template('template.html', list_ladies=list_ladies)


if __name__ == '__main__':
    app.run(debug=True)
