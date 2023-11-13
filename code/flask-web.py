import pickle

from flask import Flask, request, render_template

app = Flask(__name__, template_folder="flaskTemplates")

# load data and extract all the vectors
with open('data/output/embedData/2023-11-12 18-28-07/Indices.pkl', 'rb') as f:
    indices0 = pickle.load(f)

indices = indices0["Character"]
list_ladies = sorted([ladyID for ladyID in indices.keys()])


@app.route("/", methods=['GET', 'POST'])
def template_test():
    if request.method == 'POST':
        selected_ladyID = request.form.get('selected_ladyID')
        selected_metric = request.form.get('selected_metric')
        selectedLadyData = next(ladyID for ladyID in indices if ladyID == selected_ladyID)
        similar_books = [indices[i] for i in selectedLadyData[selected_metric]]
        return render_template('template.html',
                               list_ladies=list_ladies,
                               book_selected=selectedLadyData,
                               similar_books=similar_books[:6])
    else:
        return render_template('template.html', list_ladies=list_ladies)


if __name__ == '__main__':
    app.run(debug=True)
