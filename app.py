from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
from wtforms import Form, TextAreaField, validators
import pandas as pd

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))
victor=pickle.load(open('victor.pkl','rb'))
stopwords=pickle.load(open('stopwords.pkl','rb'))

#dd = pd.DataFrame(victor)

def classify(document):
    label = {0: 'negative', 1: 'positive'}
    X = victor.transform([document])
    y = model.predict(X)[0]
    proba = np.max(model.predict_proba(X))
    return label[y], proba


class ReviewForm(Form):
    moviereview = TextAreaField('',[validators.DataRequired(),validators.length(min=2)])

@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('1_ere_page.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        tweet = request.form['moviereview']
        y, proba = classify(tweet)
        return render_template('results.html',content=tweet,prediction=y,probability=round(proba*100, 2))
    return render_template('1_ere_page.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)


"""""
@app.route('/')
def hello_world():
    return render_template("analyse_sentiment.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    label = {0: 'negative', 1: 'positive'}
    entrée=[str(x) for x in request.form.values()]
    entrée_array=[np.array(entrée)]
    prediction=model.predict(entrée_array)

   
    if output>str(0.5):
        return render_template('analyse_sentiment.html',pred='le tweet est positif {}'.format(output))
        print ('hello')
    else:
        return render_template('analyse_sentiment.html',pred='le tweet est négatif {}'.format(output))
        print ('hello2')

if __name__ == '__main__':
    app.run()
"""