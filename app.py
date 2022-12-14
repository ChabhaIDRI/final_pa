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

class Tweet_Form(Form):
    tweet_analyser = TextAreaField('',[validators.DataRequired(),validators.length(min=2)])
    
def resultat_rl(txt):
    label = {0: 'negative', 1: 'positive'}
    X = victor.transform([txt])
    y = model.predict(X)[0]
    probabilite = np.max(model.predict_proba(X))
    return label[y], probabilite


@app.route('/')
def index():
    form = Tweet_Form(request.form)
    return render_template('1_ere_page.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = Tweet_Form(request.form)
    if request.method == 'POST' and form.validate():
        tweet = request.form['tweet_analyser']
        y, probabilite = resultat_rl(tweet)
        return render_template('results.html',content=tweet,prediction=y,probability=round(probabilite*100, 2))
    return render_template('1_ere_page.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)

