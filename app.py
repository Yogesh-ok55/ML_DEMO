from flask import Flask, render_template,request
import numpy as np
import pickle
model=pickle.load(open('model.pkl','rb'))
app=Flask(__name__)

@app.route('/')
def hello_world():
    return "Hello users How are you navigate to /about to calculate sum of two numbers"

@app.route('/about',methods=['GET','POST'])
def hello_world1():
    if request.method=='POST':
         int_features=[int(x) for x in request.form.values()]
         final=[np.array(int_features)]
        
         prediction=model.predict(final)
         output='{0:.1f}'.format(prediction[0])
         return render_template("reply.html",Name=output)
    return render_template("index.html")

if __name__=="__main__":
    app.run(debug=True,port=8000)
