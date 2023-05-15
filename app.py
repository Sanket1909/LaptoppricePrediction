from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# import the model
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html', df=df)
    

@app.route('/predict', methods=['POST'])
def predict():
    # get the form data
    company = request.form['company']
    type = request.form['type']
    ram = int(request.form['ram'])
    weight = float(request.form['weight'])
    touchscreen = request.form['touchscreen']
    ips = request.form['ips']
    screen_size = float(request.form['screen_size'])
    resolution = request.form['resolution']
    cpu = request.form['cpu']
    hdd = int(request.form['hdd'])
    ssd = int(request.form['ssd'])
    gpu = request.form['gpu']
    os = request.form['os']
    


    # preprocess the data
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

    query = query.reshape(1, 12)

    # predict the price
    price = int(np.exp(pipe.predict(query)[0]))

    # render the predicted price
    return render_template('result.html', price=price)

if __name__ == '__main__':
    app.run(debug=True)



