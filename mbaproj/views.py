import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from django.shortcuts import render

# Get the base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construct the path to the dataset file
dataset_path = os.path.join(BASE_DIR, 'datasets', 'data.csv')

# Load the dataset
data = pd.read_csv(dataset_path, encoding='latin-1')

# Define your models
features = ['GSF', 'No of Levels']
target1 = 'Concrete'
target2 = 'Structural Steel-MT'

model1 = LinearRegression()
model2 = LinearRegression()

model1.fit(data[features], data[target1])
model2.fit(data[features], data[target2])

def home_view(request):
    if request.method == 'POST':
        gsf = float(request.POST.get('gsf'))
        levels = int(request.POST.get('levels'))

        concrete_prediction = model1.predict([[gsf, levels]])
        steel_prediction = model2.predict([[gsf, levels]])

        # Calculate other variables based on the predictions
        cement_quantity = (concrete_prediction / 100) * 8 * 50
        global_warming_concrete = cement_quantity * 0.71
        nr_energy_concrete = cement_quantity * 4.04
        water_cons_concrete = cement_quantity * 0.000509
        waste_concrete = cement_quantity * 0.000105550333333333

        global_warming_steel = steel_prediction * 3040
        nr_energy_steel = steel_prediction * 3553.65
        water_cons_steel = steel_prediction * 4265.23
        waste_steel = steel_prediction * 17.01

        context = {
            'gsf': gsf,
            'levels': levels,
            'concrete_prediction': concrete_prediction[0],
            'steel_prediction': steel_prediction[0],
            'global_warming_concrete': global_warming_concrete[0],
            'nr_energy_concrete': nr_energy_concrete[0],
            'water_cons_concrete': water_cons_concrete[0],
            'waste_concrete': waste_concrete[0],
            'global_warming_steel': global_warming_steel[0],
            'nr_energy_steel': nr_energy_steel[0],
            'water_cons_steel': water_cons_steel[0],
            'waste_steel': waste_steel[0]
        }
        return render(request, 'result.html', context)

    return render(request, 'index.html')