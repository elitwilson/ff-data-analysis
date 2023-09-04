from flask import Flask, request, jsonify
from src.data_preprocessing import load_and_split_data
from src.model import train_and_evaluate

app = Flask(__name__)


# Initialize and train the model when the application starts
X_train, X_test, y_train, y_test = load_and_split_data()
mse, rmse, model = train_and_evaluate(X_train, y_train, X_test, y_test)

print("Model trained")


@app.route('/')
def index():
    return f"Mean Squared Error: {mse}, Root Mean Squared Error: {rmse}"


@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json['features']
    if not input_data:
        return jsonify({'error': 'Invalid input'}), 400

    # Assuming input_data is a list of feature values
    prediction = model.predict([input_data])[0]

    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(debug=True, port=6543)


