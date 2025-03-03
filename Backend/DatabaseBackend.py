import sqlite3
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})  # Allow React app on port 3000

# Connect to the database and fetch all records
def get_all_predictions():
    conn = sqlite3.connect('predictions.db')  # Adjust path as necessary
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM predictions')
    records = cursor.fetchall()
    conn.close()
    return records

@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    records = get_all_predictions()
    response = []
    for row in records:
        response.append({
            'id': row[0],
            'date': row[1],
            'time': row[2],
            'platform': row[3],
            'status': row[4],
            'confidence': row[5],
            'media_format': row[6]  # Fetch the new media_format column
        })
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Change to port 5001
