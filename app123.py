from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/calculate", methods=["POST"])
def calculate():
    try:
        data = request.get_json()
        A = np.array(data.get("A"))
        B = np.array(data.get("B"))
        op = data.get("operation")

        if op == "add":
            result = (A + B).tolist()
        elif op == "sub":
            result = (A - B).tolist()
        elif op == "mul":
            result = np.dot(A, B).tolist()
        elif op == "rankA":
            result = [[np.linalg.matrix_rank(A)]]
        elif op == "rankB":
            result = [[np.linalg.matrix_rank(B)]]
        elif op == "invA":
            result = np.linalg.inv(A).tolist()
        elif op == "invB":
            result = np.linalg.inv(B).tolist()
        else:
            return jsonify({"error": "Неизвестная операция"})

        for i in range(len(result)):
            if isinstance(result[i], list):
                result[i] = [int(x) if abs(x - round(x)) < 1e-9 else round(x, 3) for x in result[i]]
            else:
                result[i] = int(result[i]) if abs(result[i] - round(result[i])) < 1e-9 else round(result[i], 3)

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
