from flask import Flask, render_template, request, jsonify
import numpy as np
import os

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

        # --- Арифметические операции ---
        if op == "add":
            result = (A + B).tolist()
        elif op == "sub":
            result = (A - B).tolist()
        elif op == "mul":
            result = np.dot(A, B).tolist()

        # --- Определители ---
        elif op == "det_a":
            result = [[round(np.linalg.det(A), 3)]]
        elif op == "det_b":
            result = [[round(np.linalg.det(B), 3)]]

        # --- Ранги ---
        elif op == "rankA":
            result = [[np.linalg.matrix_rank(A)]]
        elif op == "rankB":
            result = [[np.linalg.matrix_rank(B)]]

        # --- Обратные матрицы ---
        elif op == "invA":
            if np.linalg.det(A) == 0:
                return jsonify({"error": "Матрица A вырождена, обратной не существует"})
            result = np.linalg.inv(A).tolist()
        elif op == "invB":
            if np.linalg.det(B) == 0:
                return jsonify({"error": "Матрица B вырождена, обратной не существует"})
            result = np.linalg.inv(B).tolist()

        else:
            return jsonify({"error": "Неизвестная операция"})

        # --- Округление и целые числа ---
        for i in range(len(result)):
            if isinstance(result[i], list):
                result[i] = [int(x) if abs(x - round(x)) < 1e-9 else round(x, 3) for x in result[i]]
            else:
                result[i] = int(result[i]) if abs(result[i] - round(result[i])) < 1e-9 else round(result[i], 3)

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

