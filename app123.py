from flask import Flask, render_template, request
import numpy as np
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        try:
            # получаем размеры
            size_a = int(request.form.get("size_a", 2))
            size_b = int(request.form.get("size_b", 2))

            # считываем матрицы из формы
            A = np.zeros((size_a, size_a))
            B = np.zeros((size_b, size_b))

            for i in range(size_a):
                for j in range(size_a):
                    A[i, j] = float(request.form.get(f"A{i}{j}", 0))

            for i in range(size_b):
                for j in range(size_b):
                    B[i, j] = float(request.form.get(f"B{i}{j}", 0))

            operation = request.form.get("operation")

            # выполняем операции
            if operation == "add":
                if A.shape != B.shape:
                    raise ValueError("Матрицы должны быть одинакового размера для сложения.")
                result = A + B

            elif operation == "subtract":
                if A.shape != B.shape:
                    raise ValueError("Матрицы должны быть одинакового размера для вычитания.")
                result = A - B

            elif operation == "multiply":
                if A.shape[1] != B.shape[0]:
                    raise ValueError("Число столбцов A должно равняться числу строк B для умножения.")
                result = A.dot(B)

            elif operation == "inverse_A":
                if A.shape[0] != A.shape[1]:
                    raise ValueError("Матрица A должна быть квадратной.")
                result = np.linalg.inv(A)

            elif operation == "inverse_B":
                if B.shape[0] != B.shape[1]:
                    raise ValueError("Матрица B должна быть квадратной.")
                result = np.linalg.inv(B)

            elif operation == "rank_A":
                result = np.linalg.matrix_rank(A)

            elif operation == "rank_B":
                result = np.linalg.matrix_rank(B)

            elif operation == "divisor_A":
                A_int = A.astype(int)
                divisor_A = np.gcd.reduce(A_int.flatten())
                result = f"Наибольший общий делитель элементов матрицы A: {divisor_A}"

            elif operation == "divisor_B":
                B_int = B.astype(int)
                divisor_B = np.gcd.reduce(B_int.flatten())
                result = f"Наибольший общий делитель элементов матрицы B: {divisor_B}"

            else:
                raise ValueError("Неизвестная операция.")

        except Exception as err:
            error = str(err)

    return render_template("index.html",
                           result=result,
                           error=error)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
