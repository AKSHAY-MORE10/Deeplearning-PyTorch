from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

users = []


@app.route("/users/new", methods=["GET"])
def new_user_form():
    return render_template("users_new.html")


@app.route("/users", methods=["POST"])
def create_user():
    if request.is_json:
        data = request.get_json()
    else:
        data = {
            "name": request.form.get("name"),
            "age": request.form.get("age"),
        }

        if data["age"] is not None:
            try:
                data["age"] = int(data["age"])
            except ValueError:
                return jsonify({"error": "Age must be a number"}), 400

    if not data or not data.get("name"):
        return jsonify({"error": "Name is required"}), 400

    users.append(data)

    if request.is_json:
        return jsonify({"message": "User added"})

    return render_template("users_new.html", message="User added", users=users)


@app.route("/users", methods=["GET"])
def get_users():
    return render_template("users_new.html", users=users)


@app.route("/users/<int:index>", methods=["PUT"])
def update_user(index):
    users[index] = request.get_json()

    return jsonify({"message": "User updated"})


@app.route("/users/<int:index>", methods=["DELETE"])
def delete_user(index):
    users.pop(index)

    return jsonify({"message": "User deleted"})


if __name__ == "__main__":
    app.run(debug=True)