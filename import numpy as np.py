import numpy as np
from sklearn.ensemble import RandomForestClassifier

THRESHOLD = 1.5
WINDOW = 20


def make_features(history):

    x = np.array(history, dtype=float)

    features = []

    # Overall statistics
    features.extend([
        np.mean(x),
        np.median(x),
        np.std(x),
        np.min(x),
        np.max(x),
        np.mean(x > 1.5),
        np.mean(x < 2),
        np.mean(x >= 5),
    ])

    # Recent statistics
    for n in [5, 10, 20]:

        recent = x[-n:]

        features.extend([
            np.mean(recent),
            np.median(recent),
            np.std(recent),
            np.min(recent),
            np.max(recent),
            np.mean(recent > 1.5),
        ])

    # Last round information
    features.extend([
        x[-1],
        x[-1] - x[-2],
        np.mean(np.diff(x))
    ])

    # Current streak
    last_type = x[-1] > THRESHOLD
    streak = 0

    for value in x[::-1]:

        if (value > THRESHOLD) == last_type:
            streak += 1
        else:
            break

    features.extend([
        streak,
        float(last_type)
    ])

    return np.array(features)


def create_dataset(rounds):

    X = []
    y = []

    for i in range(WINDOW, len(rounds)):

        history = rounds[i-WINDOW:i]

        target = int(
            rounds[i] > THRESHOLD
        )

        X.append(
            make_features(history)
        )

        y.append(target)

    return np.array(X), np.array(y)


def train_ai(rounds):

    X, y = create_dataset(rounds)

    if len(X) < 50:

        print("\nNot enough data.")

        print(
            f"Give the AI at least "
            f"{WINDOW + 50} rounds."
        )

        return None

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X, y)

    return model


def predict(model, rounds):

    history = rounds[-WINDOW:]

    features = make_features(
        history
    ).reshape(1, -1)

    probabilities = model.predict_proba(
        features
    )[0]

    probability_above = probabilities[1]

    if probability_above >= 0.5:

        prediction = "> 1.5x"

    else:

        prediction = "<= 1.5x"

    print("\n")
    print("=" * 40)
    print("        AVIATOR AI")
    print("=" * 40)

    print(
        f"\nEstimated probability:"
    )

    print(
        f"> 1.5x : "
        f"{probability_above * 100:.2f}%"
    )

    print(
        f"<= 1.5x: "
        f"{(1 - probability_above) * 100:.2f}%"
    )

    print(
        f"\nAI estimate: {prediction}"
    )

    print("=" * 40)

    print(
        "\nIMPORTANT:"
    )

    print(
        "This is a statistical ML estimate."
    )

    print(
        "It cannot guarantee the next "
        "Aviator result."
    )


def main():

    print("=" * 40)
    print("        AVIATOR AI SYSTEM")
    print("=" * 40)

    print(
        "\nEnter historical multipliers."
    )

    print(
        "Example:"
    )

    print(
        "1.12,2.31,1.05,3.72,1.44"
    )

    text = input(
        "\nMultipliers: "
    )

    rounds = []

    for value in text.split(","):

        try:

            number = float(
                value.strip()
            )

            if number > 0:

                rounds.append(number)

        except ValueError:

            pass

    print(
        f"\nLoaded {len(rounds)} rounds."
    )

    if len(rounds) < WINDOW + 50:

        print(
            "\nYou need at least "
            f"{WINDOW + 50} rounds."
        )

        return

    model = train_ai(rounds)

    if model:

        predict(
            model,
            rounds
        )


if __name__ == "__main__":

    main()