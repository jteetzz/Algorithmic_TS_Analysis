def make_rul_categories(df):
    df = df.copy()

    q10 = df["rul"].quantile(0.10)
    q50 = df["rul"].quantile(0.50)
    q90 = df["rul"].quantile(0.90)

    def label_rul(x):
        if x < q10:
            return "Extremely Low RUL"
        elif x < q50:
            return "Moderately Low RUL"
        elif x < q90:
            return "Moderately High RUL"
        else:
            return "Extremely High RUL"

    df["rul_class"] = df["rul"].apply(label_rul)

    quantiles = {
        "Q10": q10,
        "Q50": q50,
        "Q90": q90
    }

    return df, quantiles


def select_10_sensors(sensor_cols):
    # systematic selection: evenly spaced across detected sensors
    if len(sensor_cols) <= 10:
        return sensor_cols

    step = len(sensor_cols) / 10
    selected = []
    for i in range(10):
        idx = int(i * step)
        selected.append(sensor_cols[idx])

    # remove duplicates if any
    selected = list(dict.fromkeys(selected))

    # if fewer than 10 due to duplicates, fill from remaining sensors
    for s in sensor_cols:
        if len(selected) == 10:
            break
        if s not in selected:
            selected.append(s)

    return selected
