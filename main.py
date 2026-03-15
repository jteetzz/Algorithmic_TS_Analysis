from load_data import load_dataset


def main():

    df, sensors = load_dataset()

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nNumber of sensors:", len(sensors))


if __name__ == "__main__":
    main()
