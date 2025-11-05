import pandas as pd


def load_data(file_path):
    df = pd.read_excel(file_path)
    df.head()


load_data("data_motobikes.xlsx")
