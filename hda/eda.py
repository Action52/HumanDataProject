from .config import DATA_FOLDER, TEST_FILE
from scipy.io import loadmat
import pandas as pd





def load_sample_ecg(filename: str, folder:str = DATA_FOLDER, as_pandas=False):
    mat = loadmat(file_name=f"{folder}{filename}")
    if not as_pandas:
        return mat
    electrode_data = mat['o'][0][0][
        5]  # Modify the index as per your data structure
    # Create a DataFrame
    df = pd.DataFrame(electrode_data)

    # Add column names based on the electrode order in the description
    electrode_names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1",
                       "O2", "A1", "A2", "F7", "F8", "T3", "T4", "T5", "T6",
                       "Fz", "Cz", "Pz", "X3"]
    df.columns = electrode_names[:df.shape[
        1]]  # Adjust the slicing as per the number of columns in your
    return df


def main():
    mat = load_sample_ecg(filename=TEST_FILE, folder=DATA_FOLDER)
    print(mat, type(mat))
    electrode_data = mat['o'][0][0][
        5]  # Modify the index as per your data structure

    # Create a DataFrame
    df = pd.DataFrame(electrode_data)

    # Add column names based on the electrode order in the description
    electrode_names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1",
                       "O2", "A1", "A2", "F7", "F8", "T3", "T4", "T5", "T6",
                       "Fz", "Cz", "Pz", "X3"]
    df.columns = electrode_names[:df.shape[
        1]]  # Adjust the slicing as per the number of columns in your data

    print(df)


if __name__ == '__main__':
    main()
