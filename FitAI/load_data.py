from pandas import read_csv


#: Simple - use pandas.read_csv to load in csv data
#: Set column names to specified names
def load_file(filename, names, skiprows=0):
    data = read_csv(filename, names=names, skiprows=skiprows)
    return data
