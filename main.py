import os

# this is the main file
from web.read_data import ReadData

if __name__ == '__main__':
    os.system("streamlit run "+os.path.join("web", "fronted.py"))
