from setuptools import setup, find_packages

setup(
    name = "answers_680",
    version="1.0.0",
    description = "Answers to the exercises in labs_680",
    author = "Thomas Winckelman",
    author_email = "winckelman@tamu.edu",
    py_modules = [ f"answers_week_{j+1}" for j in range(15) ],  # List of top-level Python files (without .py)
    packages = find_packages()
)
