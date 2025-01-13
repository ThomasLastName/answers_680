from setuptools import setup

setup(
    name = "answers_680",
    version="1.0.0",
    description = "Answers to the exercises in labs_680",
    author = "Thomas Winckelman",
    author_email = "winckelman@tamu.edu",
    packages = [ "answers_680" ],
    py_modules = [ f"answers_week_{j+1}" for j in range(15) ],
    packages = find_packages(),
    include_package_data = True
)
