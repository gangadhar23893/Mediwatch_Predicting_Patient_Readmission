from setuptools import find_packages,setup
from typing import List


#function to return list of all requirements from requirements.txt 

def get_requirements() -> List[str]:

    requirements_lst: list[str] = []

    try:
        with open("requirements.txt","r") as file:
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()

                if requirement and requirement !='-e .':
                    requirements_lst.append(requirement)

    except FileNotFoundError:

        print("Requirements.txt file not found")

    return requirements_lst



setup(
    name = "Diabestes_readmission_prediction",
    version = "0.0.1",
    author= 'gangadhar',
    author_email= "gangadhar23893@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements()
)