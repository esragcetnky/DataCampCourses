# DataCamp Courses Code Repository

This repository contains the code I worked on while studying various DataCamp courses. Each folder represents a different course, and within these folders, you'll find the necessary scripts, data, and dependencies to reproduce the projects.

## Certificates

- [Introduction to Network Analysis in Python](https://www.datacamp.com/completed/statement-of-accomplishment/course/9578189579175f7bcfc7c4e830bad5462543ff46)
- [Introduction to Embeddings with the OpenAI API](https://www.datacamp.com/completed/statement-of-accomplishment/course/438991e5c0485a081d71763bd448f80c39249003)
- [Developing LLM Applications with LangChain](https://www.datacamp.com/completed/statement-of-accomplishment/course/6617b2439bdfe635fc1c53cede9bb13ad2c62784)
- [Intermediate Network Analysis in Python](https://www.datacamp.com/completed/statement-of-accomplishment/course/e59b47cb69928b1ba0195d0141dc0c5bd756567d)
- [Cluster Analysis in Python](https://www.datacamp.com/completed/statement-of-accomplishment/course/f31589eec4e9a09dd898f943253f74bf7d7bc611)

## Repository Structure

The repository is organized as follows:

- **Course Folders**: Each folder corresponds to a specific DataCamp course. The folder names are self-explanatory and indicate their course or topic (e.g., `Introduction_to_Machine_Learning`, `Data_Visualization`, etc.).
  
- **requirements.txt**: Each course folder includes a `requirements.txt` file that lists the Python packages and dependencies required to run the code for that course. You can install the required packages by running:
  
      pip install -r requirements.txt

- **data**: Some course folders have a data directory that contains the datasets used in the course exercises or projects. The data is usually in CSV format but may vary depending on the course.

- **credentials.yml**: I use a credentials.yml file to manage API keys needed for certain projects securely (e.g. when accessing APIs in courses related to data extraction or machine learning with external services). You will need to create your credentials.yml file to run those projects. The file should follow this structure:

      openai_api_key: your_api_key
      huggingface_api_key: another_key

# Getting Started
## Prerequisites
Make sure you have Python 3.7+ installed. Then follow these steps to get the repository set up:

Clone this repository:

    git clone https://github.com/your_username/your_repo_name.git
    cd your_repo_name

## Install the required dependencies for each course:

    pip install -r path/to/course_folder/requirements.txt

If a course requires API keys, you can just set up your credentials.yml file as described above.

Run the scripts for each course by navigating to the relevant folder and executing the code.

Example
For example, to run the code for the "Introduction to Machine Learning" course:

    cd Introduction_to_Machine_Learning
    pip install -r requirements.txt
    python main_script.py

# Notes
Each folder's requirements.txt is specific to that course and may contain overlapping dependencies. It's recommended to use virtual environments to manage dependencies:
      
      python -m venv env_name
      source env_name/bin/activate
  
The data folder in each course directory is optional and may not exist if the course does not use external data files.

# Contribution
Feel free to open an issue or submit a pull request if you find a bug or have suggestions for improvement.


# Contact 
For any questions or suggestions, feel free to reach out:

- Email: esragcetinkaya@gmail.com
- Linkedin : [esragcetinkaya](https://www.linkedin.com/in/esra-gul-cetinkaya/?locale=en_US)
