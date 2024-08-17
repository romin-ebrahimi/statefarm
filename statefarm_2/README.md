# State Farm ML Engineer

### Contents
- `run_api.sh` - bash script that runs the necessary commands to launch the Docker image and the model API.
- `Dockerfile` - initialization of docker container for hosting fast API.
- `main.py` - main module that contains all of the code needed to run the model API on port 1313. The variables returned by the API should be the decimal value class probability for the positive class `phat`, the variables used in the model, and the predicted class `business_outcome` $\in \{ 0,1 \}$, which is defined by the business partner.
- `model_train.py` - module for training and saving the model pipeline as a .pkl for deployment.
- `GLM Model 26.ipynb` - the given jupyter notebook that contains the model that the business wants to deploy.
- `data` - directory containing the training and test data.
- `tests` - contains unittesting modules.

### Overview

The Docker container hosting the model API on port 1313 accepts POST calls and can return an array of N elements containing the
`business_outcome`, estimated probabilities `phat`, and the variables passed to the model.

Here is an example curl call to your API:

curl --request POST --url http://localhost:1313/predict --header 'content-type: application/json' --data '{"x0": "-1.018506", "x1": "-4.180869", "x2": "5.70305872366547", "x3": "-0.522021597308617", ...,"x99": "2.55535888"}'

or a batch curl call:

curl --request POST --url http://localhost:1313/predict --header 'content-type: application/json' --data '[{"x0": "-1.018506", "x1": "-4.180869", "x2": "5.70305872366547", "x3": "-0.522021597308617", ...,"x99": "2.55535888"},{"x0": "-1.018506", "x1": "-4.180869", "x2": "5.70305872366547", "x3": "-0.522021597308617", ...,"x99": "2.55535888"}]'

Each of the 10,000 rows in the test dataset will be passed through an API call. The call could be a single batch call w/ all 10,000 rows, or 10,000 individual calls. API should be able to handle either case with minimal impact to performance. Reminder: The predictions returned by the API should be the class probability (i.e. decimal value) for belonging to the positive class, the variables used in the model, and the predicted class (defined by the business partner). The results should be in a JSON format.

The returned payload will be in alphabetical order e.g. [{{"business_outcome": 0}, {"phat": 0.40}, {"x0": "-1.018506", "x1": "-4.180869", ...,
"x99": "2.55535888"}}]

The `business_outcome` uses a probability cutoff of 75th percentile as defined by the exploratory notebook `GLM Model 26`. This cutoff is 
hardcoded as the probability >= `0.712`.

### Setup Git Hooks
The files `.flake8`, `pyproject.toml`, and `.pre-commit-config.yaml` are used with black to autoformat code
to meet code style guidelines. e.g. line length must be <= 80. In order to use this, follow these steps:
1. Within the project repo, run `pre-commit install`.
2. Then run `pre-commit autoupdate`.
3. To run pre-commit git hooks for flake8 and black run use 
`pre-commit run --all-files`.
