Step 0:
A few words of caution: 
1) Read all the way through the instructions. 
2) Read all the way through the jupyter notebook.
2) Models must be deployed as an API using Python; please use port 1313.
3) No additional data may be added or used. 
5) The variables returned by the API should be the class probability (i.e. decimal value) for belonging to the positive class, the variables used in the model, and the predicted class (defined by the business partner). The results should be in a JSON format.

Step 1:
Prepare the model deployment for production: Update your code to meet common production coding standards and best practices. These include modularization, code quality, proper unit testing, and comments/documentation. The code will be evaluated using tooling that evaluates code coverage and code quality. Note: The deployed code should conform to the expectations of the business partner. Also, the model has been approved by the business partner, so the final GLM equation should not be modified.

Step 2:
Wrap the model code inside an API: The model must be made callable via API call (port 1313). The call will pass 1 to N rows of data in JSON format, and expects a N responses each with a predicted class and probability belonging to the predicted class. 

Here is an example curl call to your API:

curl --request POST --url http://localhost:8080/predict --header 'content-type: application/json' --data '{"x0": "-1.018506", "x1": "-4.180869", "x2": "5.70305872366547", "x3": "-0.522021597308617", ...,"x99": "2.55535888"}'

or a batch curl call:

curl --request POST --url http://localhost:8080/predict --header 'content-type: application/json' --data '[{"x0": "-1.018506", "x1": "-4.180869", "x2": "5.70305872366547", "x3": "-0.522021597308617", ...,"x99": "2.55535888"},{"x0": "-1.018506", "x1": "-4.180869", "x2": "5.70305872366547", "x3": "-0.522021597308617", ...,"x99": "2.55535888"}]'

Each of the 10,000 rows in the test dataset will be passed through an API call. The call could be a single batch call w/ all 10,000 rows, or 10,000 individual calls. API should be able to handle either case with minimal impact to performance. Reminder: The predictions returned by the API should be the class probability (i.e. decimal value) for belonging to the positive class, the variables used in the model, and the predicted class (defined by the business partner). The results should be in a JSON format.

Step 3:
Wrap your API in a Docker image: Create a Dockerfile that builds your API into an image. Write a shell script titled run_api.sh that either runs your image using traditional docker run commands or orchestrates your deployment using Compose, Swarm or Kubernetes (include relevant *.yml config files). 

Step 4:
Optimize your deployment for enterprise production and scalability: Identify opportunities to optimize your deployment for scalability. Consider how your API might handle a large number of calls (thousands per minute). What additional steps/tech could you add to your deployment in order to make it scalable for enterprise level production. You can incorporate any relevant code (optional), or you can describe your steps in the write-up as part of Step 5. 

Step 5:
Submit your work: Please submit all of your code, including relevant python files for the API, data prep, model build, Dockerfile (if relevant, orchestration config files), startup shell script, and a brief write-up documenting justification for your end-to-end process in PDF format. Recommend to tar or zip all files into a single archive for submission.  

Please do not submit the original data back to us. Your work will be scored on accuracy of the implementation of the model, API performance and scalability, and code quality and coverage.

Potential Checklist of Files:
* python code, including tests and API  
* pickle file (if applicable)  
* Dockerfile to create a Docker image to run their code as a container . Base image needs to be readily available on DockerHub.  
* A write-up describing the end-to-end process  
* Optional:  
    * Orchestration configuration files for Docker Swarm or Kubernetes  
    * CI/CD pipeline configuration file  
    * requirements.txt file with python packages/versions
