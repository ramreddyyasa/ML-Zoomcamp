# Airline-Passenger-Satisfaction
- I have worked on this project as part of the ML-Zoomcamp MidTerm project conducted by <a href="https://github.com/alexeygrigorev">Alexey Grigorev</a>. You can refer to <a href="https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp">ML-Zoomcamp</a> to know more about it.
### Dataset Description:

- This dataset contains an airline passenger satisfaction survey. 

- Attributes: 
    
    - **Categorical Attributes:**
    
        - Gender: Gender of the passengers (Female, Male)
        - Customer Type: The customer type (Loyal customer, disloyal customer)
        - Type of Travel: Purpose of the flight of the passengers (Personal Travel, Business Travel)
 
    - **Ordinal Attributes:(0:Not Applicable;1-5)**
        
        - Class: Travel class in the plane of the passengers (Business, Eco, Eco Plus)
        - Inflight wifi service: Satisfaction level of the inflight wifi service 
        - Departure/Arrival time convenient: Satisfaction level of Departure/Arrival time convenient
        - Ease of Online booking: Satisfaction level of online booking
        - Gate location: Satisfaction level of Gate location
        - Food and drink: Satisfaction level of Food and drink
        - Online boarding: Satisfaction level of online boarding
        - Seat comfort: Satisfaction level of Seat comfort
        - Inflight entertainment: Satisfaction level of inflight entertainment
        - On-board service: Satisfaction level of On-board service
        - Leg room service: Satisfaction level of Leg room service
        - Baggage handling: Satisfaction level of baggage handling
        - Check-in service: Satisfaction level of Check-in service
        - Inflight service: Satisfaction level of inflight service
        - Cleanliness: Satisfaction level of Cleanliness
    
    - **Numerical Attributes:**
    
        - Age: The actual age of the passengers
        - Flight distance: The flight distance of this journey
        - Departure Delay in Minutes: Minutes delayed when departure
        - Arrival Delay in Minutes: Minutes delayed when Arrival

    - **Target**
    
        - Satisfaction: Airline satisfaction level(Satisfaction, neutral or dissatisfaction)


#### Goal : Predict whether the customer is satisfied or not with airline services? 

- It is very important to understand what customer likes and don't like of a service you are selling. In the airline industry as there is a huge competition it is better to get the feedback from the customer and understand what they like and don't like. As we imporve the services that people are not satisfied with, we can keep on improving the loyal customer base. 

- Using this dataset, we will be understanding with which services customers are not happy with and what are the factors that are mostly influencing the customer satisfaction and will be predicting whether the customer is satisfied or not given all the above attributes information.

- Source : <a href='https://www.kaggle.com/teejmahal20/airline-passenger-satisfaction' target='_blank'> Airline Passenger Satisfaction </a>


### Exploratory Data Analysis
 
- In this project, I have done following:
    - Checked if there are any null values and replaced the null values with mean of that feature.
    - Checked if the data is sampled correctly or not and whether this problem is balanced or not.
    - How independent variables is affecting dependent variable with visualisations.
    - Also calculated mutual information score of all categorical and ordinal variables with target variable to see which feature provides more information about the target variable (satisfaction). 
    - Now, calculated the correlation scores of all numerical features with target variable to check which feature is highly correlated.
    - Removed the independent variables that are highly correlated with other independent variable.
- You can find the notebook.ipynb [here](notebook.ipynb)
### Model Training 

- I have trained following machine learning models for this problem and tuned them using auc metric on validation data.

    - `Logistic Regression`
    - `Decision Tree`
    - `Random Forest`
    - `XGBoost`
    
- Also checked the feature importances of each trained model except logistic regression and compared them with EDA.

### Final Model 

- I  chose the final model based on results on test dataset. I have used auc metric to choose the model.
- Final model is a XGBoost classifier which resulted 0.99 auc on test dataset.
- But as I couldn't install `xgboost` using pipenv on aws, I have selected random forest model as it has similar result to xgboost model.

### Deployment

#### Creating a virtual environment:
   - It's very important that we create a virtual environment before we deploy because we might face issues due to different versions of one or many packages. Different verisons issue can be eliminated by maintaining the version of all packages that we use for deployment.
    - To create a virtual environment and maintain a database of all packages with versions, let's use pipenv.
    - Run the following commands in the project folder.
        - To Install pipenv
            - `pip install pipenv`
        - To Install all required packages using pipenv for this project
            - `pipenv install requests scikit-learn==1.0.0 flask numpy`
        - To activate the virtual environment
            - `pipenv shell`
    - Now, you will see `Pipfile` and `Pipfile.lock`. In the `Pipfile.lock` you will see all the packages that are installed and the version information.
    
#### Deploying using flask, gunicorn
[All the required files are here](flask/)
- Step 1:
    - First let's create [`train.py`](train.py) from notebook.ipynb to train the final model as it would be easy for anyone to understand what we have deployed instead of going through the whole notebook.
    - `train.py` contains the code to train the DictVectorizer, final model and save them to `dv.bin` and `model.bin` using pickle. You can find the code here.
    - Now, run `train.py` in your project folder using `python train.py`.
    - Once the code is executed you will see `model.bin` and `dv.bin` in your project folder.
- Step 2:
    - In the project folder, now we have to create to a new file `predict.py` to load `model.bin` and `dv.bin` and predict the satisfaction rate for the customer when the request is sent to the flask app.
    - Once you have `predict.py`, make sure to activate virtual environment using pipenv.
    - Run `python predict.py` and check whether the flask application is up and running on given port number. (here it is 8000).
    - If it is running, test it using `test_predict.py` which consists of a customer information and prints the customer satisfaction rate.
    - If the there are no bugs in the code, we will get the customer satisfaction rate when we run `test_predict.py`
    - You can change the customer inputs in `test_predict.py`
        ```python
            Customer 1={'gender': 'male',
                 'customer_type': 'loyal_customer',
                 'age': 16,
                 'type_of_travel': 'business_travel',
                 'flight_distance': 311,
                 'inflight_wifi_service': 3,
                 'departure/arrival_time_convenient': 3,
                 'ease_of_online_booking': 3,
                 'gate_location': 3,
                 'food_and_drink': 5,
                 'online_boarding': 5,
                 'seat_comfort': 3,
                 'inflight_entertainment': 5,
                 'on-board_service': 4,
                 'leg_room_service': 3,
                 'baggage_handling': 1,
                 'checkin_service': 1,
                 'inflight_service': 2,
                 'cleanliness': 5,
                 'arrival_delay_in_minutes': 0.0}```
        ```
        ```python
         Customer2={'gender': 'female',
             'customer_type': 'loyal_customer',
             'age': 33,
             'type_of_travel': 'business_travel',
             'flight_distance': 325,
             'inflight_wifi_service': 2,
             'departure/arrival_time_convenient': 5,
             'ease_of_online_booking': 5,
             'gate_location': 5,
             'food_and_drink': 1,
             'online_boarding': 3,
             'seat_comfort': 4,
             'inflight_entertainment': 2,
             'on-board_service': 2,
             'leg_room_service': 2,
             'baggage_handling': 2,
             'checkin_service': 3,
             'inflight_service': 2,
             'cleanliness': 4,
             'arrival_delay_in_minutes': 7.0}
         ```
     - Demo:
    
    <a href="http://www.youtube.com/watch?feature=player_embedded&v=wfdZSUP2Zx4" target="_blank"><img src="https://i9.ytimg.com/vi/wfdZSUP2Zx4/mq1.jpg?sqp=CPDmiIwG&rs=AOn4CLBC9TJ2Jyr_N1hoZwYU8dgMFG8bGg" alt="Flask Demo" width="320" height="180" border="10" /></a>

- Step 3:
    - We will use gunicorn for as flask is not a developement server. First stop the flask application that we have run before. 
    - Install gunicorn by running following command in your terminal
        - `pipenv install gunicorn`
    - Now run `gunicorn --bind 0.0.0.0:8000 predict:app` in the terminal. We will see that the server is up.
- Step 4:
    - Check by running `test_predict.py`.
    
    - Demo:
     
     <a href="http://www.youtube.com/watch?feature=player_embedded&v=OdOuHgkmOgI" target="_blank"><img src="https://i9.ytimg.com/vi/OdOuHgkmOgI/mq2.jpg?sqp=CJzpiIwG&rs=AOn4CLA-SuHklg8vMmnMQ_ML-7k-S65yew" alt="Gunicorn Demo" width="320" height="180" border="10" /></a>

- Congratulations, you have successfully deployed a flask application using gunicorn.

#### Deploying using docker
[ Required files ](docker/)
- Follow all the above steps and stop the development server that we have just created using gunicorn.
- Make sure that you have installed docker on your system. If not, follow the steps in <a href='https://docs.docker.com/get-docker/' target="_blank">Docker Download</a> to download docker on your system.
- Now add the current user to docker sudo group to run docker commands without using sudo by following this <a href="https://docs.docker.com/engine/install/linux-postinstall/">documentation.</a>
- Once you have installed docker, create a docker file with below content in project folder
    ```docker
    FROM python:3.8.12-slim
    RUN pip install pipenv
    WORKDIR /app
    COPY ["Pipfile", "Pipfile.lock", "./"]
    RUN pipenv install --system --deploy
    COPY ["*.py","*.bin", "./"]
    EXPOSE 8000
    ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:8000", "predict:app"]
    ```
- Now, we have to make sure that predict.py, Pipfile, Pipfile.lock, model.bin and dv.bin are in the current folder where we are going to build our docker conatiner as we are copying those files to the container.

- Now lets build the docker container using below command
    - `docker build -t airline_satisfaction .`
- Once the docker container is built, we will run it using following command
    - ` docker run -it -p 8000:8000 airline_satisfaction:latest`
- Finally we have successfully deployed a prediction application inside a docker container.
- Demo:

<a href="http://www.youtube.com/watch?feature=player_embedded&v=maZS-kxTkv0" target="_blank"><img src="https://i9.ytimg.com/vi/maZS-kxTkv0/mq2.jpg?sqp=CMjriIwG&rs=AOn4CLANV7HbmsYValFaUY2mmdr1xGinsw" alt="Docker Demo" width="320" height="180" border="10" /></a>

#### Deploying on AWS using AWS BeanStalk
[Required files](aws/)
- Create an aws account and launch an ubuntu EC2 instance. 
- Check whether python is present in the instance, if not install python using following commmands:
    - `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
    - `bash Miniconda3-latest-Linux-x86_64.sh`
    - `source .bashrc`
- Now, install docker on ec2 instance and add the current user to docker sudo group to run docker commands without using sudo.
- Now we have to install pipenv using `pip install pipenv`.
- Upload the project folder to ec2 instance.
- Go to the project folder and run following commands to install aws beanstalk:
    - `pipenv install awsebcli`
- Activate the virtual environment using `pipenv shell`.
- Check whether we have installed aws beanstalk or not by running `eb` command, if it executes without any issues then we can move forward to launch the docker applications.
- Create the aws beanstalk application using following command:
    - `eb init -p docker -r us-east-2 satisfaction-airline`
- Run the aws beanstalk application locally on ec2 instance that we have created above using following command:
    - `eb local run --port 8000`
- Now we can test the aws beanstalk application by running `python test_predict.py` on ec2 instance. If we get the response back then we have successfully created an aws beanstalk application.
- To make it public we have to create aws elastic beanstalk environment by running `eb create satisfaction-env`. After the command successfully run we will see the public address at end.
    - Ex: satisfaction-env.eba-rexdiqp2.us-east-2.elasticbeanstalk.com
- We can send the requests to above address from any machine in the world. To check it we need to make following changes to `test_predict.py`.
    ```python
    host = "satisfaction-env.eba-rexdiqp2.us-east-2.elasticbeanstalk.com"
    url = f'http://{host}/predict'
    ```
- Now we can run `test_predict.py` from the local machine, if we get the response then everything is working as expected.
- To terminate aws beanstalk application we can delete from aws beanstalk console or using following command in ec2 instance.
    - `eb terminate satisfaction-env`
- Congratulations, we have succesfully installed an aws beanstalk application which runs a docker container.
- Demo:

<a href="http://www.youtube.com/watch?feature=player_embedded&v=m0gWB3Begag" target="_blank"><img src="https://i9.ytimg.com/vi/m0gWB3Begag/mq2.jpg?sqp=CPj0iIwG&rs=AOn4CLCHWBmCVYmrvriX2ZoFxZ0y7qPqUQ" alt="AWS Demo" width="320" height="180" border="10" /></a>

