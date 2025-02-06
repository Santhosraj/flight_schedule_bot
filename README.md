Overview

**Airmate✈️** is an AI-powered chatbot designed to assist users with flight availability queries. It utilizes a simple neural network for intent classification and integrates with a flight API to fetch real-time flight data. The chatbot is deployed with a Chainlit-based front-end for user interaction.

https://github.com/user-attachments/assets/d1796119-dcd0-4c72-82b5-7148f42564b1



**Project Workflow** 📈

**1. Data Preparation**

o The dataset consists of intents stored in a JSON format.

o Each intent includes sample user queries and corresponding responses.

o The data is preprocessed and transformed into numerical representations for training.

**2. Model Training**

o A simple feedforward neural network is trained using Keras Sequential API.

**The architecture includes:**

o Input layer with 128 neurons and ReLU activation.

o Dropout layer (50%) for regularization.

o Hidden layer with 64 neurons and ReLU activation.

o Dropout layer (50%) for further regularization.

o Output layer with softmax activation for intent classification.

o The model is compiled using the Stochastic Gradient Descent (SGD) optimizer with momentum and Nesterov acceleration.

o Training is performed with 100 epochs and a batch size of 5.

**3. Inference and API Integration**

The trained model classifies user inputs into predefined intents.

If a user query contains a keyword related to flight availability, the chatbot triggers the API.

The API fetches real-time flight data and returns relevant results.

The chatbot formats the response and displays it to the user.

**4. Deployment and Frontend**

The chatbot is deployed using Chainlit for an interactive front-end experience.

Chainlit enables smooth communication between users and the chatbot.

The interface is designed to handle user queries and display API responses seamlessly.

**5. Techstacks**🖥️

**Backend:** Python , TensorFlow/Keras

**Machine Learning:** Neural Networks (Sequential Model)

**Frontend:** Chainlit

**APIs:** Amadeus


**6. Future Enhancements**💫

**Improve Model Performance:**📈

o Expand dataset for better intent recognition.

o Integrate multi-agent architecture to enhance chatbot capabilities for trip planning, itinerary preparation, and flight booking.

o Experiment LLM models .

**Enhanced API Features:**

o Add more filters (e.g., airline preference, layovers, ticket price range).

o Integrate multiple flight APIs for better coverage.

**Voice Assistant Feature:**

o Use speech-to-text and text-to-speech for voice-based interactions.
