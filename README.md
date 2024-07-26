# Weather API Bot

This project implements a weather bot using Streamlit to provide a user-friendly interface for interacting with weather data APIs. It leverages OpenAI's GPT model to parse complex user queries into simplified API calls, retrieves data from a weather API, and stores the data in a Neo4j graph database for further analysis and querying.

## Setup

To run the project, you need to have the following installed on your system:

- Python 3.8 or higher
- Streamlit
- Neo4j
- Required Python libraries (listed in `requirements.txt`)

## Installation

### Clone the repository:

```bash
git clone https://github.com/your-repo/weather-bot.git
cd weather-bot
```
## Install Dependencies

To install the necessary dependencies, run the following command:

```bash
pip install langchain streamlit openai
```

## Start Neo4j

Ensure that Neo4j is running on your machine. You can download and install Neo4j from here.

Set Environment Variables
Export your OpenAI API key and weather API key:

```bash
export OPENAI_API_KEY='your-openai-api-key'
export WEATHER_API_KEY='your-weather-api-key'
```

## Run the Streamlit App

To start the Streamlit app, use the following command:

```bash
streamlit run weather_data.py
```

## Project Structure
**weather_data.py:**  The main script containing the Streamlit app and logic for handling user queries.
**W_api_spec.json:** The OpenAPI specification file used to understand available API endpoints and their descriptions.

**Main Functions**

- **get_today_date()**

  Description: Returns today's date in the format YYYY MM DD.

- **create_prompt_1()**

  Description: Generates a prompt for the OpenAI API to simplify complex user queries into basic data retrieval queries.

- **call_openai_api_1()**

  Description: Calls the OpenAI GPT API to convert a complex query into a list of simpler queries.

- **get_api_info()**

  Description: Extracts API endpoint information from the OpenAPI specification file.

- **get_data()**

  Description: Fetches weather data from the weather API based on simplified queries and cleans the data by removing unwanted entries.

- **generate_cypher_query()**
  
  Description: Generates a Cypher query to initialize the Neo4j graph with weather data.

- **remove_duplicate_relationships()**

  Description: Cleans the graph schema by removing duplicate relationships.

- **update_graph_schema()**

  Description: Updates the existing graph schema with new data and relationships.

- **generate_cypher_query()**

  Description: Generates a Cypher query for searching the graph database based on user queries.

- **call_openai_api()**

  Description: Calls the OpenAI API to answer the user's query based on the data retrieved from the knowledge graph.

## Usage

Launch the Streamlit App

Run the app using the command:

```bash
streamlit run weather_data.py
```

## Interact with the Bot

Enter your query in the chat box, such as "Compare yesterday's weather to today's weather in London." The bot will:

- Convert this query into simple queries.
- Call the weather API and store the data in Neo4j.
- Use the stored data to provide a meaningful response.

**Graph Database**: 
The app initializes a Neo4j graph database, creates nodes and relationships based on the weather data, and performs queries to retrieve and analyze data.

**OpenAPI Specification**: 
The app uses the OpenAPI specification to identify the appropriate API paths and parameters for querying the weather API.

**API Keys**: 
Ensure that you have valid API keys for OpenAI and the weather API you are using. Replace the placeholders in the code with your actual API keys.

## Graph Example

Here is an example of the graph used in the application:

![image](https://github.com/user-attachments/assets/31e3c4a0-8671-40c6-931f-bcdc42d1ee10)


## Chatbot Example

Here is an image of the chatbot interface:

![image](https://github.com/user-attachments/assets/bf0c045d-5127-4939-8323-76d15ad8e8bb)



