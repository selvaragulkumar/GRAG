from langchain.graphs import Neo4jGraph
import requests
import json
import openai
from datetime import date
import streamlit as st

NEO4J_URI = '<fill it>'
NEO4J_USERNAME = '<fill it>'
NEO4J_PASSWORD = '<fill it>'
NEO4J_DATABASE = '<fill it>'

kg = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE
)

l=0

def get_today_date():
    today = date.today()
    return today.strftime("%Y %m %d")

today_date = get_today_date()
#print(today_date)

w_key="<fill it>"
openai_key = '<fill it>'

base_url="https://api.weatherapi.com/v1"

#load api spec

file_path = '/<your_path>/W_api_spec.json'

with open(file_path, 'r') as file:
    api_spec = json.load(file)
    
#Streamlit app input 

st.title("Weather API bot")

# Initialize session state variables
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from the session
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#Get the prompt from the chat box
if prompt := st.chat_input("Ask me anything!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

#recieving prompt from streamlit app
t = query = prompt

#prompt for simplifying query to list of queries
prompt='''
Today's date: {date}
User's Query: {query}
Default location is London, if location not provided in query

Convert the Complex users query to a list of very simple basic data retrieval queries that has all the basic information to get the relevant information and asnwer. 
The queries should only be to get the required data alone to asnwer the query

use the word history if the date requested is past to todays date
use word future if the requested date is more than 14 days in the future to todays date
use the word current if requested for todays date
use the word forecast if it is for a date between todays date and 14 days from today

example:
    
    query: Compare yesterday weather to today weather
    
    result: ['get me history weather on 2024-01-01 in London', 'get forecast weather on 2024-07-25 in London']
    
Make sure the simple queries have the subject of comaprison mentioned in all of them, like if you need to comapare for hot or cold then mention the word, in all simple query.

return it as a json file. remember double qoutes
The json you return will be parsed hence dont give any comments or quotes, just dictionary!!
'''

#Calling OpenAI gpt

def create_prompt_1():
    return prompt.format(date=today_date, query=query)

def call_openai_api_1(prompt=create_prompt_1()):
    openai.api_key = '<Fill it up>'
    response = openai.chat.completions.create(
      model="gpt-4o",
      temperature=0,
      messages=[
            {"role": "system", "content": "You are a helpful agent that converts a complex query into a list of simple queries to get relevant data."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

#List of simple queries
queries=[]
if query:
    result=call_openai_api_1()
    result=json.loads(result)
    queries=result.get('queries')
    print(queries)
    
#GPT prompt to get path from the open_api_spec

prompt_template='''
Today's date: {date}
User's Query: {query}
paths:
    {info}

Based on the user query comapre it with description of paths and return what path to choose 

Give values for all the parameters that's required under that path extracted from the user's query
Make sure all the required parameters have a value


Output format: (dictionary)
    
     path: [key:value]
     "parameters": [dictionary]
         <prameter1>: [key:value]
         <parameter2>: [key:value]
         
    
return it as a json file. remember double qoutes
The json you return will be parsed hence dont give any comments or quotes, just dictionary!!
'''

#Get the main path and parameter info to do api calls from the spec

def get_api_info(data):
    result = []
    paths = data.get('paths', {})
    
    for path, methods in paths.items():
        for method, details in methods.items():
            summary = details.get('summary', 'No summary provided')
            description = details.get('description', 'No description provided')
            parameters = details.get('parameters', [])
            
            result.append(f"Path: {path}")
            result.append(f"  Summary: {summary}")
            result.append(f"  Description: {description}")
            result.append("  Parameters:")
            
            if not parameters:
                result.append("    None")
            else:
                for param in parameters:
                    param_name = param.get('name', 'No name provided')
                    param_description = param.get('description', 'No description provided')
                    param_required = param.get('required', False)
                    result.append(f"    - Name: {param_name}")
                    result.append(f"      Description: {param_description}")
                    result.append(f"      Required: {param_required}")
                    
            result.append("\n" + "-"*50 + "\n")
    
    return "\n".join(result)

#Get the api data after cleansing it

def get_data(query):
    def create_prompt(query):
        info=get_api_info(api_spec)
        return prompt_template.format(date=today_date, query=query, info=info)
    
    def call_openai_api(query):
        prompt=create_prompt(query) 
        openai.api_key = '<fill it>'
        response = openai.chat.completions.create(
          model="gpt-4o",
          temperature=0,
          messages=[
                {"role": "system", "content": "You are a helpful agent that tells how to do api calls based on user queries."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    
    def get_data(path, params):
        endpoint = f"{base_url}{path}"
        print(params)
        response = requests.get(endpoint, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()
    
    def remove_epoch_entries(data):
        if isinstance(data, dict):
            return {key: remove_epoch_entries(value) for key, value in data.items() if ("epoch" not in key.lower()) and ("icon" not in key.lower()) and ("code" not in key.lower())}
        elif isinstance(data, list):
            return [remove_epoch_entries(item) for item in data]
        else:
            return data
    
    result=call_openai_api(query)
    print(result)
    
    result=json.loads(result)
    
    path= result.get("path")
    params={}
    params["key"]=w_key
    
    temp=result.get("parameters")
    
    for key in temp:
        params[key]=temp[key]
    
    w_data = get_data(path, params)
    
    w_data=remove_epoch_entries(w_data)
    
    return w_data

#Generate cypher query for initializing the graph with the api data

def generate_cypher_query(data):
    def sanitize(value):
        return str(value).replace('"', '\\"')

    query = ""
    node_counters = {}
    graph_schema = {"nodes": {}, "relationships": []}

    def create_node(node_type, node_data, parent_type=None, parent_id=None):
        nonlocal query
        nonlocal node_counters
        nonlocal graph_schema

        if not isinstance(node_data, dict):
            return None

        # Initialize or increment the counter for the current node type
        if node_type not in node_counters:
            node_counters[node_type] = 1
            graph_schema["nodes"][node_type] = {
                "count": 0,
                "properties": set(),
                "ids": []
            }
        else:
            node_counters[node_type] += 1

        node_instance_id = f"{node_type}_{node_counters[node_type]}"
        node_id = None
        properties = {}
        child_nodes = []

        # Collect the properties and child nodes
        for key, value in node_data.items():
            if isinstance(value, dict):
                child_nodes.append((key, value))
            elif isinstance(value, list):
                child_nodes.append((key, value))
            else:
                if node_id is None:
                    node_id = sanitize(value)
                properties[key] = sanitize(value)
                graph_schema["nodes"][node_type]["properties"].add(key)

        if node_id is None:
            node_id = node_instance_id

        graph_schema["nodes"][node_type]["count"] += 1
        graph_schema["nodes"][node_type]["ids"].append(node_id)
        
        # Create the node with its properties
        node_query = f'MERGE ({node_instance_id}:{node_type} {{id: "{node_id}"}})\n'
        for key, value in properties.items():
            node_query += f'ON CREATE SET {node_instance_id}.{key} = "{value}"\n'

        query += node_query

        # Establish relationship with parent if exists
        if parent_type and parent_id:
            relationship = {
                "start_node": parent_type,
                "end_node": node_type,
                "type": f'HAS_{node_type.upper()}'
            }
            graph_schema["relationships"].append(relationship)
            relationship_query = f'''
            WITH 1 as l
            MATCH (p:{parent_type} {{id: "{parent_id}"}})
            MATCH (c:{node_type} {{id: "{node_id}"}})
            MERGE (p)-[:HAS_{node_type.upper()}]->(c)\n
            '''
            query += relationship_query

        # Create child nodes and establish relationships
        for child_key, child_value in child_nodes:
            if isinstance(child_value, dict):
                child_node_id = create_node(child_key, child_value, node_type, node_id)
                relationship = {
                    "start_node": node_type,
                    "end_node": child_key,
                    "type": f'HAS_{child_key.upper()}'
                }
                graph_schema["relationships"].append(relationship)
                relationship_query = f'''
                WITH 1 as l
                MATCH (a:{node_type} {{id: "{node_id}"}})
                MATCH (b:{child_key} {{id: "{child_node_id}"}})
                MERGE (a)-[:HAS_{child_key.upper()}]->(b)\n
                '''
                query += relationship_query
            elif isinstance(child_value, list):
                for item in child_value:
                    if isinstance(item, dict):
                        child_node_id = create_node(child_key, item, node_type, node_id)
                        relationship = {
                            "start_node": node_type,
                            "end_node": child_key,
                            "type": f'HAS_{child_key.upper()}'
                        }
                        graph_schema["relationships"].append(relationship)
                        relationship_query = f'''
                        WITH 1 as l
                        MATCH (a:{node_type} {{id: "{node_id}"}})
                        MATCH (b:{child_key} {{id: "{child_node_id}"}})
                        MERGE (a)-[:HAS_{child_key.upper()}]->(b)\n
                        '''
                        query += relationship_query

        return node_id

    # Process the root-level dictionaries and establish relationships between them
    previous_root_id = None
    prev_name = None
    for key, value in data.items():
        root_id = create_node(key, value)
        if previous_root_id:
            relationship = {
                "start_node": prev_name,
                "end_node": key,
                "type": f'HAS_{key.upper()}'
            }
            graph_schema["relationships"].append(relationship)
            relationship_query = f'''
            WITH 1 as l
            MATCH (a {{id: "{previous_root_id}"}})
            MATCH (b {{id: "{root_id}"}})
            MERGE (a)-[:HAS_{key.upper()}]->(b)\n
            '''
            query += relationship_query
        previous_root_id = root_id
        prev_name = key

    # Convert sets of properties to lists for JSON serialization
    for node in graph_schema["nodes"].values():
        node["properties"] = list(node["properties"])

    return query, graph_schema

#Cleanse the graph schema generated by initializing the graph

def remove_duplicate_relationships(graph_schema):
    relationships = graph_schema["relationships"]
    seen_relationships = set()
    unique_relationships = []

    for rel in relationships:
        rel_tuple = (rel["start_node"], rel["end_node"], rel["type"])
        if rel_tuple not in seen_relationships:
            seen_relationships.add(rel_tuple)
            unique_relationships.append(rel)

    graph_schema["relationships"] = unique_relationships
    return graph_schema

#Function to update the exisitng graph schema with new graph schema

def update_graph_schema(existing_schema, new_schema):
    # Update nodes
    for node_type, node_info in new_schema["nodes"].items():
        if node_type not in existing_schema["nodes"]:
            existing_schema["nodes"][node_type] = {"count": 0, "properties": set(), "ids": set()}
        
        existing_schema["nodes"][node_type]["count"] += node_info["count"]
        
        # Convert properties to set for updating
        existing_properties = set(existing_schema["nodes"][node_type]["properties"])
        new_properties = set(node_info["properties"])
        existing_schema["nodes"][node_type]["properties"] = existing_properties.union(new_properties)
        
        # Convert ids to set for updating
        existing_ids = set(existing_schema["nodes"][node_type]["ids"])
        new_ids = set(node_info["ids"])
        existing_schema["nodes"][node_type]["ids"] = existing_ids.union(new_ids)

    # Update relationships
    existing_relationships = {(rel["start_node"], rel["end_node"], rel["type"]) for rel in existing_schema["relationships"]}
    for relationship in new_schema["relationships"]:
        rel_tuple = (relationship["start_node"], relationship["end_node"], relationship["type"])
        if rel_tuple not in existing_relationships:
            existing_schema["relationships"].append(relationship)
            existing_relationships.add(rel_tuple)

    # Convert sets of properties and ids back to lists for JSON serialization
    for node in existing_schema["nodes"].values():
        node["properties"] = list(node["properties"])
        node["ids"] = list(node["ids"])

    return existing_schema


b='''
match(n) detach delete n
'''

#Free the graph

kg.query(b)

if queries:  
    #for loop to get data for the list of queries and add the data to the graph
    for i in queries:
        
        data = get_data(i)
        x,y = generate_cypher_query(data)
        kg.query(x)
        y=remove_duplicate_relationships(y)
        w=None
        try:
            with open('<your_path>/graph_schema.json', 'r') as file:
                w = json.load(file)
        except:
            print("first schema")

        if w:    
            y=update_graph_schema(w, y)

        with open('<your_path>/graph_schema.json', 'w') as json_file:
            json.dump(y, json_file, indent=4)
    
    print("Data has been imported into Neo4j successfully.")   
  
#prompt to search the required data from the graph schema
prompt_template='''

User Query: {query}

Todays date: {date}

Based on the above query, give address to the required info from the below graph 

Schema: {info}

for today's data you can either get current or today's date.
If simple weather is asked return the main attributes that can define weather properly
if want to go to specific node id, then mention node id, else dont include id. end_node_id for id in end node, and start_node_id for id of the start node. 
example:
    
      query: [ <list of dictionary
        
          start_node: <name>,
          start_node_id: <id >,
          relationship: <relationship>,
          end_node: < name>,
          "properties": []
        
        
          start_node: < > ,
          relationship:< > ,
          end_node: < >,
          end_node_id: 
          "properties": [<list of required properties from the schema>]
        
        ]

return it as a json file. remember double qoutes
The json you return will be parsed hence dont give any comments or quotes, just dictionary!!
'''

def create_prompt_1(query):
    return prompt_template.format(query=query, info=y, date=today_date)

def call_openai_api_1(query):
    prompt=create_prompt_1(query)
    openai.api_key = '<fill it>'
    response = openai.chat.completions.create(
      model="gpt-4o",
      temperature=0.4,
      messages=[
            {"role": "system", "content": "You are a helpful agent that tells how to query a graph database based on the graph schema, return just the query json."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

#Function to generate cypher query for searching the graph

def generate_cypher_query(query_schema):
    query = ""
    match_statements = []
    return_statements = []

    previous_node_alias = None

    for i, element in enumerate(query_schema):
        start_node = element.get("start_node")
        start_node_id = element.get("start_node_id")
        relationship = element.get("relationship")
        end_node = element.get("end_node")
        end_node_id = element.get("end_node_id")
        properties = element.get("properties", [])

        # Determine the current node aliases
        start_node_alias = f"a{i}" if previous_node_alias is None else previous_node_alias
        end_node_alias = f"b{i}"

        # Create MATCH statement for different cases
        if start_node_id and end_node_id:
            match_statement = f'MATCH ({start_node_alias}:{start_node} {{id: "{start_node_id}"}})-[:{relationship}]->({end_node_alias}:{end_node} {{id: "{end_node_id}"}})'
        elif start_node_id and not end_node_id:
            match_statement = f'MATCH ({start_node_alias}:{start_node} {{id: "{start_node_id}"}})-[:{relationship}]->({end_node_alias}:{end_node})'
        elif not start_node_id and end_node_id:
            match_statement = f'MATCH ({start_node_alias}:{start_node})-[:{relationship}]->({end_node_alias}:{end_node} {{id: "{end_node_id}"}})'
        else:
            match_statement = f'MATCH ({start_node_alias}:{start_node})-[:{relationship}]->({end_node_alias}:{end_node})'

        match_statements.append(match_statement)

        # Update previous node alias for the next iteration
        previous_node_alias = end_node_alias

        # Create RETURN statement for properties
        if properties:
            props = ", ".join([f'{end_node_alias}.{prop}' for prop in properties])
            return_statements.append(props)

    # Combine all MATCH statements
    query += "\n".join(match_statements)

    # Combine all RETURN statements
    if return_statements:
        query += "\nRETURN " + ", ".join(return_statements)

    return query


def call_openai_api(prompt):
    
    openai.api_key = '<fill it>'
    response = openai.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a helpful bot that answers the query based on the retrieved data from the weather API."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

#Final prompt to answer the user Query

prompt='''
You are supposed to answer this query based on the data retrieved from the knowledge graph.
User query: {query}

Data retrieved:
    {data}
'''
data=""

#for loop to retrieve data for each simple query

if queries:
    for i in queries:
        result=call_openai_api_1(i)
        result=json.loads(result)
        x=generate_cypher_query(result)
        print(x)
        x = kg.query(x)
        #retrieved data
        data= data+ i + ": Data: " + str(x) + "\n"
        
    print(data)
    
    #answer the final prompt with retrieved data
    
    with st.chat_message("assistant"):
        # Get response from the OpenAI API
        response_text = call_openai_api(prompt.format(query=t, data=data))
         
        # Display the assistant's response
        st.markdown(response_text)

    # Append the assistant's response to the session state
    st.session_state.messages.append({"role": "assistant", "content": response_text})
        
        








