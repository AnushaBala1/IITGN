import json
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
import re
import io
import sys
import os

# Placeholder for JSON data
# Replace with your actual JSON data
json_data = '''
[
    {"item": "foodgrains", "no. of sample households": "1,566", "consumer expenditure in three months (90 days) in Rupees per household": "Rs. 167.28", "per person": "Rs. 30.99", "percentage of total": "40.28"},
    {"item": "pulses", "no. of sample households": "1,564", "consumer expenditure in three months (90 days) in Rupees per household": "24.43", "per person": "4.50", "percentage of total": "5.85"},
    {"item": "edible oil", "no. of sample households": "1,566", "consumer expenditure in three months (90 days) in Rupees per household": "15.17", "per person": "2.83", "percentage of total": "3.68"},
    {"item": "vegetables", "no. of sample households": "1,566", "consumer expenditure in three months (90 days) in Rupees per household": "8.48", "per person": "1.54", "percentage of total": "2.00"},
    {"item": "milk & milk products", "no. of sample households": "1,566", "consumer expenditure in three months (90 days) in Rupees per household": "25.58", "per person": "4.76", "percentage of total": "6.19"},
    {"item": "meat, egg and fish", "no. of sample households": "1,566", "consumer expenditure in three months (90 days) in Rupees per household": "4.12", "per person": "0.77", "percentage of total": "1.00"},
    {"item": "fruits", "no. of sample households": "1,566", "consumer expenditure in three months (90 days) in Rupees per household": "6.30", "per person": "1.16", "percentage of total": "1.51"},
    {"item": "refreshments", "no. of sample households": "1,566", "consumer expenditure in three months (90 days) in Rupees per household": "1.16", "per person": "0.26", "percentage of total": "0.34"},
    {"item": "salt", "no. of sample households": "1,555", "consumer expenditure in three months (90 days) in Rupees per household": "1.67", "per person": "0.26", "percentage of total": "0.34"},
    {"item": "spices", "no. of sample households": "1,566", "consumer expenditure in three months (90 days) in Rupees per household": "7.33", "per person": "1.28", "percentage of total": "1.66"},
    {"item": "sugar", "no. of sample households": "1,566", "consumer expenditure in three months (90 days) in Rupees per household": "18.38", "per person": "3.34", "percentage of total": "4.34"},
    {"item": "pan (betel leaves)", "no. of sample households": "1,566", "consumer expenditure in three months (90 days) in Rupees per household": "1.67", "per person": "0.26", "percentage of total": "0.34"},
    {"item": "tobacco", "no. of sample households": "1,566", "consumer expenditure in three months (90 days) in Rupees per household": "7.84", "per person": "1.42", "percentage of total": "1.85"},
    {"item": "intoxicants", "no. of sample households": "1,566", "consumer expenditure in three months (90 days) in Rupees per household": "1.67", "per person": "0.26", "percentage of total": "0.34"},
    {"item": "fuel and light", "no. of sample households": "1,566", "consumer expenditure in three months (90 days) in Rupees per household": "28.28", "per person": "5.27", "percentage of total": "6.85"},
    {"item": "clothing : cotton", "no. of sample households": "1,566", "consumer expenditure in three months (90 days) in Rupees per household": "18.95", "per person": "3.50", "percentage of total": "4.55"},
    {"item": ": silk", "no. of sample households": "1,566", "consumer expenditure in three months (90 days) in Rupees per household": "0.44", "per person": "0.08", "percentage of total": "0.10"},
    {"item": ": wool", "no. of sample households": "1,566", "consumer expenditure in three months (90 days) in Rupees per household": "1.11", "per person": "0.20", "percentage of total": "0.26"},
    {"item": "bedding", "no. of sample households": "1,566", "consumer expenditure in three months (90 days) in Rupees per household": "2.57", "per person": "0.47", "percentage of total": "0.61"},
    {"item": "amusements", "no. of sample households": "1,566", "consumer expenditure in three months (90 days) in Rupees per household": "2.16", "per person": "0.39", "percentage of total": "0.51"},
    {"item": "education", "no. of sample households": "1,566", "consumer expenditure in three months (90 days) in Rupees per household": "1.47", "per person": "0.27", "percentage of total": "0.35"},
    {"item": "medicine", "no. of sample households": "1,566", "consumer expenditure in three months (90 days) in Rupees per household": "5.07", "per person": "0.93", "percentage of total": "1.21"},
    {"item": "toilet", "no. of sample households": "1,566", "consumer expenditure in three months (90 days) in Rupees per household": "0.90", "per person": "0.15", "percentage of total": "0.20"},
    {"item": "petty articles", "no. of sample households": "1,566", "consumer expenditure in three months (90 days) in Rupees per household": "1.53", "per person": "0.27", "percentage of total": "0.35"},
    {"item": "conveyance", "no. of sample households": "1,566", "consumer expenditure in three months (90 days) in Rupees per household": "4.32", "per person": "0.81", "percentage of total": "1.05"},
    {"item": "services", "no. of sample households": "1,566", "consumer expenditure in three months (90 days) in Rupees per household": "20.55", "per person": "3.81", "percentage of total": "4.95"},
    {"item": "furniture", "no. of sample households": "1,566", "consumer expenditure in three months (90 days) in Rupees per household": "0.78", "per person": "0.14", "percentage of total": "0.18"},
    {"item": "sundry equipments", "no. of sample households": "1,566", "consumer expenditure in three months (90 days) in Rupees per household": "0.74", "per person": "0.14", "percentage of total": "0.18"},
    {"item": "musical instruments", "no. of sample households": "1,566", "consumer expenditure in three months (90 days) in Rupees per household": "0.11", "per person": "0.02", "percentage of total": "0.03"},
    {"item": "ornaments", "no. of sample households": "1,566", "consumer expenditure in three months (90 days) in Rupees per household": "5.79", "per person": "1.07", "percentage of total": "1.39"},
    {"item": "foot-wear", "no. of sample households": "1,566", "consumer expenditure in three months (90 days) in Rupees per household": "2.32", "per person": "0.43", "percentage of total": "0.56"},
    {"item": "utensils", "no. of sample households": "1,566", "consumer expenditure in three months (90 days) in Rupees per household": "1.03", "per person": "0.19", "percentage of total": "0.25"},
    {"item": "ceremonials", "no. of sample households": "1,566", "consumer expenditure in three months (90 days) in Rupees per household": "22.33", "per person": "4.12", "percentage of total": "5.36"},
    {"item": "rent", "no. of sample households": "1,566", "consumer expenditure in three months (90 days) in Rupees per household": "5.21", "per person": "0.96", "percentage of total": "1.25"},
    {"item": "taxes", "no. of sample households": "1,566", "consumer expenditure in three months (90 days) in Rupees per household": "0.38", "per person": "0.07", "percentage of total": "0.09"}
]
'''

# Load JSON data into a Pandas DataFrame
data = json.loads(json_data)
df = pd.DataFrame(data)

# Clean the DataFrame columns
def clean_numeric(value):
    if isinstance(value, str):
        return float(value.replace('Rs. ', '').replace(',', ''))
    return float(value)

# Apply cleaning to relevant columns
for col in ['consumer expenditure in three months (90 days) in Rupees per household', 'per person', 'no. of sample households', 'percentage of total']:
    if col in df.columns:
        df[col] = df[col].apply(clean_numeric)

# Save the cleaned DataFrame to a temporary pickle file
df.to_pickle("temp_df.pkl")

# Initialize the GPT-3.5 Turbo model
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    openai_api_key="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"  # Replace with your actual OpenAI API key
)

# Define the updated prompt template
prompt = PromptTemplate(
    input_variables=["data_description"],
    template="""
    You are an AI assistant tasked with analyzing a dataset and generating visualizations.

    The dataset is already loaded into a Pandas DataFrame called `df`. Do NOT attempt to load any external files except reading `df` from a pickle file 'temp_df.pkl' using `pd.read_pickle('temp_df.pkl')`. Here is some information about the data:

    {data_description}

    Your task is to:

    1. Analyze all data points in `df` to find exactly five mind-blowing insights.
    2. Create four separate Python scripts, each generating one plot that visualizes a useful insight.

    Return a single JSON object with the following structure:
    - "insights": A list of five strings, each in the format "Insight X: [description]" (where X is 1 to 5), representing surprising patterns, correlations, trends, or anomalies from the full dataset.
    - "plot1": A string containing the Python code for the first plot script (equivalent to plot1.py).
    - "plot2": A string containing the Python code for the second plot script (equivalent to plot2.py).
    - "plot3": A string containing the Python code for the third plot script (equivalent to plot3.py).
    - "plot4": A string containing the Python code for the fourth plot script (equivalent to plot4.py).

    Each plot script should:
    - Import necessary libraries (e.g., pandas as pd, matplotlib.pyplot as plt)
    - Load the DataFrame using `df = pd.read_pickle('temp_df.pkl')`
    - Create one plot tied to an insight, save it as 'plotX.png' (where X is 1 to 4), and display it using plt.show()
    - Include a descriptive title or label explaining the insight
    - Assume columns are pre-cleaned (numeric values are floats without 'Rs.' or commas)

    Ensure the JSON is valid, the insights are derived from analyzing the entire dataset, and the plots provide useful visualizations. Return the JSON within ```json and ``` markers.
    """
)

# Generate a description of the data
old_stdout = sys.stdout
sys.stdout = buffer = io.StringIO()
df.info()
sys.stdout = old_stdout
info_output = buffer.getvalue()

data_description = "First few rows:\n" + str(df.head()) + "\n\nData info:\n" + info_output

# Create a RunnableSequence
chain = RunnableSequence(prompt | llm)

# Run the agent to generate the JSON response
response = chain.invoke({"data_description": data_description})

# Extract the JSON from the response
json_match = re.search(r"```json\n(.*)\n```", response.content, re.DOTALL)

if json_match:
    json_str = json_match.group(1)
    try:
        result = json.loads(json_str)
        
        # Print the five insights
        print("Five Insights:")
        for insight in result["insights"]:
            print(insight)
        print("\n")

        # Process and execute the four plot scripts
        for i in range(1, 5):
            plot_key = f"plot{i}"
            filename = f"plot{i}.py"
            code = result[plot_key]
            
            # Write the script to a file
            with open(filename, 'w') as f:
                f.write(code)
            
            print(f"Generated {filename}:")
            print(code)
            print("\nExecuting the code...\n")
            
            # Execute the script
            try:
                with open(filename) as f:
                    exec(f.read())
                print(f"\n{filename} executed successfully. Check the plot file (plot{i}.png) and the displayed plot.")
            except Exception as e:
                print(f"An error occurred while executing {filename}: {e}")
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}\nResponse:\n{response.content}")
else:
    print("No JSON block found in the response.\nResponse:\n{response.content}")

# Clean up temporary pickle file
os.remove("temp_df.pkl")