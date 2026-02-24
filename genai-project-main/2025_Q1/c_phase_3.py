import dspy
from dspy.teleprompt import BootstrapFewShot
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
import boto3
import re
import json
import faiss
import numpy as np
import io
from typing import List, Dict

# Configuration Dictionary
CONFIG = {
    "bucket_name": "your-s3-bucket-name",
    "csv_folder": "csv_metadata",
    "embedding_model": "amazon.titan-embed-text-v1",  # Bedrock model ID
    "index_folder": "faiss_index",
    "index_file": "faiss_index.faiss",
    "embedding_dimension": 1536,
    "nprobe": 16
}

# Initialize S3 client
s3 = boto3.client('s3')

# Initialize Bedrock client
bedrock = boto3.client('bedrock-runtime')

# Configuration Widgets
bucket_name_widget = widgets.Text(description="Bucket Name:", placeholder="your-s3-bucket")
csv_folder_widget = widgets.Text(description="CSV Folder:", placeholder="csv_metadata")
index_folder_widget = widgets.Text(description="FAISS Folder:", placeholder="faiss_index")
embedding_dimension_widget = widgets.IntText(description="Embedding Dim:", value=1536)
embedding_model_widget = widgets.Text(description="Embedding Model:", placeholder="amazon.titan-embed-text-v1")
nprobe_widget = widgets.IntText(description="nprobe:", value=16)

# Teleprompting Widgets
use_teleprompter_widget = widgets.Checkbox(description="Use Teleprompter?", value=False)
#use_rationale_model_widget = widgets.Checkbox(description="Rationale Model", value=False)
#use_qa_model_widget = widgets.Checkbox(description="Q&A Model?", value=False)
num_fewshot_widget = widgets.IntText(description="# Fewshot Examples", value=3)

# Display Configuration Widgets
display(bucket_name_widget, csv_folder_widget, index_folder_widget, embedding_dimension_widget, embedding_model_widget, nprobe_widget)

# Display Teleprompting Widgets
display(use_teleprompter_widget) #,use_rationale_model_widget, use_qa_model_widget, num_fewshot_widget)

# Helper functions
def load_data_from_s3(config: Dict, filename: str) -> pd.DataFrame:
    """Loads data from CSV file in S3 into a Pandas DataFrame."""
    s3_key = f"{config['csv_folder']}/{filename}"
    try:
        response = s3.get_object(Bucket=config["bucket_name"], Key=s3_key)
        csv_string = response['Body'].read().decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_string))  # Use StringIO to read from string
        print(f"Data loaded from s3://{config['bucket_name']}/{s3_key}")
        return df
    except Exception as e:
        print(f"Error loading data from s3://{config['bucket_name']}/{s3_key}: {e}")
        return None

def generate_embedding(text: str) -> List[float]:
    """Generate embedding for the given text using Bedrock."""
    response = bedrock.invoke_model(
        modelId=CONFIG["embedding_model"],
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": text})
    )
    embedding = json.loads(response['body'].read())['embedding']
    return embedding

def save_faiss_index_s3(config: Dict, index: faiss.Index, filename: str = "faiss_index.faiss") -> None:
    """Saves a FAISS index to a local file and uploads it to S3."""
    index_path = filename

    try:
        # Save the index locally
        faiss.write_index(index, index_path)
        print(f"FAISS index saved locally to {index_path}")

        # Upload the index to S3
        s3_key = f"{config['index_folder']}/{filename}"  # Key for the S3 object
        s3.upload_file(index_path, config["bucket_name"], s3_key)
        print(f"FAISS index uploaded to S3: s3://{config['bucket_name']}/{s3_key}")

    except Exception as e:
        print(f"Error saving and uploading FAISS index: {e}")

def load_faiss_index_s3(config: Dict, filename: str = "faiss_index.faiss") -> faiss.Index:
    """Downloads a FAISS index from S3 to a local file and loads it into memory."""
    index_path = filename

    try:
        # Download the index from S3
        s3_key = f"{config['index_folder']}/{filename}"
        s3.download_file(config["bucket_name"], s3_key, index_path)
        print(f"FAISS index downloaded from S3: s3://{config['bucket_name']}/{s3_key} to {index_path}")

        # Load the index
        index = faiss.read_index(index_path)
        print("FAISS index loaded into memory.")
        return index

    except Exception as e:
        print(f"Error downloading and loading FAISS index: {e}")
        return None

def query_faiss_index(index: faiss.Index, query_vector: np.ndarray, k: int = 5) -> List[int]:
    """Retrieves relevant vectors from a FAISS index."""
    D, I = index.search(query_vector.reshape(1, -1).astype('float32'), k)  # Assuming query_vector is a numpy array
    return I.tolist()[0]  # Returns the indices of the top k nearest neighbors

def load_faiss_index_with_config(config:Dict) -> faiss.Index:

    # Update the config
    CONFIG["bucket_name"] = bucket_name_widget.value
    CONFIG["index_folder"] = index_folder_widget.value
    CONFIG["embedding_dimension"] = int(embedding_dimension_widget.value)
    CONFIG["embedding_model"] = embedding_model_widget.value
    CONFIG["nprobe"] = int(nprobe_widget.value)

    #Load FAISS here.
    faiss_index = load_faiss_index_s3(CONFIG, filename = CONFIG["index_file"])

    # If loading the FAISS index fails, stop execution.
    if faiss_index is None:
        print("Failed to load FAISS index from S3. Exiting...")
        return None
    else:
        return faiss_index

def load_dataframe_from_csv(config:Dict, filename:str):

    # Update the config
    CONFIG["bucket_name"] = bucket_name_widget.value
    CONFIG["csv_folder"] = csv_folder_widget.value

    # Load data from CSV files
    dataframe = load_data_from_s3(CONFIG, filename)

    return dataframe

# Define DSPy Signatures
class RationaleSignature(dspy.Signature):
    """Generate rationale for a query based on given sources."""
    query: str = dspy.InputField()
    sources: list = dspy.InputField()
    ranked_response: str = dspy.OutputField()
    rationale: str = dspy.OutputField()

class ChainOfThoughtQASignature(dspy.Signature):
    """Answer questions about a document using Chain of Thought."""
    query: str = dspy.InputField()
    context: str = dspy.InputField()
    reasoning_steps: str = dspy.OutputField()
    final_answer: str = dspy.OutputField()

# Load data from CSV files
print("Loading Data")
sd_reports_df = load_dataframe_from_csv(CONFIG, "sd_reports_metadata.csv")
sd_details_df = load_dataframe_from_csv(CONFIG, "sd_details_metadata.csv")
faiss_index = load_faiss_index_with_config(CONFIG)

if sd_reports_df is None or sd_details_df is None or faiss_index is None:
    print("Error: Could not load data or FAISS index. Check configurations and S3 access.")
else:

    # Display input/selection widgets
    report_number_widget = widgets.IntSlider(description="Report #", min=0, max=len(sd_reports_df)-1, value=0)
    num_relevant_docs_widget = widgets.IntSlider(description="# Relevant Docs", min=1, max=5, value=1)

    display(report_number_widget, num_relevant_docs_widget)

    query_widget = widgets.Textarea(description="Example Query:", placeholder="Enter your query about report...")
    qa_answer_output_widget = widgets.Textarea(description="Final Answer:", placeholder="Answers will appear here")
    rationale_output_widget = widgets.Textarea(description="Generated Rationale:", placeholder="Reasoning will appear here")
    ranked_response_output_widget = widgets.Textarea(description="Ranked response:", placeholder="Rank will appear here")

    display(query_widget)
    display(qa_answer_output_widget)
    display(rationale_output_widget)
    display(ranked_response_output_widget)

    example_insight_button_output_widget = widgets.Button(description="What was the best example")
    display (example_insight_button_output_widget)

    # Define DSPy Modules
    class RationaleModel(dspy.Module):
        def __init__(self):
            super().__init__()

        @dspy.predict(RationaleSignature)
        def __call__(self, query, sources):
            return RationaleSignature(query=query, sources=sources)

    class ChainOfThoughtQAModel(dspy.Module):
        def __init__(self):
            super().__init__()

        @dspy.predict(ChainOfThoughtQASignature)
        def __call__(self, query, context):
            return ChainOfThoughtQASignature(query=query, context=context)

    # Load and initialize the FAISS index
    def getTopRecommendations(example_index:int) -> List[int]:

        #Example from SD Details
        example_insight = sd_reports_df['key_insights'][example_index]
        try:
            query_vector = np.array(generate_embedding(example_insight)).reshape(1,-1)
        except Exception as e:
            print ("Problem connecting to bedrock, please investigate " + str(e))
            return None

        index = faiss_index

        # Ensure the data type is float32 before passing it to the FAISS index
        query_vector = query_vector.astype(np.float32)
        index.nprobe = int(CONFIG["nprobe"]) #Ensure proper dimensions to load
        return query_faiss_index(index, query_vector, k = 5)


    def update_text_components(button):
        global rationale_model, cot_qa_model

        try:

            #Get top index from current configuration
            top_results = getTopRecommendations(report_number_widget.value)

            # Example Usage with Data
            example_number = report_number_widget.value
            num_relevant_docs = num_relevant_docs_widget.value

            #Get best document.
            example_insight = sd_reports_df['key_insights'][example_number]
            ranked = ""
            for i in top_results[:num_relevant_docs]:
                ranked = ranked + str(sd_details_df['key_insights'][i]) + "\n===\n"

            relevant_project = ranked

            # Initialize Model (with updated value)
            rationale_model = dspy.Predict(RationaleSignature)
            cot_qa_model = dspy.Predict(ChainOfThoughtQASignature)

            #Chain of Thought
            CoTModel = ChainOfThoughtQAModel()

            #Use to make results
            results= CoTModel(query=query_widget.value, context=relevant_project)
            QA = results.__dict__["final_answer"]
            example_rationale = results.__dict__["reasoning_steps"]

            #print (f"This was result " + str (QA.__dict__))
            try:
                ranked_response = str (relevant_project)
                rationale = example_insight

                #Print the results
                qa_answer_output_widget.value = str (QA) #str(prediction.final_answer)
                rationale_output_widget.value = str (rationale)
                ranked_response_output_widget.value = str (example_rationale)
            except Exception as f:
                print ("There was an issue formatting the output due to " +str (f))

        except Exception as fe:
            print ("Please check S3 settings, there may be something corrupted here. Error is " +str (fe))

    example_insight_button_output_widget.on_click(update_text_components)

    teleprompter_trainset = []
    process_button = widgets.Button(description="Run Validation")

    def runTelePrompter(b):

        if use_teleprompter_widget.value:
            #Load data
            df = load_data_from_s3(CONFIG, "sd_reports_metadata.csv")

            # Example: create simple training data
            train_data_range = min (10, len(df))
            examples = [dspy.Example(query=df['key_insights'][example_id], answer="Correct insight") for example_id in range(train_data_range)]

            #Check which to use
            global rationale_model, cot_qa_model

            print (f"Using BootStrap FewShot to compile RationaleModel")
            teleprompter = BootstrapFewShot(metric=None) #No metric needed

            #Create model
            compiled_qa = teleprompter.compile(ChainOfThoughtQAModel(), trainset=examples)
            print ("Running Example")
            try:
                newTestValue = compiled_qa(query="Explain the value of this work", context = "this project works").__dict__["final_answer"]
                qa_answer_output_widget.value = "New generated value"+ str (newTestValue)
                print ("The new test val" + newTestValue)

            except Exception as fe:
                print ("The error with testValue " + str (fe))

            print ("done! Good day")

        runTelePrompter(None)

    process_button.on_click(runTelePrompter)
    display(process_button)
