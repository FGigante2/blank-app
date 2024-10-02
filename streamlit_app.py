import os, time, random, re
import streamlit as st
from gpt4all import GPT4All
from st_draggable_list import DraggableList
import numpy as np
import pandas as pd
import scipy.stats as stats
from fuzzywuzzy import process
 

# Initialize session state for page tracking
if 'page' not in st.session_state:
    st.session_state.page = 1
if 'llm_response_generated' not in st.session_state:
    st.session_state.llm_response_generated = False
if 'llm_response' not in st.session_state:
    st.session_state.llm_response = ""
if 'updated_list' not in st.session_state:
    st.session_state.updated_list = []  # Initialize as empty for now
    

def calculate_weight_of_advice(final_estimate, initial_estimate, advice):
    numerator = abs(final_estimate - initial_estimate)
    denominator = abs(advice - initial_estimate)
    
    # Handle division by zero
    if denominator == 0:
        return None
    
    weight_of_advice = numerator / denominator
    print(f"Calcolato : {weight_of_advice}")
    return weight_of_advice

# Simulated function for generating LLM response
def simulate_llm_response(prompt, duration=5):
        st.write("LLM is being called now...")
        model_name = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
        model_path = os.path.expanduser("~/Library/Application Support/nomic.ai/GPT4All")
        model_file_path = os.path.join(model_path, model_name)
        model = GPT4All(model_name=model_name, model_path=model_path, allow_download=False)

        # Generate a response based on the user's ranked list
        if os.path.exists(model_file_path):
    
            response = model.generate(prompt, max_tokens = 2048)
            st.text_area("LLM Response:", value=response, height=800)
            st.session_state.llm_response = response
            st.session_state.llm_response_generated = True
            progress_bar = st.progress(0)  # Create a progress bar widget
            for i in range(1, 101):
                time.sleep(duration / 100)  # Simulate LLM processing time
                progress_bar.progress(i)  # Increment progress
            return "Simulated LLM response based on the user's ranked list."

def extract_ai_ranking(llm_response):
    # Use regex to find patterns like "1. **Item Name**"
    pattern = r"\d+\.\s*\*\*(.*?)\*\*"
    matches = re.findall(pattern, llm_response)

    if matches:
        # Create a dictionary with the AI ranking, extracting only item names
        ai_ranking = {item.strip(): idx + 1 for idx, item in enumerate(matches)}
        return ai_ranking
    else:
        pattern = r"\d+\.\s*([^\n]+?)\s*[\n-]"  # This handles both dash or en-dash as a separator
        matches = re.findall(pattern, llm_response)
        ai_ranking2 = {item.strip(): idx + 1 for idx, item in enumerate(matches)}
        return ai_ranking2

# Function to go to the next page
def next_page():
    st.session_state.page += 1
    st.rerun()

# Function to go to the previous page (optional)
def prev_page():
    if st.session_state.page > 1:
        st.session_state.page -= 1
        st.rerun()


 # Normalize AI ranked list to match NASA ranking using fuzzy matching
def get_best_match(item, choices, threshold=80):

        # Find the best match with a similarity score above the threshold
        best_match, score = process.extractOne(item, choices)
        if score >= threshold:
            return best_match
        return None
# Page 1: Model Initialization ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
if st.session_state.page == 1:

    model_name = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
    model_path = os.path.expanduser("~/Library/Application Support/nomic.ai/GPT4All")
    model_file_path = os.path.join(model_path, model_name)
    
    st.write("Looking for model file at:", model_file_path)

  
    # User input and response generation
    st.title("Lista dei 15 oggetti più importanti da portare sulla luna")
    
    st.text("")

    # Change page on button click
    if st.button("Next"):
        next_page()

# Page 2: Draggable List ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
elif st.session_state.page == 2:
    st.title("Lista dei 15 oggetti più importanti da portare sulla luna")
    st.write("Tieni premuto un oggetto e trascinalo per ordinarlo.\nOrdina gli oggetti secondo la tua personale opinione")

    # List of items
    items = [
        {"id": 1, "name": "Box of matches"},
        {"id": 2, "name": "Food concentrate"},
        {"id": 3, "name": "50 feet of nylon rope"},
        {"id": 4, "name": "Parachute silk"},
        {"id": 5, "name": "Portable heating unit"},
        {"id": 6, "name": "Two .45 caliber pistols"},
        {"id": 7, "name": "One case of dehydrated milk"},
        {"id": 8, "name": "Two 100 lb. tanks of oxygen"},
        {"id": 9, "name": "Stellar map"},
        {"id": 10, "name": "Self-inflating life raft"},
        {"id": 11, "name": "Magnetic compass"},
        {"id": 12, "name": "20 liters of water"},
        {"id": 13, "name": "Signal flares"},
        {"id": 14, "name": "First aid kit, including injection needle"},
        {"id": 15, "name": "Solar-powered FM receiver-transmitter"}
    ]

    # Create a draggable list
    draggable_list = DraggableList(items, key="draggable_list")

    # Change page on button click
    if st.button("Next"):
        st.session_state.user_list = draggable_list
        next_page()

# Page 3: AI Response ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
elif st.session_state.page == 3:
    
    
    # Prepare the user's ranked list for the LLM
    user_ranked_list = ", ".join([item['name'] for item in st.session_state.user_list])
    #st.write("Your ranked list of items:")
    #st.text(user_ranked_list)
    messaggio_attesa = st.empty()
    messaggio_attesa.text("Stiamo inviando la tua risposta al LLM, perfavore attendere un minuto")

    if(st.session_state.llm_response_generated == False):

        #Setto la keyword per la modalità di risposta dell'AI
        keyword = "insecure"
        randomGenerator = random.seed(a=None , version=2 )
        tipoAI = random.randint(1,2)
        
        #if(tipoAI == 1 ):
        #    keyword = "assertive"
        #elif(tipoAI == 2):
        #    keyword = "insecure"
        
            
        #print(keyword)
        prompt = (
            f"Given the following list of items ranked by importance:\n{user_ranked_list}, "
            "Rank all of these items in order of importance in a moon expedition, explaining why each item is ranked as it is. "
            "Include only items i provided and and void duplicates"
            "Your ranking format should be : 1. Item Name " 
            f"You should explain the ranks in a very {keyword} way."
        )

        model_name = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
        model_path = os.path.expanduser("~/Library/Application Support/nomic.ai/GPT4All")
        model_file_path = os.path.join(model_path, model_name)
        model = GPT4All(model_name=model_name, model_path=model_path, allow_download=False)

        # Generate a response based on the user's ranked list
        if os.path.exists(model_file_path):
            
            response = model.generate(prompt, max_tokens = 3000)
            #response = example_response
            
            st.text_area("LLM Response:", value=response, height=800)
            st.session_state.llm_response = response
            st.session_state.llm_response_generated = True
            ai_ranking = extract_ai_ranking(st.session_state.llm_response)
            st.session_state.ai_ranking = ai_ranking
            st.write("AI-generated ranking:", ai_ranking)
            messaggio_attesa.empty()

    else:
        response = st.session_state.llm_response
        #response = example_response
        st.text_area("LLM Response:", value=response, height=800)
        st.write("AI-generated ranking:", st.session_state.ai_ranking)
    
    
   
    st.write("Adesso, in base alla risposta del LLM, puoi decidere se cambiare l'ordine nella tua lista o meno")
    new_draggable_list = DraggableList(st.session_state.user_list, key="list")
    
    
    if(st.button("Next")):
        st.session_state.updated_list = new_draggable_list  # Save the final updated list
        next_page()

# Page 4: Result analysis ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

elif st.session_state.page == 4:
    
    
    # NASA's ranking
    nasa_ranking = {
        "Box of matches": 15,
        "Food concentrate": 4,
        "50 feet of nylon rope": 6,
        "Parachute silk": 8,
        "Portable heating unit": 13,
        "Two .45 caliber pistols": 11,
        "One case of dehydrated milk": 12,
        "Two 100 lb. tanks of oxygen": 1,
        "Stellar map": 3,
        "Self-inflating life raft": 9,
        "Magnetic compass": 14,
        "20 liters of water": 2,
        "Signal flares": 10,
        "First aid kit, including injection needle": 7,
        "Solar-powered FM receiver-transmitter": 5
    }

    
    
    #Dizionari dei vari ranking
    user_ranking = {item['name']: idx + 1 for idx, item in enumerate(st.session_state.user_list)}
    user_ranking_afterai = {item['name']: idx + 1 for idx, item in enumerate(st.session_state.updated_list)}
    ai_ranked_list = st.session_state.ai_ranking
    normalized_ai_ranks_dictionary = {}
    

    
    #Converto i ranking in una lista (contiene solo i valori dei ranking ordinati per tipo di item)
    nasa_ranks = []
    user_ranks = []
    user_ranks_afterai = []
    ai_ranks = []
    
    #Normalizzo gli item generati dall'AI (potrebbe dare un output con nomi di item leggermente diversi da quelli forniti)
    normalized_ai_ranks = [] #----- nomi normalizzati
    nasa_items = list(nasa_ranking.keys()) 
    
    
    
    #Creo il dizionario normalizzato
    for item in ai_ranked_list:
        rank = ai_ranked_list[item]
        normalized_name = get_best_match(item.lower().strip(), nasa_items)
        if normalized_name:
            normalized_ai_ranks_dictionary[normalized_name] = rank
        else:
            normalized_ai_ranks.append(None)  # Handle unmatched cases
    
            
    ai_ranks = ai_ranks[:15]
    
    
    #Aggiungo i rank ordinati nelle liste - le liste servono per calcolare lo spearman rank correlation
    for item in nasa_ranking.keys():
        nasa_ranks.append(nasa_ranking[item])
        user_ranks.append(user_ranking[item])
        user_ranks_afterai.append(user_ranking_afterai[item])
        ai_ranks.append(normalized_ai_ranks_dictionary[item])

    
    #Calcolo il WoA
    sum=0
    den = 0
    for i in range (0,14):
        initial = user_ranks[i]
        final = user_ranks_afterai[i]
        advice = ai_ranks[i]
        if(calculate_weight_of_advice(final,initial,advice) != None):
            sum += calculate_weight_of_advice(final,initial,advice)
            den += 1
        
    # Calculate Spearman's Rank Correlation
    spearman_corr, _ = stats.spearmanr(user_ranks, nasa_ranks)
    spearman_corr_afterai, _ = stats.spearmanr(user_ranks_afterai, nasa_ranks)
    spearman_corr_ai, _ = stats.spearmanr(ai_ranks, nasa_ranks)

    print(sum)
    print(den)

    st.write(f"Spearman's Rank Correlation between User and NASA Ranking: {spearman_corr:.2f}")
    st.write(f"Spearman's Rank Correlation between User and NASA Ranking after ai: {spearman_corr_afterai:.2f}")
    st.write(f"Spearman's Rank Correlation between AI and NASA Ranking : {spearman_corr_ai:.2f}")
    st.write(f"weight of advice is {sum/den}")
    st.write(f"improvement is {((spearman_corr_afterai - spearman_corr) / (1 - spearman_corr)) * 100}")
    
    
    if spearman_corr_afterai == 1:
        st.write("Perfect match with NASA's ranking!")
    elif spearman_corr_afterai > 0.7:
        st.write("The user ranking is very similar to NASA's ranking.")
    elif spearman_corr_afterai > 0.4:
        st.write("The user ranking somewhat matches NASA's ranking.")
    else:
        st.write("The user ranking is quite different from NASA's ranking.")
        
        

    
        
