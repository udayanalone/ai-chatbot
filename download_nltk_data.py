import nltk

def download_nltk_data():
    print("Downloading NLTK data...")
    
    # Download punkt tokenizer models
    try:
        nltk.download('punkt', quiet=False)
        print("Successfully downloaded punkt")
    except Exception as e:
        print(f"Error downloading punkt: {e}")
    
    # Download wordnet
    try:
        nltk.download('wordnet', quiet=False)
        print("Successfully downloaded wordnet")
    except Exception as e:
        print(f"Error downloading wordnet: {e}")
    
    # Download averaged_perceptron_tagger
    try:
        nltk.download('averaged_perceptron_tagger', quiet=False)
        print("Successfully downloaded averaged_perceptron_tagger")
    except Exception as e:
        print(f"Error downloading averaged_perceptron_tagger: {e}")
    
    # Download stopwords
    try:
        nltk.download('stopwords', quiet=False)
        print("Successfully downloaded stopwords")
    except Exception as e:
        print(f"Error downloading stopwords: {e}")
    
    # Download punkt_tab
    try:
        nltk.download('punkt_tab', quiet=False)
        print("Successfully downloaded punkt_tab")
    except Exception as e:
        print(f"Note: punkt_tab might not be available. Error: {e}")
        print("Trying alternative approach...")
        try:
            # Try alternative approach for punkt_tab
            import urllib.request
            import os
            
            # Create nltk_data directory if it doesn't exist
            nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
            punkt_tab_dir = os.path.join(nltk_data_dir, 'tokenizers', 'punkt_tab', 'english')
            
            os.makedirs(punkt_tab_dir, exist_ok=True)
            
            # Download punkt_tab pickle file
            url = 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab/english.pickle'
            urllib.request.urlretrieve(url, os.path.join(punkt_tab_dir, 'english.pickle'))
            print("Successfully downloaded punkt_tab using alternative method")
        except Exception as e2:
            print(f"Alternative approach also failed: {e2}")

if __name__ == "__main__":
    download_nltk_data()
