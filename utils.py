import numpy as np
import pandas as pd
import os
import re
from nltk.tokenize import RegexpTokenizer # regexp tokenizers use to split words from text
from nltk.stem import WordNetLemmatizer
import pickle
import webbrowser
import requests
import time
import mss
import pyautogui

import tensorflow as tf
from tensorflow.keras.preprocessing import image

from sklearn.metrics.pairwise import cosine_similarity
import warnings # ignores pink warnings 
warnings.filterwarnings('ignore')

def read_images(df):
    path = 'pages/data/images'
    images_list = []
    
    # Iterate over the 'Domain' column in the DataFrame
    for domain in df['Domain']:
        image_path = os.path.join(path, domain +'.png')
        
        # Append to the images_list
        images_list.append(image_path)
    
    return images_list
    

def check_accessibility(url):
    if 'example123abchelloworlddef.net.in' in url or 'exampleabc.com' in url:
        return True
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:97.0) Gecko/20100101 Firefox/97.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'TE': 'Trailers',
        }
        
        # Use a session to handle cookies
        with requests.Session() as session:
            response = session.get(url, headers=headers, timeout=10)
        
        # Check the status code
        if response.ok:
            print(f"{url}: {response.status_code}")
            return True
        elif response.status_code in [401, 402, 404, 408, 456]:
            print(f"{url}: {response.status_code}")
            return False
        else:
            # Check if the response content is valid
            if response.content:
                print(f"{url}: {response.status_code}")
                return True
            else:
                print(f"{url}: {response.status_code}. No response content.")
                return False

    except requests.RequestException as e:
        print(f"Error opening homepage for {url}: {e}")
        return False

def compare_urls(vgg16, vgg19, sus16, sus19):
    array1 = vgg16.reshape(1, -1)
    array2 = vgg19.reshape(1, -1)
    array3 = sus16.reshape(1, -1)
    array4 = sus19.reshape(1, -1)
    
    # Calculate cosine similarity
    similarity1 = cosine_similarity(array1, array3)
    similarity2 = cosine_similarity(array2, array4)

    return similarity1, similarity2, (similarity1+similarity2)/2
    

def similarity_scores(df, idx, vgg16, vgg19):
    array1 = np.array(df['VGG16_values'][idx]).reshape(1, -1)
    array2 = np.array(df['VGG19_values'][idx]).reshape(1, -1)
    array3 = vgg16.reshape(1, -1)
    array4 = vgg19.reshape(1, -1)

    # Calculate cosine similarity
    similarity1 = cosine_similarity(array1, array3)
    similarity2 = cosine_similarity(array2, array4)

    return similarity1, similarity2, (similarity1+similarity2)/2

def embeddings(img_path):
        
    model_vgg1 = tf.keras.models.load_model('pages/model/vgg16.h5')
    model_vgg2 = tf.keras.models.load_model('pages/model/vgg19.h5')
    
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array_vgg1 = tf.keras.applications.vgg16.preprocess_input(img_array)
    # Extract features (embeddings) from the image
    features_vgg1 = model_vgg1.predict(img_array_vgg1)
    # Flatten the features to obtain a 1D vector
    vgg16_embeddings = features_vgg1.flatten()
    
    img_array_vgg2 = tf.keras.applications.vgg19.preprocess_input(img_array)
    # Extract features (embeddings) from the image
    features_vgg2 = model_vgg2.predict(img_array_vgg2)
    # Flatten the features to obtain a 1D vector
    vgg19_embeddings = features_vgg2.flatten()
    
    return vgg16_embeddings, vgg19_embeddings

def clean_data():
    df = pd.read_csv("pages/data/image_embeds.csv")
    df = df.drop_duplicates(subset=['Domain'], keep='first')
    df['VGG16_values'] = df['VGG16_values'].apply(lambda x: np.fromstring(x[1:-1], sep=',').astype(float))
    df['VGG19_values'] = df['VGG19_values'].apply(lambda x: np.fromstring(x[1:-1], sep=',').astype(float))
    names = os.listdir('pages/data/images')
    names = [name[:-4] if name.endswith('.png') else name for name in names]
    urls = df['Domain'].values.tolist()
    extras = set(urls) - set(names)

    mask = df['Domain'].isin(extras)
    df = df[~mask]
    return df

def screenshot(url, path):
    if 'example123abchelloworlddef.net.in' in url or 'exampleabc.com' in url:
        embed1, embed2 = embeddings('pages/data/test/test.png')
    else:
        webbrowser.open_new_tab(url)
        time.sleep(10)
        with mss.mss() as sct:
            screenshot = sct.shot(output=os.path.join(path, 'test.png'))

        pyautogui.hotkey('ctrl', 'w')
        embed1, embed2 = embeddings(os.path.join(path, 'test.png'))
    return embed1, embed2

def lemmatize_text(text):
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokenizer.tokenize(text)]

def load_model():
    loaded_model = pickle.load(open('pages/model/phishing.pkl', 'rb'))
    return loaded_model

def add_http_prefix(url):
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url  # Add the "http://" prefix if missing
    return url

def train_model(url):
    label = 0
    df = pd.read_csv('pages/data/data.csv')
    model = load_model()
    val = min(len(df[df['Label'] == 1]), len(df[df['Label'] == 0]))
    phishing_samples = df[df['Label'] == 1].sample(n=val, random_state=42)
    benign_samples = df[df['Label'] == 0].sample(n=val, random_state=42)
    phish_data = pd.concat([phishing_samples, benign_samples],axis=0)
    new_data = pd.DataFrame({'URL': [url], 'Label': [label]})
    phish_data = pd.concat([phish_data, new_data], ignore_index=True)
    phish_data['URL'] = phish_data['URL'].apply(remove_protocol_and_www)
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv('pages/data/data.csv', index=False)
    model.fit(phish_data['URL'], phish_data['Label'])
    pickle.dump(model,open('pages/model/phishing.pkl','wb'))

def url_search(url):
    df = pd.read_csv('pages/data/data_cleaned.csv')
    df1 = pd.read_csv('pages/data/data.csv')
    df2 = pd.read_csv('pages/data/2k_Accessible.csv')
    text, domain = remove_extensions(url)
    
    # For data_cleaned.csv
    mask = (df['Domain'] == text) & (df['Extension'] == domain)
    if mask.any():
        indices = df.index[mask]
        return True, df.iloc[indices], 'Alexa Top 1 Million URLs'
        
    # For data.csv
    mask = df1['URL'] == url
    if mask.any():
        indices = df1.index[mask]
        if df1.loc[indices, 'Label'].iloc[0] == 0:
            return True, df1.iloc[indices], 'Training Dataset'
    
    # For 2k_Accessible.csv
    mask = (df2['Domain'] == text) & (df2['Extension'] == domain)
    if mask.any():
        indices = df2.index[mask]
        print(indices)
        return True, df2.iloc[indices], 'Alexa Top 1 Million URLs'
    
    return False, None, None

    
def remove_extensions(url):
    parts = url.split('.')
    cleaned_url = parts[0]
    domain_part = '.'.join(parts[1:])
    domain_part = domain_part.split('/')[0]
    return cleaned_url, domain_part

def is_valid_url(url):
    # Regex pattern to match a URL
    url_pattern = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    url = url.strip()
    # Check if the URL matches the pattern
    return bool(re.match(url_pattern, url))

def remove_protocol_and_www(url):
    protocols = ['https://', 'http://']
    for protocol in protocols:
        if url.startswith(protocol):
            url = url[len(protocol):]

    if url.startswith('www.'):
        url = url[len('www.'):]

    return url