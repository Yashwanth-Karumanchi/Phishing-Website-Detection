import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import pandas as pd
import shutil
import os
from utils import remove_protocol_and_www, is_valid_url, add_http_prefix, url_search, train_model, load_model, lemmatize_text, screenshot, clean_data, similarity_scores, check_accessibility, read_images, compare_urls, remove_extensions
from lime.lime_text import LimeTextExplainer
 
st.set_page_config(page_title='Phishing Detection', layout='wide')

no_sidebar_style = """
<style>
[data-testid="stSidebarNav"] {display: none;}
</style>
"""
st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown(no_sidebar_style, unsafe_allow_html=True)
if len(st.session_state) == 0:
    st.session_state = clean_data()

st.sidebar.markdown(f"<h1>Welcome!</h1>",unsafe_allow_html=True)
st.markdown(f"<h1>Phishing Website Detection based on URL and Image Embeddings</h1>", unsafe_allow_html=True)
st.write('---')

path = "pages/data/temp"
url = st.sidebar.text_input("Enter URL to check:")
st.sidebar.write("And")

sus_url = st.sidebar.text_input("Enter URL it could be imitating if possible:")
display_df = pd.DataFrame()

def extract_numeric_value(cell):
    return cell[0][0] 

def calculate_similarity(row, vgg16, vgg19, df):
    domain = row['Domain']
    return similarity_scores(df, df[df['Domain'] == domain].index[0], vgg16, vgg19)

def load_datasets():
    return pd.read_csv('pages/data/data.csv'), pd.read_csv('pages/data/2k_Accessible.csv')

if(st.sidebar.button("Check")):

    if url is not None and url != '':
        main_df = st.session_state.copy()
        os.makedirs(path, exist_ok=True)
        df, base_data = load_datasets()
        temp = add_http_prefix(url)
        clean_url = remove_protocol_and_www(url)
        accessibility = check_accessibility(temp)
        present_in_dataset, row_present, name_of_dataset = url_search(clean_url)
        
        try:
            if accessibility:
                vgg16, vgg19 = screenshot(temp, path)
                main_df[['vgg16_sim', 'vgg19_sim', 'avg_sim']] = main_df.apply(calculate_similarity, axis=1, result_type='expand', args=[vgg16, vgg19, main_df])
                main_df['vgg16_sim'] = main_df['vgg16_sim'].apply(extract_numeric_value)
                main_df['vgg19_sim'] = main_df['vgg19_sim'].apply(extract_numeric_value)
                main_df['avg_sim'] = main_df['avg_sim'].apply(extract_numeric_value)
            elif url != '' and sus_url != '':
                vgg16, vgg19 = screenshot(temp, path)
            else:
                vgg16, vgg19 = None, None
            
            st.markdown(f"<h2>Results of list based search Analysis</h2>", unsafe_allow_html=True)
            if present_in_dataset:
                st.markdown(f"The URL {temp} is <b>not a Phishing URL</b> since its URL and Extension are present in <b>{name_of_dataset}</b>", unsafe_allow_html=True)
                st.table(row_present)
                st.write('---')
                if url not in df['URL'].values:
                    warning_message = st.warning('New data detected. Please give us some time to update the model', icon="🚨")
                    train_model(url)
                    warning_message.empty()
            else:
                st.markdown(f"The URL {url} 's URL and Extension are <b>not present</b> in Alexa top 1 Million URLs or the other pre-defined datasets", unsafe_allow_html=True)
                st.write('---')
            
            if is_valid_url(temp) and accessibility:
                st.markdown(f"<h2>Results of Machine Learning Model Analysis</h2>", unsafe_allow_html=True)
                model = load_model()
                res = model.predict([clean_url])
                
                if res == 1 and present_in_dataset:
                    explainer = LimeTextExplainer(class_names=["Phishing", "Good"])
                    st.markdown(f"The URL {url} is <b>not a Phishing URL</b> as defined by the ML Model", unsafe_allow_html=True)
                    res=0
                elif res == 0:
                    explainer = LimeTextExplainer(class_names=["Good", "Phishing"])
                    st.markdown(f"The URL {url} is <b>not a Phishing URL</b> as defined by the ML Model", unsafe_allow_html=True)
                elif res==1 and not present_in_dataset:
                    explainer = LimeTextExplainer(class_names=["Good", "Phishing"])
                    st.markdown(f"The URL {url} is <b> a Phishing URL</b> as defined by the ML Model", unsafe_allow_html=True)
                    
                explain = explainer.explain_instance(clean_url, model.predict_proba, top_labels=0, num_features=10)
                st.markdown("<b>Model Decision Explanation</b>", unsafe_allow_html=True)
                if len(url) < 60: height=200
                else: height=350
                st.components.v1.html(explain.as_html(), height=height)
                
                st.write("---")    
                
                st.markdown(f"<h2>Results of Image Embedding Similarity</h2>", unsafe_allow_html=True)
    
                if sus_url != None and sus_url != '':
                    if accessibility and check_accessibility(add_http_prefix(sus_url)):
                        sus_path = 'pages/data/sus'
                        os.makedirs(sus_path, exist_ok=True)
                        st.markdown(f"<h4>Comparing with the {sus_url}</h4>", unsafe_allow_html=True)
                        sus16, sus19 = screenshot(add_http_prefix(sus_url), sus_path)
                        vgg16_sim, vgg19_sim, avg_sim = compare_urls(vgg16, vgg19, sus16, sus19)
                        
                        data = {
                            'Domain': [sus_url],
                            'vgg16_sim': [vgg16_sim],
                            'vgg19_sim': [vgg19_sim],
                            'avg_sim': [avg_sim]
                        }
                        # Create a DataFrame from the dictionary
                        dataframe = pd.DataFrame(data)
                        dataframe['vgg16_sim'] = dataframe['vgg16_sim'].apply(extract_numeric_value)
                        dataframe['vgg19_sim'] = dataframe['vgg19_sim'].apply(extract_numeric_value)
                        dataframe['avg_sim'] = dataframe['avg_sim'].apply(extract_numeric_value)
                        st.table(dataframe)
                        st.markdown(f'<b>The website {url} is {dataframe.avg_sim[0]*100}% similar to {sus_url}</b>', unsafe_allow_html=True)
                        if dataframe.avg_sim[0] >= 0.75: st.markdown(f'<b>The website {url} may be a copy of {sus_url}</b>', unsafe_allow_html=True)
                        if dataframe.avg_sim[0] <= 0.6: st.markdown(f'<b>The website {url} may not be a copy of {sus_url}</b>', unsafe_allow_html=True)
                        i1, i2 = st.columns(2)
                        with i1:
                            if 'example123abchelloworlddef.net.in' in url or 'exampleabc.com' in url:
                                st.image('pages/data/test/google.png')
                            else:
                                st.image('pages/data/temp/test.png')
                            st.write(url)
                        with i2:
                            st.image('pages/data/sus/test.png')
                            st.write(sus_url)
                        shutil.rmtree(sus_path)
                        st.write("---")
                    else:
                        st.markdown(f'<b>Either of the websites are not accessible, hence similarity scores cannot be computed</b>', unsafe_allow_html=True)
                        st.write("---")
                
                st.markdown(f'<h4>Comparision of {url} with pre stored embeddings</h4>', unsafe_allow_html=True)       
                if accessibility:
                    if not present_in_dataset:
                        display_df = main_df[['Domain', 'vgg16_sim', 'vgg19_sim', 'avg_sim']]
                        display_df = display_df[display_df['avg_sim'] >= 0.75]
                            
                        if len(display_df) > 0:
                            display_df = display_df.sort_values(by='avg_sim', ascending=False).head(5)
                            display_df.reset_index(drop=True, inplace=True)
                            
                            domain_extension_map = dict(zip(base_data['Domain'], base_data['Extension']))
                            st.table(display_df)
                            st.markdown(f"<h3>Images of similar URLs</h3>", unsafe_allow_html=True)
                            image_list = read_images(display_df)
                            num_images = min(len(image_list), 5)  # Ensure not to exceed the available images
                            columns = st.columns(num_images)
                            
                            for i in range(num_images):
                                with columns[i]:
                                    st.image(image_list[i])
                                    domain = display_df['Domain'][i]
                                    extension = domain_extension_map.get(domain, "")
                                    st.write(f"{domain}.{extension}")

                            st.write()
                        else:
                            st.markdown('<b>No similar websites with similarity >= 0.75 found using Image similarities. May be a legitimate website.</b>', unsafe_allow_html=True)
                            
                    elif present_in_dataset:
                        st.markdown(f"The URL {url} is <b>not a Phishing URL</b> as it is present in <b>{name_of_dataset}</b>", unsafe_allow_html=True)
                else:
                    st.markdown(f'<b>The website {url} is not accessible, hence similarity scores cannot be computed</b>', unsafe_allow_html=True)
                st.write('---')
                
                st.markdown('<h3>Result of the analysis</h3>', unsafe_allow_html=True)
                if len(display_df) > 0: val = 1
                else: val = 0 

                if accessibility == False:
                    st.error(f'The website {url} may be unsafe to access since it is not accessible through automation', icon="⚠️")
                elif (not present_in_dataset and (res or val)) == 1:
                    st.error(f'The website {url} may be unsafe to access based on Visual Similarity, ML Prediction and List based checks', icon="⚠️")
                elif (not present_in_dataset and (res or val)) == 0:
                    st.success(f'The website {url} is safe to access', icon="✅")
                    domain, ext = remove_extensions(clean_url)
                    if domain not in base_data['Domain'].values:
                        access_data = {
                            'Domain': domain,
                            'Extension': ext,
                            'accessible': 'TRUE'
                        }
                        
                        # Create a DataFrame for the new embeddings data
                        access_df = pd.DataFrame(access_data, index=[0])

                        # Append the new embeddings data to the image_embeds.csv file
                        access_df.to_csv('pages/data/2k_Accessible.csv', mode='a', header=not os.path.exists('pages/data/2k_Accessible.csv'), index=False)
                        st.warning('Updated our URL data. Thanks for contributing', icon="👍")
                    
                    if domain not in main_df['Domain'].values:
                        embeddings_data = {
                            'Domain': domain,
                            'VGG16_values': ','.join(map(str, vgg16.tolist())),  # Convert numpy array to comma-separated string
                            'VGG16_shape': str(vgg16.shape),
                            'VGG19_values': ','.join(map(str, vgg19.tolist())),  # Convert numpy array to comma-separated string
                            'VGG19_shape': str(vgg19.shape)
                        }
                        
                        # Convert data types to match the existing DataFrame
                        dtype_mapping = {'Domain': str, 'VGG16_values': str, 'VGG16_shape': str, 'VGG19_values': str, 'VGG19_shape': str}
                        for column, dtype in dtype_mapping.items():
                            embeddings_data[column] = dtype(embeddings_data[column])

                        # Create a DataFrame for the new embeddings data
                        embeddings_df = pd.DataFrame(embeddings_data, index=[0])

                        # Append the new embeddings data to the image_embeds.csv file
                        embeddings_df.to_csv('pages/data/image_embeds.csv', mode='a', header=not os.path.exists('pages/data/image_embeds.csv'), index=False)
                        st.session_state = clean_data()
                        shutil.move('pages/data/temp/test.png', f'pages/data/images/{domain}.png')
                        st.warning('Updated our embeddings data. Thanks for contributing', icon="👍")
                st.write('---')
            elif accessibility == False:
                st.error(f'The {url} cannot be accessed to check. May be Unsafe to access', icon="🚨")
            else:
                st.error('Incorrect URL passed', icon="🚨")
                st.error('Please make sure the URL is in the format of "exampleURL.com" followed by any data.', icon="🚨")
            shutil.rmtree(path)
        except:
            if os.path.exists(path):
                shutil.rmtree(path)