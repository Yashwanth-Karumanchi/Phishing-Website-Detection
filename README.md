# Phishing Websites Detection

Hello, people!

With advancements in tech, phishing attacks on the internet have also risen. So, here's an attempt to weed out those websites!

This is a Phishing Website Detector that combines the traditional ML route with Image Comparisons using Image Embeddings to predict if a website is phishing or not.

The ML Logistic Regression model works with an accuracy of 97% and continuously updates itself upon use. So, basically, the user is the data provider for the model. How cool is that, being able to control a model! This project utilizes Image Embeddings of VGG16 and VGG19 to compute cosine similarity scores with the already calculated scores of legitimate websites and provides a similarity score, determining if it is a phishing website or not.

Well, how many websites does it compare with? The default number of websites is around 2000, but upon usage, even the embeddings and websites get updated whenever a legitimate website is predicted. Hence, the application can be scaled almost infinitely! If similarity is > 75%, then it may be a phishing website.

How to run it? Well, that's pretty easy.

  **PREREQUISITES**
1. Clone or download the repository
2. ENSURE PYTHON IS INSTALLED ON YOUR SYSTEM! If not the app will not open no matter how many times you click on it.
3. Double click on the install_requirements file to install all the dependencies. An alternate would be to open the command prompt in the directory where you have this repository, and execute 'pip install -r requirements.txt', and your dependencies are installed!
   
  **USING THE APPLICATION**
1.  To use the application, simply double click on the 'app' file to open the website. The initial load time is around 1 to 2 minutes so please excuse the app for the delay.
2.  To check if any website is legitimate or not, enter the url of the suspicious website in the first text field, and viola, there's our predictions.
3.  If you feel that the suspicious website is mimicking a website, you can also enter the legitimate website along with the suspicious website in the second text field, and that can give you results of comparisions between both of those websites too. This will also update the datasets and models, contributing to better performance of the application over time.

--ALSO...this is just an experimental approach we have achieved, so please feel free to use/upgrade this work!--

And there's that. This is our take on Phishing Website Detection using ML and Image Embeddings. For people who are interested in this domain, feel free to use our work for reference or even develop on it. Thank you.