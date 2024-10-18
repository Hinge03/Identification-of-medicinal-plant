from flask import Flask, request, render_template, redirect, url_for
import os
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
with open('svm_model_1.pkl', 'rb') as file:
    best_estimator = pickle.load(file)

categories = ['Tulsi', 'Sandalwood', 'Rose_apple', 'Rasna', 'Pomegranate', 'Peepal', 'Parijata', 'Oleander',
              'Neem', 'Mint', 'Mexican_Mint', 'Mango', 'lemon', 'Karanda', 'Jasmine', 'Jamun', 
              'Jamaica_Cherry-Gasagase', 'Jackfruit', 'Indian_Mustard', 'Indian_Beech', 'Hibiscus', 
              'Guava', 'Fenugreek', 'Drumstick', 'Curry', 'Crape_jasmine', 'Betel', 'Basale', 
              'Arive-Dantu', 'Roxburgh_fig']

# Medical information dictionary
medical_info = {
    'Arive-Dantu': 'Also known as Amarnath, this plant is rich in fiber, low in calories, and fat-free, making it ideal for weight loss. It helps treat ulcers, diarrhea, throat swelling, and high cholesterol. Additionally, it contains antioxidants.',

    'Basale': 'Basale has anti-inflammatory and wound-healing properties. Its leaves can be crushed and applied to burns, scalds, and wounds for faster healing. It also serves as an effective first aid remedy.',

    'Betel': 'Betel leaves have therapeutic potential, helping to cure mood swings, depression, and improve digestion by neutralizing stomach pH imbalances. They also contain anti-microbial agents that combat oral bacteria.',

    'Crape_jasmine': 'Jasmine is used to treat liver diseases like hepatitis, abdominal pain from diarrhea, and skin diseases. Its fragrance improves mood, reduces stress, and curbs food cravings. Jasmine also aids in wound healing.',


    'Curry': 'Curry leaves are nutrient-rich, aiding digestion, treating nausea, diarrhea, and lowering cholesterol. They promote hair growth and combat deficiencies in vitamins and minerals, while also reducing side effects of chemotherapy.',

    'Drumstick': 'Drumstick is high in Vitamin C and antioxidants, enhancing the immune system and reducing the risk of infections. Its bioactive compounds promote heart health and strong bones through lower artery thickening and high calcium content.',


    'Fenugreek': 'Fenugreek, known as Methi, aids in metabolic conditions like diabetes and regulates blood sugar, while also serving as an effective antacid for heartburn. Its high nutritional value and low calories make it a beneficial food for preventing obesity.',


    'Guava': 'Guava fruit is a delicious source of Vitamin C and antioxidants, helping prevent infections and treat conditions like hypertension, fever, and liver problems. Its health benefits include combating gastrointestinal, respiratory, oral, and skin infections.',


    'Hibiscus': "Hibiscus flower tea is commonly used to lower blood pressure, relieve dry coughs, and may help with fever, diabetes, gallbladder attacks, and cancer. Additionally, the plant's roots can be prepared into a tonic.",


    'Indian_Beech': 'Karanja, or Indian Beech, is a medicinal herb used for skin disorders, with oil applied to manage boils, rashes, eczema, and promote wound healing due to its antimicrobial properties. It also possesses anti-inflammatory activities, making it beneficial for arthritis.',


    'Indian_Mustard': 'Mustard and its oil are beneficial for relieving joint pain, swelling, and respiratory issues while also serving as a massage oil, skin serum, and hair treatment. Rich in monounsaturated fatty acids, mustard oil is a heart-healthy option that can be consumed. ',


    'Jackfruit': 'Jackfruits are rich in carotenoids, giving them their vibrant color and high vitamin A content. This nutrient aids in preventing heart diseases and eye issues, promoting excellent eyesight.',


    'Jamaica_Cherry-Gasagase': 'The Jamaican Cherry plant has anti-diabetic properties that may help manage type 2 diabetes and hypertension, while its tea promotes digestive health, boosts immunity, and aids in pain relief and infection prevention. Rich in nitric oxide, it relaxes blood vessels and enhances overall well-being.',

    'Jamun': 'The fruit extract of the Jamun plant treats colds, coughs, and flu, while the bark contains tannins and carbohydrates effective against dysentery. Additionally, Jamun juice alleviates sore throat issues and helps with spleen enlargement.',

    'Jasmine': 'Jasmine aids in curing liver diseases, alleviating abdominal pain from diarrhea, and improving mood while reducing stress and food cravings. It also helps combat skin diseases and accelerates wound healing.',

    'Karanda': 'Karanda is primarily used to treat digestive issues, worm infestations, gastritis, and splenomegaly, while also aiding respiratory infections like cough, cold, asthma, and tuberculosis. Its medicinal properties make it effective for various health conditions.',

    'lemon': 'Lemons are rich in Vitamin C and fiber, helping reduce heart disease risk and prevent kidney stones due to their citric acid content. Additionally, they enhance iron absorption, promoting overall health.',

    'Mango': 'Mango, the "King of Fruits," is rich in vitamins (C, K, A) and minerals like potassium and magnesium. Packed with antioxidants, it supports digestive and heart health, and may reduce cancer risk.',


    'Mexican_Mint': 'Mexican mint is a traditional remedy for respiratory issues like colds, sore throats, and congestion, with its leaves widely used for medicinal purposes. Additionally, it aids in natural skincare.',

    'Mint': "Mint not only helps combat bad breath but also relieves indigestion, upset stomach, and irritable bowel syndrome (IBS). It's rich in nutrients like Vitamin A, iron, manganese, folate, and fiber.",


    'Neem': 'Neem, a longstanding remedy, is known for treating skin diseases, boosting immunity, and acting as an insect repellent. It also alleviates joint pain and helps prevent gastrointestinal issues.',


    'Oleander': 'Oleander seeds and leaves can be used medicinally for various conditions, including heart issues and cancer, but should only be used under strict medical supervision due to their toxicity. Caution is essential, as oleander can be a deadly poison if misused.',


    'Parijata': 'The Parijata plant has anti-inflammatory and antipyretic properties, aiding in pain and fever management, and is used for laxative, rheumatism, skin ailments, and cough relief. Fresh juice from its leaves mixed with honey can help alleviate fever symptoms.',

    'Peepal': 'The bark of the Peeple tree, rich in vitamin K, is an effective complexion corrector that strengthens blood capillaries and reduces inflammation. It promotes faster healing of skin bruises, treats pigmentation, wrinkles, dark circles, and lightens scars and stretch marks.',


    'Pomegranate': "Pomegranates are rich in antioxidants, which reduce inflammation and may lower cancer risk while boosting immunity with their high Vitamin C content. They also show promise in stalling Alzheimer's progression and protecting memory.",


    'Rasna': 'The Rasna plant and its oil alleviate bone and joint pain, relieve symptoms of rheumatoid arthritis, and aid respiratory issues by clearing mucus. Additionally, it can be applied to wounds to promote healing.',


    'Rose_apple': "Rose apple seeds and leaves are used to treat asthma, fever, epilepsy, smallpox, and joint inflammation, while also enhancing brain health and cognitive abilities. Their active and volatile compounds possess anti-microbial and anti-fungal properties.",

    'Roxburgh_fig': 'Roxburgh fig is noted for its big and round leaves. Leaves are crushed and the paste is applied on the wounds. They are also used in diarrhea and dysentery.',


   'Sandalwood': "Sandalwood treats common ailments like colds, coughs, fever, and sore throat, as well as urinary tract infections, liver issues, and cardiovascular diseases. It's also beneficial for heatstroke, gonorrhea, headaches, and bronchitis.",


   'Tulsi': "The Tulsi plant is renowned for its medicinal properties, effectively treating ailments like fever, skin issues, and respiratory problems. It's widely used in traditional remedies for conditions such as heart disease and insect bites."
   }


# Define function for prediction with medical information
def image_classification_prediction(image_path):
    img = imread(image_path)
    img_resized = resize(img, (15, 15))
    img_flatten = img_resized.flatten()
    img_array = np.asarray(img_flatten)
    
    # Predict the category
    result = best_estimator.predict(img_array.reshape(1, -1))
    predicted_category = categories[result[0]]
    
    # Get medical information for the predicted category
    medical_details = medical_info.get(predicted_category, "No medical information available.")
    
    return predicted_category, medical_details

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')  # Ensure your HTML file is named 'index.html'

# Route to handle the image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if file is uploaded
        if 'myfile' not in request.files:
            return redirect(request.url)
        
        file = request.files['myfile']
        
        if file.filename == '':
            return redirect(request.url)
        
        # Save the uploaded file to the uploads directory
        upload_dir = 'static/uploads'  # Store files in the static directory
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        
        file_path = os.path.join(upload_dir, file.filename)
        file.save(file_path)

        # Call the prediction function with the saved image
        predicted_category, medical_details = image_classification_prediction(file_path)

        # Render result.html and pass the prediction result, filename, and medical information
        return render_template('result.html', prediction=predicted_category, medical_info=medical_details, filename=file.filename)

if __name__ == "__main__":
    app.run(debug=True)
