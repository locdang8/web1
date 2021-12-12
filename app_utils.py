import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def process_image(img_path):
    img = load_img(img_path, target_size=(30, 30))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

def predict_model(img_path, model):
    labels = ['20 km/h', '30 km/h', '50 km/h', '60 km/h', '70 km/h', '80 km/h', '80 km/h end', '100 km/h', 
             '120 km/h', 'No overtaking','No overtaking for tracks', 'Crossroad with secondary way', 
             'Main road', 'Give way', 'Stop', 'Road up', 'Road up for track', 'Brock','Other dangerous',
             'Turn left', 'Turn right', 'Winding road', 'Hollow road', 'Slippery road', 'Narrowing road',
             'Roadwork', 'Traffic light', 'Pedestrian', 'Children', 'Bike', 'Snow', 'Deer', 'End of the limits',
             'Only right', 'Only left', 'Only straight', 'Only straight and right', 'Only straight and left',
             'Take right', 'Take left', 'Circle crossroad', 'End of overtaking limit', 'End of overtaking limit for track']
    
    image = process_image(img_path)
    pred = np.argmax(model.predict(image), axis=1)
    prediction = f'It\'s a \"{labels[pred[0]]}\" sign!!'
    return prediction