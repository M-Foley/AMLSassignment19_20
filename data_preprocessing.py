import sys
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Read the image, reshape and then flatten
def prep_img(img_path, rows, columns):
    img = cv2.imread(img_path,cv2.IMREAD_COLOR)
    img = cv2.resize(img, (rows, columns))
    return(pd.Series(img.flatten()))

# Create the train dataset tasks:A1, A2, B2
def create_train_set(image_list, labels, data, data_min):
    
    counter_train = 0
    y_train = list()
    X_train = pd.DataFrame(np.zeros((int(len(data)), 64 * 64 * 3)))
    for img in image_list:
        X_train.iloc[counter_train, :] = prep_img(img, 64, 64) / 255
        counter_train += 1
    
    y_train = data.iloc[:,1].values.tolist()
    return X_train, y_train

# Create the test dataset tasks:A1, A2, B2
def create_test_set(image_list, labels, data, data_min):
    counter_test = 0
    X_test = pd.DataFrame(np.zeros((int(len(data)), 64 * 64 * 3)))
    y = list()
    for img in image_list:
        X_test.iloc[counter_test, :] = prep_img(img, 64, 64) / 255
        counter_test += 1

    y = data.iloc[:,1].values.tolist()
    return X_test, y

# Get the labels of the data used for the multiclass task in B1 to correspond to neural network output nodes
def get_labels(data):
    labels=[]
    for label in data:
        if label == 0: labels.append(np.array([1, 0, 0, 0, 0]))
        elif label == 1 : labels.append(np.array([0, 1, 0, 0, 0]))
        elif label == 2 : labels.append(np.array([0, 0, 1, 0, 0]))
        elif label == 3 : labels.append(np.array([0, 0, 0, 1, 0]))
        elif label == 4 : labels.append(np.array([0, 0, 0, 0, 1]))
    return np.array(labels)

# Read the image used in task B1
def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) 
    return cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)

# Prepare the images including reshape and return, used in task B1
def prep_data(images):
    count = len(images)
    data = np.ndarray((count, 3, 64, 64), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image.T
    
    return data
    
# Create the test and train datasets    
def create_dataset(labels, dataset, path, train_data, test_data, data_min):
    
    img_path = path + '/Datasets/dataset_AMLS_19-20/'+dataset+'/img/' # path of image folder
    train_image_name = [img_path + each for each in train_data['img_name'].values.tolist()]
    test_image_name = [img_path + each for each in test_data['img_name'].values.tolist()]

    if 'face_shape' in train_data.columns:
    	X_train = prep_data(train_image_name)
    	y_train = get_labels(train_data.iloc[:,1].values.tolist())
    	X_test = prep_data(test_image_name)
    	y_test = get_labels(test_data.iloc[:,1].values.tolist())
    	return X_train, y_train, X_test, y_test
    X_train, y_train = create_train_set(train_image_name, labels, train_data, data_min)
    X_test, y_test = create_test_set(test_image_name, labels, test_data, data_min)
    return X_train, y_train, X_test, y_test

# Function to get the binary classification data used in tasks A1 and A2
def get_binary_data(classifier, data_set):
    data_one = data_set[data_set[classifier] == 1]
    data_two = data_set[data_set[classifier] == 0]
    train_one_data, test_one_data = train_test_split(data_one)
    train_two_data, test_two_data = train_test_split(data_two)
    test_indices = test_one_data.index.tolist() + test_two_data.index.tolist()
    test_data = data_set.iloc[test_indices,:]
    train_data = pd.concat([data_set, test_data, test_data]).drop_duplicates(keep=False)
    
    return train_data, test_data



# Function to get the multiclass data used in tasks B1 and  B2
def get_multiclass_data(classifier, data):
    data_min = 0
    if np.amin(data[classifier].value_counts())%2==0: data_min = np.amin(data[classifier].value_counts())
    else : data_min = np.amin(data[classifier].value_counts()) -1
    zero_data = data[data[classifier]==0]
    zero_data = zero_data[-data_min:]

    one_data = data[data[classifier]==1]
    one_data = one_data[-data_min:]

    two_data = data[data[classifier]==2]
    two_data = two_data[-data_min:]

    three_data = data[data[classifier]==3]
    three_data = three_data[-data_min:]

    four_data = data[data[classifier]==4]
    four_data = four_data[-data_min:]

    train_zero_data, test_zero_data = train_test_split(zero_data)
    train_one_data, test_one_data = train_test_split(one_data)
    train_two_data, test_two_data = train_test_split(two_data)
    train_three_data, test_three_data = train_test_split(three_data)
    train_four_data, test_four_data = train_test_split(four_data)
    
    
    test_indices = test_zero_data.index.tolist() + test_one_data.index.tolist() \
                    + test_two_data.index.tolist()+ test_three_data.index.tolist() \
                    + test_four_data.index.tolist()

    test_data = data.iloc[test_indices,:]

    train_indices = train_zero_data.index.tolist() + train_one_data.index.tolist() \
                    + train_two_data.index.tolist()+ train_three_data.index.tolist() \
                    + train_four_data.index.tolist()
    train_data = data.iloc[train_indices,:]
    
    return train_data, test_data, data_min


# Main function to get all the data from the files 
def get_data():
    path = str(sys.path[0])
    celeb_labels = pd.read_csv(path + '/Datasets/dataset_AMLS_19-20/celeba/labels.csv', delimiter='\t')
    celeb_labels = celeb_labels.iloc[:,1:4]
    celeb_dict = {1: 1, -1: 0}
    celeb_labels["smiling"].replace(celeb_dict, inplace=True)
    celeb_labels["gender"].replace(celeb_dict, inplace=True)

    smile_labels = celeb_labels.drop('gender', axis=1)
    gender_labels = celeb_labels.drop('smiling', axis=1)

    train_smile_data, test_smile_data = get_binary_data('smiling', smile_labels)
    train_gender_data, test_gender_data = get_binary_data('gender', gender_labels)
    smile_X_train, smile_y_train, smile_X_test, smile_y_test = create_dataset(smile_labels, 'celeba', path, train_smile_data, test_smile_data, smile_labels.shape[0])
    gender_X_train, gender_y_train, gender_X_test, gender_y_test = create_dataset(gender_labels, 'celeba', path, train_gender_data, test_gender_data, gender_labels.shape[0])

    cartoon_labels = pd.read_csv(path + '/Datasets/dataset_AMLS_19-20/cartoon_set/labels.csv', delimiter='\t')
    cartoon_labels = cartoon_labels.iloc[:,1:4]
    face_labels = cartoon_labels.drop('eye_color', axis=1)
    eye_labels = cartoon_labels.drop('face_shape', axis=1)
    columns_faces = ["file_name","face_shape"]
    columns_eyes = ["file_name","eye_color"]
    face_labels = face_labels.reindex(columns=columns_faces).rename(columns={'file_name':'img_name'})
    eye_labels = eye_labels.reindex(columns=columns_eyes).rename(columns={'file_name':'img_name'})

    train_eye_data, test_eye_data, data_min = get_multiclass_data('eye_color', eye_labels)
    train_face_data, test_face_data, data_min = get_multiclass_data('face_shape', face_labels)
    eye_X_train, eye_y_train, eye_X_test, eye_y_test = create_dataset(eye_labels, 'cartoon_set', path, train_eye_data, test_eye_data, data_min)
    face_X_train, face_y_train, face_X_test, face_y_test = create_dataset(face_labels, 'cartoon_set', path, train_face_data, test_face_data, data_min)

    return smile_X_train, smile_y_train, smile_X_test, smile_y_test, gender_X_train, gender_y_train, gender_X_test, gender_y_test, eye_X_train, eye_y_train, eye_X_test, eye_y_test, face_X_train, face_y_train, face_X_test, face_y_test
