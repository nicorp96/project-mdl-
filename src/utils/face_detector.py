import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
from matplotlib import pyplot as plt
import numpy as np
import cv2

class FaceDetectorCV:

    def __init__(self, directory, save_dir):
        self._face_class = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')   # Specify the Classifier 
        self._directory = directory                                                                               # Get the Directory to work in 
        self.storate_directory_base = save_dir                                                                    # Set the Directory to save the Extracted Faces
        self._img_height = 500                                                                                    # Resize the Image when it is called
        self._img_width = 500
        self.shape = (264,264)                                                                                    # Tuple to reshape the extracted Face


    def set_img_height(self, img_h):
        self._img_height = img_h
    

    def set_img_width(self, img_w):
        self._img_width = img_w


    def set_img_width(self, img_w):
        self._img_width = img_w


    def set_shape(self, shape):
        self.shape = shape


    def prep_images(self): 

        def convertToRGB(img): 
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                                             # Convert Image into a colored one 
                                                                                                    # This is only necessary if the image shall be shown
        
        subfolder = os.listdir(self._directory)                                                     # Returns List with ["0", "1", "Anchor"] in case of Training
                                                                                                    # Returns List with ["0", "1", "Test"] in case of Test
        for folder in subfolder: 
            directory_to_create = os.path.join(self._directory, folder)                             # Concant new Paths
            '''
            The following Lines need if the extracted Faces shall be saved in the 
            "data"_director as well. The Lines after the next Dogstring need to be
            umcommented for that as well!!
            ''' 
            # try: 
            #     os.mkdir(directory_to_create + "/" + folder.lower())                                # Create new Subfolder in Folder 
            # except:                                                                                 # Excpet it already exist
            #     pass 
            
            content = os.path.join(self._directory, folder)                                         # Concant the new Directory 
            images = os.listdir(content)                                                            # List all Images in that Directory 
            counter = 0                                                                             # Counter to Name the found Faces later
            for image in images:                                                                    # For every Image found in the single Folders
                
                if image[-3:] == ("jpg" or "png"):                                                  # Skip the Subfolders

                    image = os.path.join(content, image)                                            # Concant the paths 
                    self.read_img = cv2.imread(image)                                               # Read the Image 
                    self.img_read = cv2.resize(self.read_img, (self._img_height, self._img_width), 
                               interpolation = cv2.INTER_AREA)                                      # Resize the found face to 500 x 500 pixels
                    haar_face_cascade = self._face_class                                            # Define the Classifier 

                    self.faces = haar_face_cascade.detectMultiScale(self.read_img, 
                                                                    scaleFactor=1.1, 
                                                                    minNeighbors=5)                 # Detect the face 
                    
                    for (x, y, w, h) in self.faces:
                        cv2.rectangle(self.read_img, (x, y), (x+w, y+h), 
                                      (0, 255, 0), 2)                                               # Create Rectangle around it
                        face_extracted = self.read_img[y:y + h, x:x + w]                            # Get Depending Array wich depicts the detected Face
                        face_extracted = cv2.resize(face_extracted, self.shape, 
                                                    interpolation = cv2.INTER_AREA)                 # Resize the found face to 264 x 264 pixels 
                        '''
                        The following Lines need if the extracted Faces shall be saved in the 
                        "data"_director as well. The Lines after the previous Dogstring need to 
                        be uncommented as well. 
                        ''' 
                        # cv2.imwrite(content + "/" + images[0] + '/' + str(counter) + '.jpg', 
                        #             face_extracted)                                                 # Save that extracted Face
                        
                        '''
                        The following Lines need to be uncommented if the whole preprocessing shall 
                        be shown. (Images with the Extracted Faces will pop up)
                        '''
                        # cv2.imshow("Test", convertToRGB(self.read_img))                             # Show the detected Face
                        # cv2.waitKey(300)
                        # cv2.destroyAllWindows

                        # Store the Faces in "./src/utils" so the Jupyter Notebook can accesss it 
                        try: 
                            os.mkdir(self.storate_directory_base + "/" + folder)                    # Create a folder
                        except: 
                            pass                                                                    # Pass if the Folder already exists

                        if "Test_Test" in folder:                                                   # Check if the Folder is the "Test_Test" folder
                            try: 
                                os.mkdir(self.storate_directory_base + "/" + folder + "/0")         # If this is the case: 
                                                                                                    # create another Subfolder for all negativ Test Samples
                                os.mkdir(self.storate_directory_base + "/" + folder + "/1")         # create another Subfolder for all positive Test Samples    
                            except: 
                                pass

                            if image[-5] == "0":                                                    # If the last character of the name is a 0 
                                path_to_store = self.storate_directory_base + "/" + folder + "/0"   # then it is a Negativ Test Sample
                            
                            elif image[-5] == "1":                                                  # If the last character of the name is a 1
                                path_to_store = self.storate_directory_base + "/" + folder + "/1"   # then it is a Positiv Test Sample 

                        else: 
                            path_to_store = os.path.join(self.storate_directory_base, folder)       # If we are not in the "Test_Test" folder, no further Subfolders are needed

                        cv2.imwrite(path_to_store +  "/" + str(counter) + '.jpg',                   # Save the extracted Face in the given Path
                                    face_extracted)   
  
                    counter += 1                                                                    # Increase the counter by 1     
    
                else: 
                    pass 
    #################################################################################################

"""
The Code below is only necessary, if this File shall be run on it's own. 
This was nesessary in the beginning to check if the faces got extracted properly. 
"""
# #Syntax to Prepare Images for Training
folder = "./data/Data_McGregor/Train"
save_dir = "./src/utils/train_data"
face_detector_train = FaceDetectorCV(folder, save_dir)
face_detector_train.prep_images()

#Syntax to Prepare Images for Testing 
# folder_test = "./data/Data_McGregor/Test"
# save_dir_test = "./src/utils/test_data"
# face_detector_test = FaceDetectorCV(folder_test, save_dir_test)
# face_detector_test.prep_images()