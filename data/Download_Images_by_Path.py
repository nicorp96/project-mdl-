import wget
import os 

# Set up a List with the image URLs
image_urls = [ "https://i0.wp.com/www.actumma.com/wp-content/uploads/2018/09/Conor-mcgregor.jpg?resize=1024%2C683&ssl=1", 
               "https://heavy.com/wp-content/uploads/2020/08/GettyImages-839138756-e1596321454287.jpg?quality=65&strip=all", 
               "https://www.lesoir.be/sites/default/files/dpistyles_v2/ena_16_9_extra_big/2020/11/20/node_338940/27809690/public/2020/11/20/B9725296557Z.1_20201120082257_000+G2JH3FPB9.2-0.jpg?itok=36mDCebP1605857001", 
                "https://i0.wp.com/pxsports.com/wp-content/uploads/2017/08/conormcgregor1_get2.jpg?fit=1795%2C1192&ssl=1", 
                "https://www.techkee.com/wp-content/uploads/2016/04/459b7da22a.png",
                "https://heavy.com/wp-content/uploads/2020/01/gettyimages-1046918366-e1589246113108.jpg?quality=65&strip=all", 
                "https://mma.uno/wp-content/uploads/2019/03/Conor-McGregor-8.jpg", 
                "https://nypost.com/wp-content/uploads/sites/2/2021/07/conor-mcgregor-500.jpg?quality=90&strip=all", 
                "https://th.bing.com/th/id/OIP.D2xioovm6cfS59b1WtQkiQHaLF?pid=ImgDet&rs=1", 
                "https://sports-images.vice.com/images/2015/07/16/conoce-a-conor-mcgregor-el-campen-de-peso-pluma-del-ufc-body-image-1437037864.jpg", 
                "https://www.thesun.ie/wp-content/uploads/sites/3/2018/10/NINTCHDBPICT000439380770-e1540240784710.jpg", 
                "https://wallpapercave.com/wp/wp5294248.jpg", 
                "https://d3h7g948tee6ho.cloudfront.net/wp-content/uploads/2017/08/connor-ufc.jpg"               
]

#########################################################################################
save_path = os.chdir(r'./data/Data_McGregor/Train/')                                    # Set the Current Working Directory where the Images shall be saved
#########################################################################################
counter = 0                                                                             # Set a Counter to check more easy if there are Images which can not be Downloaded
for image in image_urls: 
    try: 
        image_download = wget.download(image)                                           # Try to download the Image
        print("\tImage", counter, "was succesfully downloaded!")                        # Short Output for the User to detect problems
    
    except: 
        print("Image", counter, "can not be downloaded!")                               # Output if an Image can not be downloaded
    
    counter += 1                                                                        # Increase the Counter
########################################################################################

image_list = os.listdir(save_path)                                                      # Get every item of the Current Directory
counter = 0                                                                             # Reset the Counter 
for image in image_list:                                                                # For every Item in the Counter 

    if image[-4:] == ".jpg" or image[-4:] == ".png":                                    # Check if the Item is an Image 
        os.rename(image, "data_"+str(counter)+".jpg")                                   # Rename the Image 
        counter += 1
