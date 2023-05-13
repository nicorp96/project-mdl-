from google_images_download import google_images_download 
   
instance = google_images_download.googleimagesdownload()                # Call of the Mehtod

# Define a list with the Keywords to download images from Google 
search_pictures = [ 'Conor McGregor',                                   # Key Word for Positive
                    'UFC Fighters',                                     # Keyword for Negativ
                    'Conor McGregor']                                   # Keyword for Anchor
   
def get_images(keyword):                                                  # Function to Download the Pictures
    # Dictionary with the specified properties of the wanted images
    download_input = {    "keywords":  keyword,                           
                            "format":  "jpg",
                             "limit":  100,
                        "print_urls":  False,
                              "size":  "medium",
                      "aspect_ratio":  "panoramic"}

    instance.download(download_input)
         
for item in search_pictures:
    get_images(item) 
    print() 