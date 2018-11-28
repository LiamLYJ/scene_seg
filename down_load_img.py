from imutils import paths
import argparse
import requests
import cv2
import os
import glob


def download_img(url, save_path, size = 256):
    rows = open(url, "r")
    rows = rows.readlines()
    set_name = url.split('/')[-1].split('_')[-1].rstrip('.txt')
    total = 0
    save_folder = os.path.join(save_path, set_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for url in rows:
        url = url.rstrip('\n')
        try:
    		# try to download the image
            r = requests.get(url, timeout=60)

    		# save the image to disk
            p = os.path.sep.join([save_folder, "{}.jpg".format(
    			str(total).zfill(8))])
            f = open(p, "wb")
            f.write(r.content)
            f.close()

    		# update the counter
            print("[INFO] downloaded: {}".format(p))
            total += 1
        # handle if any exceptions are thrown during the download process
        except:
            print("[INFO] error downloading {}...skipping".format(url))


    # loop over the image paths we just downloaded
    for imagePath in paths.list_images(save_folder):
    	delete = False
    	try:
            image = cv2.imread(imagePath)
            if image is None:
                delete = True
            else:
                image = cv2.resize(image, (size, size))
                cv2.imwrite(imagePath, image)

    	except:
    		print("Except")
    		delete = True

    	# check to see if the image should be deleted
    	if delete:
    		print("[INFO] deleting {}".format(imagePath))
    		os.remove(imagePath)

def main():
    url_path = './urls'
    save_path = './images'
    urls = glob.glob(os.path.join(url_path, 'urls*.txt'))
    for url in urls:
        if 'green' in url:
            continue
        download_img(url, save_path)

if __name__ == '__main__':
    main()
