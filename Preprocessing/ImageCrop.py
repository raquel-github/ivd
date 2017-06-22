# Create the crops of the correct objects

from PIL import Image
import urllib
from DataReader import DataReader

def main():

    dr = DataReader('Data/preprocessed.h5', 'Data/indices.json', 'Data/Images', 'Data/image_features.h5')

    for i in dr.get_game_ids():

        print(i)

        url = dr.get_image_url(i)
        img = Image.open(dr.get_image_path(i))

        correct = dr.get_target_object(i)
        objects = dr.get_object_ids(i)
        bbox = dr.get_object_bbox(i)

        x = y = w = h = 0
        for j in range(len(objects)):
            if objects[j] == correct:
                x = bbox[j][0]
                y = bbox[j][1]
                w = bbox[j][2]
                h = bbox[j][3]

        area = (int(x), int(y), int(x + w), int(y + h))
        cropped_img = img.crop(area)
        cropped_img.save("Data/Images/Crops/"+str(i)+".jpg")


if __name__ == '__main__':
    main()
