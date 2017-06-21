#train_Oracle

from Preprocessing.DataReader import DataReader
from Models.oracle import oracle.py

def img_spatial(img_meta):
    bbox            = img_meta[0]
    width           = img_meta[1]
    height          = img_meta[2]
    image_center_x  = width / 2
    image_center_y  = height / 2

    x_min = (min(bbox[0], bbox[2]) - image_center_x) / image_center_x
    y_min = (min(bbox[1], bbox[3]) - image_center_y) / image_center_y
    x_max = (max(bbox[0], bbox[2]) - image_center_x) / image_center_x
    y_max = (max(bbox[1], bbox[3]) - image_center_y) / image_center_y
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    w_box = x_max - x_min
    h_box = y_max - y_min
    
return Variable(torch.FloatTensor([x_min, y_min, x_max, y_max, x_center, y_center, w_box, h_box]))

def train():
    # Load data
    dr = DataReader()

    visual_len = 4096
    object_len = 4096 
    category_len = 1
    spatial_len = 8
    embedding_dim = 128
    word2index = dr.get_word2ind()
    vocab_size = len(word2index)

    #Settings LSTM
    hidden_dim = 128
    
    #Settings MLP
    d_in = visual_len + spatial_len + category_len + hidden_dim + object_len
    d_hin = (d_in+d_out)/2 
    d_hidden = (d_hin+d_out)/2
    d_hout = (d_hidden+d_out)/2
    d_out = 3

    #Instance of Oracle om LSTM en MLP te runnen?
    model = Oracle(embedding_dim,hidden_dim, d_in, d_hin, d_hidden, d_hout, d_out)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    #Get Game/Question and run model
    gameids = dr.get_game_ids()
    for gid in gameids:
        image = dr.get_image_features(gid)
        crop = dr.get_crop_features(gid) 
        obj_id = dr.get_target_object(gid)
        object_class = get_category_id(gid)[obj_id] 
        img_meta = dr.get_image_meta(gid) #TODO: uitzoeken waarvan de gereturnde bounding box is?
        spatial = img_spatial(img_meta) #[obj_id] TODO: is dat nodig?
        quas = dr.get_questions(gid)
        answers = dr.get_answers(gid)
        for qi,question in quas:
            outputs = model.forward(question, spatial_info, object_class, crop, image)
            answer = answers[qi]
            cost = loss(outputs,answer)
            print(cost.data[0])

            # Backpropogate Errors 
            optimizer.zero_grad() 
            cost.backward()
            optimizer.step()

if __name__ == '__main__':
    train()
