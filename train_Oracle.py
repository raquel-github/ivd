#train_Oracle

from Preprocessing.DataReader import DataReader
from Models.oracle import Oracle

def train():
    max_iter = 1000
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
    model = Oracle(vocab_size, embedding_dim, hidden_dim, d_in, d_hin, d_hidden, d_hout, d_out)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    #Get Game/Question and run model
    gameids = dr.get_game_ids()
    for epoch in range(max_iter):
        for gid in gameids:
            image = dr.get_image_features(gid)
            crop = dr.get_crop_features(gid) 
            correct = dr.get_target_object(gid)

            objects = dr.get_object_ids(i)
            object_class = get_category_id(gid)
            spatial = dr.get_image_meta(gid) 
            for j, obj in enumerate(objects):
                if obj == correct:
                    #print("found correct object")
                    spatial = spatial[j]
                    spatial = img_spatial(spatial)
                    object_class = object_class[j]

            quas = dr.get_questions(gid)
            answers = dr.get_answers(gid)
            for qi,question in enumerate(quas):
                outputs = model.forward(question, spatial, object_class, crop, image)

                answer = answers[qi]
                cost = loss(outputs,answer)
                print(cost.data[0])
    
                # Backpropogate Errors 
                optimizer.zero_grad() 
                cost.backward()
                optimizer.step()

if __name__ == '__main__':
    train()
