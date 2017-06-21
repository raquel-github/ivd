#train_Oracle

from Preprocessing.DataReader import DataReader
from Models.oracle import oracle.py


def train():
    # Load data
    dr = DataReader()

    visual_len = 4096
    object_len = 4096 #TODO: check if true
    category_len = 1
    spatial_len = 4
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
    optimizer = optim.Adam(model.parameters())

    #Get Game/Question and run model
    gameids = dr.get_game_ids()
    for gid in gameids:
        image = dr.get_image_features(gid)
        #crop = dr.get_crop_features(gid) #TODOL get features of cropped images
        obj = get_target_object(gid)
        quas = dr.get_questions(gid)
        answers = dr.get_answers(gid)
        for qi,question in quas:
            outputs = model.forward(question, spatial_info, object_class, crop, image)
            answer = answers(qui)
            cost = loss(outputs,answer)

            # Backpropogate Errors ||TODO: also applies to LSTM?
            optimizer.zero_grad() 
            cost.backward()
            optimizer.step()

if __name__ == '__main__':
    train()
