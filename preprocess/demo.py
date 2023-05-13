from dataset import Dataset

if __name__ == "__main__":
    # training dataset
    dst_train = Dataset(type="train")
    print(dst_train.get_data_size())
    print(dst_train.get_item(300))

    # testing dataset
    dst_test = Dataset(type="test", year=2015)
    print(dst_test.get_data_size())
    print(dst_test.get_item(200))

    # validation dataset
    dst_val = Dataset(type="validate")
    print(dst_val.get_data_size())
    print(dst_val.get_item(100))
