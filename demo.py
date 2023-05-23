from preprocess.dataset import Dataset
from models.MEM import Model
from evaluation.metric import Metric
from tqdm import tqdm

if __name__ == "__main__":
    # testing dataset
    dst_test = Dataset(type="test", year=2015)
    test_data = dst_test.get_all_item()
    target_pred = []
    target_gt = []
    
    # get prediction
    model = Model()
    for pair in tqdm(test_data):
        source = pair["en"]
        pred = model.translate(source)
        target_pred.append(pred)
        target_gt.append(pair["zh"])
    
    # calculate metric
    metric = Metric()
    _, avg = metric.eval(pred=target_pred, 
                            target=target_gt)
    print('avg BLUE-1:', avg)