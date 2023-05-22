from preprocess.dataset import Dataset
from models.MEM import Model
from evaluation.metric import Metric

if __name__ == "__main__":
    # testing dataset
    dst_test = Dataset(type="test", year=2015)
    source = dst_test.get_item(10)['en']
    target_gt = dst_test.get_item(10)['zh']
    
    # get prediction
    model = Model()
    target_pred = model.translate(source)
    
    # calculate metric
    print(source)
    print(target_pred)
    print(target_gt)

    metric = Metric()
    BLUE, avg = metric.eval(pred=[target_pred], 
                            target=[target_gt])
    print(BLUE)
    print(avg)