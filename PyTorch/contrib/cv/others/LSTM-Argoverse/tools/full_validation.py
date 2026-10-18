"""Every official validation trajectory, identical selected CPU/NPU model weights."""
import torch
from torch.utils.data import DataLoader
from common import write_json,sha256
import model_task as task


def validate_all(model,cpu,dataset,device,metadata,npu_metrics,output):
    maximum=0.0;violations=elements=sequences=0
    model.eval();cpu.eval()
    with torch.no_grad():
        for batch in DataLoader(dataset,batch_size=512,shuffle=False,drop_last=False):
            actual=task.predict(model,batch,device).cpu()
            expected=task.predict(cpu,batch,torch.device('cpu'))
            if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
                raise RuntimeError('Nonfinite complete validation trajectory prediction.')
            maximum=max(maximum,float((actual-expected).abs().max()))
            violations+=int((~torch.isclose(actual,expected,rtol=.003,atol=.003)).sum())
            elements+=actual.numel();sequences+=len(batch[0])
    cpu_metrics=task.evaluate(cpu,dataset,torch.device('cpu'),512,metadata)
    paired_npu_metrics=task.evaluate(model,dataset,device,512,metadata)
    passed=sequences==39472 and elements==39472*30*2 and violations==0
    report={'model':'lstm-argoverse','result':'pass' if passed else 'fail',
        'checkpoint_sha256':sha256(output/'checkpoint.pt'),
        'data_scope':metadata['scope'],'validation_sequences':sequences,
        'coordinate_elements_compared':elements,'prediction_rtol':.003,'prediction_atol':.003,
        'normalized_prediction_max_absolute_error':maximum,'elements_outside_tolerance':violations,
        'cpu_city_coordinate_metrics':cpu_metrics,'npu_city_coordinate_metrics':paired_npu_metrics,
        'paired_metric_batch_size':512,
        'metrics_units':'meters after the original coordinate inverse; miss rate threshold 2m, K=1',
        'validation_used_for_checkpoint_selection':True,'independent_test_or_leaderboard_claimed':False,
        'formal_business_accuracy_claimed':False}
    write_json(output/'full_prediction_control.json',report)
    if not passed:raise RuntimeError('Complete 39,472-trajectory CPU/NPU numerical validation failed; evidence preserved.')
    return report
