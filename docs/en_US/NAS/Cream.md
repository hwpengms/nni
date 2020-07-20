# Cream

## Introduction

## Reproduction Results




## Examples

[Example code](https://github.com/microsoft/nni/tree/master/examples/nas/cream)

```bash
# In case NNI code is not cloned. If the code is cloned already, ignore this line and enter code folder.
git clone https://github.com/Microsoft/nni.git

# build environments for searching.
pip install -r ./examples/nas/cream/requirements.txt

# search the best architecture
sh ./examples/nas/cream/supernet.sh

# test the accuracy of searched architectures.
sh ./examples/nas/cream/test.sh

# We provide 14M/42M/114M/285M/470M/600M pretrained models in [google drive](https://drive.google.com/drive/folders/1CQjyBryZ4F20Rutj7coF8HWFcedApUn2)

# To test different FLOPs of models, speicify `--model_selection`.
```

## Reference

### PyTorch

```eval_rst
..  autoclass:: nni.nas.pytorch.cdarts.CdartsTrainer
    :members:

..  autoclass:: nni.nas.pytorch.cdarts.RegularizedDartsMutator
    :members:

..  autoclass:: nni.nas.pytorch.cdarts.DartsDiscreteMutator
    :members:

..  autoclass:: nni.nas.pytorch.cdarts.RegularizedMutatorParallel
    :members:
```
