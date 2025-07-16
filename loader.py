from testcaseIILoader import loadTestcaseIIState
from newFormatLoader import loadNewFormatState
from diffSPHLoader import loadDiffSPHState

from util import getStyle

def loadState(inFile, key, configuration, device, dtype):
    style = getStyle(inFile)

    if style == 'newFormat':
        # print(f'Loading new format state for key: {key}')
        return loadNewFormatState(inFile, key, configuration, device, dtype)
    elif style == 'testcase_II':
        # print(f'Loading Testcase II state for key: {key}')
        return loadTestcaseIIState(inFile, key, configuration, device, dtype)
    elif style == 'diffSPH':
        return loadDiffSPHState(inFile, key, configuration, device, dtype)
    else:
        raise ValueError(f'Unknown style: {style}. Supported styles are "newFormat" and "testcaseII".')
    
import h5py
import torch
def loadEntry(entry, configuration, device = 'cpu', dtype = torch.float32):
    inFile = h5py.File(entry[0], 'r')
    key = entry[1]
    
    result = loadState(inFile, key, configuration, device, dtype)
    inFile.close()
    return result

def loadBatch(dataset, batch, configuration, device = 'cpu', dtype = torch.float32):
    priorStates, currentState, nextStates, domains, configs, = [], [], [], [], []
    for b in batch:
        entry = dataset[b]

        prior, current, nexts, dom, cfg = loadEntry(entry, configuration, device, dtype)
        priorStates.append(prior)
        currentState.append(current)
        nextStates.append(nexts)
        domains.append(dom)
        configs.append(cfg)
    
    return priorStates, currentState, nextStates, domains, configs