

def check_atlas_loss(item, thresh=0.01):
    metadata = item['annotation']['metadata'].get('metadata_atlas', None)
    loss = None
    if metadata is not None:
        loss = metadata.get('loss', None)
    return loss is None or loss < thresh


def check_nlf_mae(item, thresh=20):
    metadata = item['annotation']['metadata'].get('metadata_nlf', None)
    mae = None
    if metadata is not None:
        mae = metadata.get('mae', None)
    return mae is None or mae < thresh

def check_nlf_pve(item, thresh=110):
    metadata = item['annotation']['metadata'].get('metadata_nlf', None)
    pve = None
    if metadata is not None:
        pve = metadata.get('pve', None)
    return pve is None or pve < thresh
